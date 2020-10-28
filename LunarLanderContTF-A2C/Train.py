from Agent import Agent
from Buffer import Buffer
from OUnoise import OUNoise
import gym
import numpy as np
import tensorflow as tf
import time

# =============================================================== #
#                           Parameters:
# =============================================================== #

agent = Agent(lr_actor=0.0001,      # Learning rate of actor
              lr_critic=0.0003,     # Learning rate of critic
              num_actions=2,        # Number of actions the agent can perform
              num_states=8,         # Number of state inputs
              gamma=0.99,           # Gamma coefficient / discount factor
              batch_size=64)        # Batch size for networks / buffer

noise_ = OUNoise(size=(1, 2),       # Size of noise output - matches action
                 seed=2,            # Seed for noise
                 mu=0,              # Parameters of OU-noise
                 theta=0.15,
                 sigma=0.2)

buffer = Buffer(buffer_size=1000000,
                batch_size=agent.batch_size,
                num_action=agent.num_actions,
                num_states=agent.num_states)

env = gym.make('LunarLanderContinuous-v2')
env.seed(88)
num_episodes = 2500                # Number of episodes the agent does
tf.random.set_seed(88)              # Init seed for the noise
total_timestep = 0                  # Defining total time step counter

episodic_reward_list = []           # Counts the reward for one episode
average_reward_list = []            # Counts avg of last 100 episodes
time_start = time.time()            # For elapsed time estimation

# =============================================================== #
#                           Initializing:
# =============================================================== #

actor = agent.get_actor()           # Actor
critic = agent.get_critic()         # Critic

opt_actor = tf.optimizers.Adam(learning_rate=agent.lr_actor)
opt_critic = tf.optimizers.Adam(learning_rate=agent.lr_critic)

actor.optimizer = opt_actor
critic.optimizer = opt_critic

# GPU support:
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# =============================================================== #
#                             Training:
# =============================================================== #
if __name__ == "__main__":

    for ep in range(1, num_episodes + 1):

        # Reset environment after each run
        episodic_reward = 0.
        timestep = 0
        state = env.reset()

        while True:

            # env.render()
            # Select an action with actor/policy:
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            # Explore / Exploit
            if total_timestep < agent.batch_size:
                action = env.action_space.sample()
            else:
                action = tf.squeeze(actor(tf_state))
                # Add noise, clip, reshape
                # noise = tf.random.normal(shape=(1, agent.num_actions), mean=0.0, stddev=0.1)
                noise = noise_.sample()
                noise = np.clip(noise, -0.5, 0.5)
                action += noise
                action = np.clip(action, -1, 1)
                action = np.reshape(action, newshape=(agent.num_actions,))
            # Perform action, and get new information:
            new_state, reward, done, info = env.step(action)
            # Save reward:
            episodic_reward += reward
            # Store new values in buffer:
            action = np.squeeze(action)
            buffer.record((state, action, reward, new_state))
            # Update state with the new one:
            state = new_state

            """ Update / Learn """
            # Sample from the buffer:
            s_batch, a_batch, r_batch, ns_batch = buffer.batch_sample()

            s_batch = tf.convert_to_tensor(s_batch)
            a_batch = tf.convert_to_tensor(a_batch)
            r_batch = tf.convert_to_tensor(r_batch)
            ns_batch = tf.convert_to_tensor(ns_batch)

            # Select action according to the actor/policy:
            next_action = actor(ns_batch)
            next_action = np.clip(next_action, -1, 1)

            with tf.GradientTape(persistent=True) as tape:
                # Q(s',a') by critic
                q_ = critic([ns_batch, next_action])

                # Target q value using: r + gamma * Q(s',a')
                y = r_batch + (agent.gamma * q_)

                # Q(s,a) by critic
                q = critic([s_batch, a_batch])
                # Time difference would be: delta = y - q
                # Loss is calculated as the MSE of (y, q)
                q_loss = tf.keras.losses.MSE(y, q)

                # Optimize the critic network:
            critic_gradient = tape.gradient(q_loss, critic.trainable_variables)
            critic.optimizer.apply_gradients(
                zip(critic_gradient, critic.trainable_variables))
            del tape

            # Update actor with gradient
            with tf.GradientTape() as tape:
                actions = actor(s_batch)
                critic_value = critic([s_batch, actions])
                actor_loss = -tf.math.reduce_mean(critic_value)
            actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
            actor.optimizer.apply_gradients(
                zip(actor_grad, actor.trainable_variables))
            del tape

            timestep += 1
            total_timestep += 1
            if done:
                break

        # Bookkeeping:
        episodic_reward_list.append(episodic_reward)
        avg_reward = np.mean(episodic_reward_list[-100:])
        average_reward_list.append(avg_reward)
        s = int(time.time() - time_start)
        print(
            "Episode -- {} \t Elapsed Time -- {:02}:{:02}:{:02} \t Timestep -- {} \t Total Timesteps -- {} \t"
            " Avg Reward -- {} \t Reward -- {}".format(ep, s // 3600, s % 3600 // 60, s % 60, timestep,
                                                       total_timestep, avg_reward.round(decimals=1),
                                                       episodic_reward))

    # Saving the model, graph in a new folder:
    agent.save_log(actor, critic, average_reward_list, episodic_reward_list)
