from Agent import Agent
from Buffer import Buffer
from OUnoise import OUNoise
import gym
import numpy as np
import tensorflow as tf
import time

""" --------------------------- Parameters --------------------------- """
agent = Agent(lr_actor=0.001,  # Learning rate of actor
              lr_critic=0.001,  # Learning rate of critic
              num_actions=4,  # Number of actions the agent can perform
              num_states=24,  # Number of state inputs
              gamma=0.99,  # Gamma coefficient / discount factor
              tau=0.005,  # Target network update parameter
              delay_frequency=2,  # Delay rate of actor update
              batch_size=100)  # Batch size for networks / buffer

noise_ = OUNoise(size=(1, 4),  # Size of noise output - matches action
                 seed=1,  # Seed for noise
                 mu=0.1,  # Parameters of OU-noise
                 theta=0.7,
                 sigma=0.7)

buffer = Buffer(buffer_size=100000,
                batch_size=agent.batch_size,
                num_action=agent.num_actions,
                num_states=agent.num_states)

num_episodes = 50000  # Number of episodes the agent does
begin_learning = agent.batch_size
tf.random.set_seed(88)  # Init seed for the noise
start_timestep = 100  # Number of time steps the agent behaves randomly
total_timestep = 0  # Defining total time step counter
""" ------------------------------------------------------------------ """

episodic_reward_list = []
average_reward_list = []
""" 
Creating the networks: 
    - One actor for the policy/actions
    - Two critics for current Q values
    - One actor target for the target actions
    - Two critic targets for the target Q values
"""
actor = agent.get_actor()
critic_1 = agent.get_critic()
critic_2 = agent.get_critic()
actor_target = agent.get_actor()
critic_target_1 = agent.get_critic()
critic_target_2 = agent.get_critic()

# Set weights of the target networks equal initially to the online ones
actor_target.set_weights(actor.get_weights())
critic_target_1.set_weights(critic_1.get_weights())
critic_target_2.set_weights(critic_2.get_weights())

"""
Compiling all of the networks, even target ones, that will
require soft update:
"""
opt_actor = tf.optimizers.Adam(learning_rate=agent.lr_actor)
opt_critic = tf.optimizers.Adam(learning_rate=agent.lr_critic)

actor.optimizer = opt_actor
critic_1.optimizer = opt_critic
critic_2.optimizer = opt_critic
actor_target.optimizer = opt_actor
critic_target_1.optimizer = opt_critic
critic_target_2.optimizer = opt_critic

env = gym.make('BipedalWalker-v3')
env.seed(88)
time_start = time.time()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# gpu not supported...

""" ---------------------------- Training ---------------------------- """
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
            if total_timestep < begin_learning:
                action = env.action_space.sample()
            else:
                action = tf.squeeze(actor(tf_state))
                # Add noise, clip, reshape
                # noise = tf.random.normal(shape=(1, 4), mean=0.1, stddev=0.2)
                noise = noise_.sample()
                noise = np.clip(noise, -0.5, 0.5)
                action += noise
                action = np.clip(action, -1, 1)
                action = np.reshape(action, newshape=(4, ))
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
            noise = tf.random.normal(shape=(1, 4), mean=0.0, stddev=0.1)
            # noise = noise_.sample()
            noise = np.clip(noise, -0.5, 0.5)
            next_action += noise
            next_action = np.clip(next_action, -1, 1)
            # next_action = tf.convert_to_tensor(next_action)

            with tf.GradientTape(persistent=True) as tape:
                # Target Q values via target critic networks (next state, next action)
                q1_ = critic_target_1([ns_batch, next_action])
                q2_ = critic_target_2([ns_batch, next_action])
                # Choose minimum from these two values for the double Q update rule
                q_ = tf.math.minimum(q1_, q2_)
                # Calculate actual Q value:
                y = r_batch + (agent.gamma * q_)

                # Current Q values via critic networks (state, action)
                q1 = critic_1([s_batch, a_batch])
                q2 = critic_2([s_batch, a_batch])

                # Loss is calculated as the sum of the MSE loss between target Q value and q1, q2
                q1_loss = tf.keras.losses.MSE(y, q1)
                q2_loss = tf.keras.losses.MSE(y, q2)
                q_loss = q1_loss + q2_loss

                # Optimize critic networks:
            critic_gradient = tape.gradient(q_loss, critic_1.trainable_variables)
            critic_1.optimizer.apply_gradients(
                zip(critic_gradient, critic_1.trainable_variables))

            critic_gradient = tape.gradient(q_loss, critic_2.trainable_variables)
            critic_2.optimizer.apply_gradients(
                zip(critic_gradient, critic_2.trainable_variables))
            del tape

            """ Delayed update of actor and target networks """
            if timestep % agent.delay_frequency == 0:

                # Update actor with gradient
                with tf.GradientTape() as tape:
                    actions = actor(s_batch)
                    critic_value = critic_1([s_batch, actions])
                    # print(critic_value)
                    # Used `-value` as we want to maximize the value given
                    # by the critic for our actions
                    actor_loss = -tf.math.reduce_mean(critic_value)
                actor_grad = tape.gradient(actor_loss, actor.trainable_variables)
                actor.optimizer.apply_gradients(
                    zip(actor_grad, actor.trainable_variables))

                # Update target critics: - soft update with tau
                new_weights_1 = []
                target_variables_1 = critic_target_1.weights
                for i, variable in enumerate(critic_1.weights):
                    new_weights_1.append(variable * agent.tau + target_variables_1[i] * (1 - agent.tau))
                critic_target_1.set_weights(new_weights_1)

                new_weights_2 = []
                target_variables_2 = critic_target_2.weights
                for i, variable in enumerate(critic_2.weights):
                    new_weights_2.append(variable * agent.tau + target_variables_2[i] * (1 - agent.tau))
                critic_target_2.set_weights(new_weights_2)

                # Update target actor:
                new_weights_3 = []
                target_variables_3 = actor_target.weights
                for i, variable in enumerate(actor.weights):
                    new_weights_3.append(variable * agent.tau + target_variables_3[i] * (1 - agent.tau))
                actor_target.set_weights(new_weights_3)

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
        if avg_reward > 300:
            break

    # Saving the model, graph in a new folder:
    agent.save_log(actor, critic_1, critic_2, actor_target, critic_target_1, critic_target_2, average_reward_list,
                   episodic_reward_list)
