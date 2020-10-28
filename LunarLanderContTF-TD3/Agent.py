""" Twin delayed deep deterministic policy gradient agent: """
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import datetime
from tensorflow.keras import layers


class Agent:
    def __init__(self,
                 lr_actor,
                 lr_critic,
                 num_actions,
                 num_states,
                 gamma,
                 tau,
                 delay_frequency,
                 batch_size
                 ):
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.num_actions = num_actions
        self.num_states = num_states
        self.gamma = gamma
        self.tau = tau
        self.delay_frequency = delay_frequency
        self.batch_size = batch_size

    def get_actor(self):
        """ Input state --> outputs an action """
        inputs = layers.Input(shape=(self.num_states,), name='actor_input')
        out = layers.Dense(400, activation='relu', name='actor_fc1')(inputs)
        out = layers.Dense(300, activation='relu', name='actor_fc2')(out)
        output = layers.Dense(self.num_actions, activation='tanh', name='actor_output')(out)
        model = tf.keras.Model(inputs, output)
        return model

    def get_critic(self):
        """ Gets a state and action input, concatenates them, and returns Q value"""
        state_input = layers.Input(shape=(self.num_states), name='critic_input_s')
        # state_output = layers.Dense(100, activation='relu')(state_input)

        action_input = layers.Input(shape=(self.num_actions,), name='critic_input_a')
        # action_output = layers.Dense(100, activation='relu')(action_input)

        concat = layers.Concatenate()([state_input, action_input])
        # print(concat) -> Tensor("concatenate_3/concat:0", shape=(None, 28),
        # dtype=float32)

        c1 = layers.Dense(400, activation='relu', name='critic_fc1')(concat)
        c1 = layers.Dense(300, activation='relu', name='critic_fc2')(c1)
        c1 = layers.Dense(1, activation=None, name='critic_fc3')(c1)
        c1_model = tf.keras.Model([state_input, action_input], c1, name='critic_output')
        # print(c1_model.summary())
        return c1_model

    def save_log(self, actor, critic_1, critic_2, target_actor, target_critic_1, target_critic_2, avg_reward_list,
                 ep_reward_list):
        # Makes a new folder system with current time as name
        mydir = os.path.join(os.getcwd(), 'Data', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Weights')
        mydir2 = os.path.join(os.getcwd(), 'Data', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Models')
        os.makedirs(mydir2)
        os.makedirs(mydir)
        plotdir = os.path.join(os.getcwd(), 'Data', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        # Creates paths to the weights
        a_path = os.path.join(mydir, 'actor.h5')
        c1_path = os.path.join(mydir, 'critic_1.h5')
        c2_path = os.path.join(mydir, 'critic_2.h5')
        ta_path = os.path.join(mydir, 'target_actor.h5')
        tc1_path = os.path.join(mydir, 'target_critic1.h5')
        tc2_path = os.path.join(mydir, 'target_critic2.h5')
        a_model_path = os.path.join(mydir2, 'actor_model.png')
        c_model_path = os.path.join(mydir2, 'critic_model.png')
        print(mydir)

        # Saves weights for each network
        actor.save_weights(filepath=a_path)
        critic_1.save_weights(filepath=c1_path)
        critic_2.save_weights(filepath=c2_path)
        target_actor.save_weights(filepath=ta_path)
        target_critic_1.save_weights(filepath=tc1_path)
        target_critic_2.save_weights(filepath=tc2_path)

        # Save the plots for learning
        # Plotting graph
        # Episodes versus Avg. Rewards
        plt.figure()
        plt.style.use('ggplot')
        plt.plot(avg_reward_list, label='Utolsó 100 epizód átlagos jutalma')
        plt.plot(ep_reward_list, label='Epizódonkénti jutalom', alpha=0.5)
        plt.xlabel("Epizód")
        plt.ylabel("Jutalom")
        plt.title("TD3-LunarLanderContinuous-v2")
        plt.legend()
        # plt.show()
        plt.savefig(os.path.join(plotdir, 'avg_plot'))
