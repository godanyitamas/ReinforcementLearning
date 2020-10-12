from Agent import Agent


agent = Agent(lr_actor=0.0005,  # Learning rate of actor
              lr_critic=0.0005,  # Learning rate of critic
              num_actions=2,  # Number of actions the agent can perform
              num_states=8,  # Number of state inputs
              gamma=0.99,  # Gamma coefficient / discount factor
              tau=0.005,  # Target network update parameter
              delay_frequency=2,  # Delay rate of actor update
              batch_size=100)  # Batch size for networks / buffer

actor = agent.get_actor()
critic = agent.get_critic()

if __name__ == '__main__':
    actor.save('actor.h5')
    critic.save('critic.h5')
