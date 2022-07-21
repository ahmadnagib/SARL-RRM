__author__ = "Ahmad Nagib"
__version__ = "1.0.0"

from tensorforce import Agent, Environment
import numpy as np


class GenericAgent(object):

    def __init__(self, agent_package, agent_algorithm):
        super(GenericAgent, self).__init__()
        self.package = agent_package
        self.algorithm = agent_algorithm   

    def get_action(self, observation):
        pass


class TensorforceAgent(GenericAgent):
    def __init__(self, agent_algorithm, env, batch_size, memory, exploration, decay_rate, loading, num_steps, device, learning_rate, final_value, random_seed, discount_factor):
        # hold the current agent name
        # instantiates a Tensorforce agent for the passed env
        # and using the algorithm
        # these are the basic parameters but other can be configured
        # for random and constant agents, remove batch_size and memory
        self.agent_package = "tensorforce"
        self.agent_algorithm = agent_algorithm
        if ((loading == 'no') and (agent_algorithm != 'random') and (agent_algorithm != 'constant') and (agent_algorithm != 'custom_ppo_2')):


            if (agent_algorithm == 'ppo') or (agent_algorithm == 'a2c') or (agent_algorithm == 'ac') \
            or (agent_algorithm == 'ddqn') or (agent_algorithm == 'dueling_dqn') or (agent_algorithm == 'dqn') \
            or (agent_algorithm == 'trpo') or (agent_algorithm == 'reinforce') or (agent_algorithm == 'vpg'):
                self.agent = Agent.create(
                    environment=env,
                    agent = agent_algorithm,
                    network = dict(type= "auto", rnn = False),
                    memory = memory,
                    batch_size =  batch_size,
                    state_preprocessing = "linear_normalization",
                    config = dict(seed=random_seed)
                    )

        elif(agent_algorithm == 'random'):
            self.agent = Agent.create(
                environment=env,
                agent = agent_algorithm,
                )

    # takes an action using the created agent
    def get_action(self, observation):
        action = self.agent.act(states=observation)
        return action

    

