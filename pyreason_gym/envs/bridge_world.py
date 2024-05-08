
import gym
from gym import spaces

from pyreason_gym.pyreason_bridge_world.pyreason_bridge_world import PyReasonBridgeWorld
from pyreason_gym.simulator_other.other_simulator import OtherSimulator


class BridgeWorldEnv(gym.Env):

    def __init__(self, pyreason_simulator=True, graph=None, rules=None):

        super(BridgeWorldEnv, self).__init__()

        self.pyreason_simulator = pyreason_simulator

        # Initialize the PyReason gridworld
        self.pyreason_bridge_world = PyReasonBridgeWorld(graph, rules)
        self.other_simulator = OtherSimulator()

        self.observation_space = spaces.Dict(
            {
                'costs': spaces.Sequence(spaces.Dict()),
                'slots': spaces.Sequence(spaces.Dict()),
                'slots_available': spaces.Dict({'h1': spaces.Discrete(n=7), 'h2': spaces.Discrete(n=7), 'h3': spaces.Discrete(n=7)}),
                'blocks_available': spaces.Dict({'red-vertical': spaces.Discrete(n=11), 'red-horizontal': spaces.Discrete(n=11),
                                                 'green-vertical': spaces.Discrete(n=11), 'green-horizontal': spaces.Discrete(n=11),
                                                 'blue-vertical': spaces.Discrete(n=11), 'blue-horizontal': spaces.Discrete(n=11)}),


            }
        )

        self.action_space = spaces.Dict(
            {
                'pickup': spaces.MultiDiscrete([10] * 1),
                'assign': spaces.MultiDiscrete([10] * 1),
                'next_action': spaces.Dict({'red-vertical': spaces.Discrete(n=2), 'red-horizontal': spaces.Discrete(n=2),
                                                 'green-vertical': spaces.Discrete(n=2), 'green-horizontal': spaces.Discrete(n=2),
                                                 'blue-vertical': spaces.Discrete(n=2), 'blue-horizontal': spaces.Discrete(n=2)})

            }
        )

        self.current_observation = None
        self.previous_observation = None
        self.legal_action_counter = 0
        self.illegal_action_counter = 0
        self.total_illegal_reward = 0
        self.prev_reward = 0
        self.count = 0

    def _get_obs(self):
        if self.pyreason_simulator:
            self.current_observation = self.pyreason_bridge_world.get_obs()
        else:
            self.current_observation = self.other_simulator.get_obs()
        return self.current_observation

    def _get_info(self, observation):
        info_dict = {'success_step':0}
        len_slots = len(observation['slots'].keys())
        previous_action_counter = self.legal_action_counter
        if len_slots > previous_action_counter:
            info_dict['success_step'] = 1
        return info_dict

    def _get_rew(self, observation):
        '''
        :return: calculated reward
        '''
        #
        illegal_slot = -2
        legal_slot = 1
        done = 10
        incomplete = 0
        impossible = 0


        len_slots = len(observation['slots'].keys())
        final_reward = self.prev_reward
        previous_action_counter = self.legal_action_counter
        total_blocks = 5
        total_slots = 3
        number_of_left_blocks = 0
        for k, v in observation['blocks_available'].items():
            number_of_left_blocks += v

        # number_of_left_blocks = total_blocks - len(observation['costs'].keys())
        number_of_left_slots = total_slots - len(observation['slots'].keys())
        flag_illegal = False
        flag_legal = False
        flag_incomplete = False
        flag_complete = False

        # Illegal slot
        # print(observation)
        if len_slots == previous_action_counter:
            flag_illegal = True
            # extra_blocks = set(observation['costs'].keys()) - set(observation['slots'].values())
            final_reward += illegal_slot
            # for extra_block in extra_blocks:
            #     cost = observation['costs'][extra_block]
            #     self.total_illegal_reward += illegal_slot
            #
                # final_reward = final_reward - cost / 2 - illegal_slot
                # final_reward = final_reward - cost / 2
        # Legal slot
        elif len_slots > previous_action_counter:
            flag_legal = True
            final_reward += legal_slot

        # Incomplete
        if number_of_left_blocks < number_of_left_slots:
            flag_impossible = True
            final_reward += impossible

        if len(observation['slots']) == 3:
            flag_complete = True
            final_reward += done
        else:
            flag_incomplete = True


        # final_reward += self.total_illegal_reward
        self.legal_action_counter = len_slots
        self.prev_reward = final_reward

        if flag_illegal and flag_incomplete:
            return illegal_slot
        if flag_legal and flag_incomplete:
            return legal_slot
        if flag_legal and flag_complete:
            return legal_slot + done

        # return final_reward

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial conditions

        :param seed: random seed if there is a random component, defaults to None
        :param options: defaults to None
        """
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        #If pyreason!
        self.current_observation = None
        self.previous_observation = None
        self.legal_action_counter = 0
        self.illegal_action_counter = 0
        self.total_illegal_reward = 0
        self.prev_reward = 0
        self.count = 0
        if self.pyreason_simulator:
            self.pyreason_bridge_world.reset()
        #Else:
        else:
            self.other_simulator.reset()

        observation = self._get_obs()
        info = self._get_info(observation)
        return observation, info

    def step(self, action):

        if self.pyreason_simulator:
            self.pyreason_bridge_world.move(action)
        else:
            self.other_simulator.move(action)

        observation = self._get_obs()
        info = self._get_info(observation)

        # Get reward
        rew = self._get_rew(observation)

        # End of game
        done = self.is_done(observation)

        if self.count > 15 and not done:
            rew = -10
            truncate = True
        else:
            truncate = False
        self.count += 1

        return observation, rew, done, truncate, info

    def is_done(self, observation):
        done = False
        if self.pyreason_simulator:
            if len(observation['slots']) == 3:
                done = True
        else:
             self.other_simulator.is_done(observation)
        return done
    def initialize_facts(self, initial_facts):
        if self.pyreason_simulator:
            self.pyreason_bridge_world.initialize_env(initial_facts)

        else:
            self.other_simulator.initialize_env(initial_facts)
        observation = self._get_obs()
        return observation
