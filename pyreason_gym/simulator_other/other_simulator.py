import os

class OtherSimulator:
    def __init__(self):

        current_path = os.path.abspath(os.path.dirname(__file__))
        self.counter = 0
        # picked_blocks_cost

    def reset(self):
        '''
        Implementation for restarting the simulation goes here
        '''
        print("Restart the simulation here!")


    def move(self, action):
        '''
        :param action: str(eg: b1) or tuple(eg: (h3,b1))
        If action is string: b1 -> pickup block (eg: b1)
        If action is tuple: (h3,b1) -> assign block(b1) picked in previous timestep in slot(h3)
        '''
        # First stage of action: Block pickup
        if isinstance(action, str):
            print(f'Implementation for picking up block {action} goes here!')
        else:
            print(f'Implementation for assigning block {action[1]} to slot {action[0]} goes here!')



    def get_obs(self):
        observation = {'costs': {}, 'slots': {}, 'trash':[]}

        '''
            costs: dictionary of picked blocks until this timestep with (key, value) pair as (block, cost of block) -> Eg: (b1, 4)
            slots: dictionary of picked blocks assigned some slot as (key, value) pair as (slot, block) -> Eg: (h3, b1)
            trash: list of all blocks that are discarded in the policy.
            reward: calculated based on actions.
        '''
        '''
            Implementation for adding costs, slots, trash and reward goes here!
        '''
        # dummy observation for certain timestep.
        observation = {'costs': {'b1': 2, 'b2': 2, 'b3': 6, 'b4': 4, 'b5': 2, 'b6': 6, 'b10': 4}, 'slots': {'h1': 'b1', 'h2': 'b2', 'h3': 'b5', 'h4': 'b3', 'h5': 'b4', 'h6': 'b6'}, 'trash': ['b10']}
        return observation
    def initialize_env(self, initial_facts):
        '''

        :param initial_facts: dictionary key -> blocks, value-> list of [color_block, shape_block, cost_block]
        '''
        print(f'Implementation for adding properties to the blocks in the environemnt goes here.')

    def is_done(self, observation):
        '''

        :param observation: current observation dictionary
        :return: True if goal achieved else return False
        '''
        done = False
        '''
        Implementation for this goes here
        '''

        return done