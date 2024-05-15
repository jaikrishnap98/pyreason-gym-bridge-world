from collections import deque

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import random
import csv
import gym
import pyreason_gym
import matplotlib.pyplot as plt
import shutil


pref_type = 'color_color'
color_color = 'red_green'
data_directory = f'bridgeworld_data_pref_{pref_type}'
data_sub_directory = f'bridgeworld_data_pref_{color_color}'
model_file = f'bridge_world_dql_pref_{color_color}.pt'
final_model_file = f'bridge_world_dql_pref_{color_color}_final.pt'
figure_file = f'bridge_world_dql_pref_{color_color}.png'
class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.fc2 = nn.Linear(h1_nodes, h1_nodes*2)
        self.out = nn.Linear(h1_nodes*2, out_actions)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.out(x)
        return x

class LegalBridgeDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.01  # learning rate (alpha)
    discount_factor_g = 0.99  # discount rate (gamma)
    network_sync_rate = 5  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000  # size of replay memory
    mini_batch_size = 32  # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()  # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None  # NN Optimizer. Initialize later.

    ACTIONS = ['red-vertical', 'red-horizontal','green-vertical', 'green-horizontal', 'blue-vertical', 'blue-horizontal']

    def test(self, episodes, test_set, model):
        #

        env = gym.make('PyReasonBridgeWorld-v0', preferential_constraint=True, preferential_type=pref_type, color_color=color_color)
        num_states = 9
        num_actions = 6

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=64, out_actions=num_actions)

        policy_dqn.load_state_dict(torch.load(model))

        policy_dqn.eval()    # switch model to evaluation mode

        # print('Policy (trained):')
        # self.print_dqn(policy_dqn)
        done_count = 0
        step_count = 0
        rewards_per_episode = np.zeros(episodes)
        for i in range(episodes):
            episode_reward = 0
            # print('===================================')
            # print(f'Episode {test_set[i]}')
            state = env.reset()[0]
            real_to_node_initial_facts, real_initial_facts = self.get_initial_blocks_dict(
                csv_file=f'{data_directory}/{data_sub_directory}/{test_set[i]}.csv')
            # print(test_set[i])
            state_dict = env.initialize_facts(real_to_node_initial_facts)
            # print(state_dict)
            input_tensor = self.get_input_tensor_from_state_dict(state_dict)
            # print(input_tensor)
            block_availability_list = self.get_block_availability_list(real_initial_facts).copy()
            # print(block_availability_list)

            terminated = False
            truncated = False
            policy_actions_slots = ['h1', 'h2', 'h3']
            # Agent navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).

            temp_block_availability_list = block_availability_list.copy()
            while (not terminated and not truncated):

                with torch.no_grad():
                    # print('Inpu tensor: ',input_tensor)

                    action_number = policy_dqn(input_tensor).argmax().item()
                    action_string = self.get_action_string(action_number)
                    # print('Action: ', action_string)
                    action_block_number = self.get_action_block_number(action_number, block_availability_list)
                    if action_block_number == 'b0':
                        reward = -5
                        episode_reward += reward
                        step_count += 1
                        break
                    # print(action_block_number)
                    # print('=======================================================================================')

                # print(policy_actions_slots[0], action_block_number)
                new_state_dict, reward, terminated, truncated, info_dict = env.step(
                    (policy_actions_slots[0], action_block_number))
                new_state = self.get_input_tensor_from_state_dict(new_state_dict)
                # print(policy_actions_slots[0], action_block_number, action_string)
                # print((input_tensor, action_number, new_state_dict, new_state, reward, terminated, info_dict))
                episode_reward += reward
                if terminated:
                    done_count += 1
                    break

                if info_dict['success_step'] == 1:
                    del policy_actions_slots[0]
                    index_to_remove = temp_block_availability_list[action_string].index(action_block_number)
                    del block_availability_list[action_string][index_to_remove]
                    temp_block_availability_list = block_availability_list.copy()
                    # print(policy_actions_slots)
                    # print(available_blocks)
                    # print(block_availability_list)
                else:
                    temp_block_availability_list[action_string] = []
                    new_state = self.update_input_tensor_on_block_availability(new_state, temp_block_availability_list)
                    # print(policy_actions_slots)
                    # print(temp_block_availability_list)

                # print((input_tensor, action_number, new_state_dict, new_state, reward, terminated, info_dict))
                episode_reward += reward
                input_tensor = new_state

                step_count += 1

            rewards_per_episode[i] = episode_reward
        env.close()
        return rewards_per_episode, done_count


    def update_input_tensor_on_block_availability(self, new_state_tensor, temp_block_availability_dict):
        output_tensor = new_state_tensor.clone().detach()
        output_tensor[3] = len(temp_block_availability_dict['red-vertical'])
        output_tensor[4] = len(temp_block_availability_dict['red-horizontal'])
        output_tensor[5] = len(temp_block_availability_dict['green-vertical'])
        output_tensor[6] = len(temp_block_availability_dict['green-horizontal'])
        output_tensor[7] = len(temp_block_availability_dict['blue-vertical'])
        output_tensor[8] = len(temp_block_availability_dict['blue-horizontal'])

        return output_tensor


    def get_action_block_number(self, action_number, block_availability_list):

        # Now let us assign block numbers to the given type of blocks
        block_type = self.get_action_string(action_number)
        block_number = 'b0'
        if len(block_availability_list[block_type]) >= 1:
            block_number = block_availability_list[block_type][0]
            # del block_availability_list[block_type][0]
        # else:
            # print('No such block available in environemnt')
        return block_number

    def get_block_availability_list(self, initial_facts):
        block_availability_list = {'red-vertical': [], 'red-horizontal': [],
                                 'green-vertical': [], 'green-horizontal': [],
                                 'blue-vertical': [], 'blue-horizontal': []}

        # Filling up the counter dictionary based on the initial facts of environment
        for block, list_properties in initial_facts.items():
            color = list_properties[0]
            shape = list_properties[1]
            block_availability_list[f'{color}-{shape}'].append(block)
        return block_availability_list

    def get_action_string(self, action_number):
        action_string_dict = {'red-vertical': 0, 'red-horizontal': 1,
                                 'green-vertical': 2, 'green-horizontal': 3,
                                 'blue-vertical': 4, 'blue-horizontal': 5}
        reverse_dict = {value: key for key, value in action_string_dict.items()}
        return reverse_dict[action_number]
    def get_action_number(self, action_string):
        action_string_dict = {'red-vertical': 0, 'red-horizontal': 1,
                              'green-vertical': 2, 'green-horizontal': 3,
                              'blue-vertical': 4, 'blue-horizontal': 5}
        return action_string_dict[action_string]
    def get_input_tensor_from_state_dict(self, state_dict):
        blocks_available = state_dict.get('blocks_available', {})
        slots_available = state_dict.get('slots_available', {})

        # Extracting values in a specific order
        tensor_values = [
            slots_available.get('h1', 0)-1,
            slots_available.get('h2', 0)-1,
            slots_available.get('h3', 0)-1,
            blocks_available.get('red-vertical', 0),
            blocks_available.get('red-horizontal', 0),
            blocks_available.get('green-vertical', 0),
            blocks_available.get('green-horizontal', 0),
            blocks_available.get('blue-vertical', 0),
            blocks_available.get('blue-horizontal', 0)

        ]

        tensor_values = torch.Tensor(tensor_values)
        return tensor_values
    def combine_values(self, dict1):
        combined_list = []
        for key, value in dict1.items():
            combined_list.append('-'.join(value[:2]))  # Joining first two elements with a hyphen
        return combined_list
    def get_initial_blocks_dict(self, csv_file):
        initial_facts = {}
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for idx, row in enumerate(reader):
                key = f'b{idx + 1}'
                initial_facts[key] = row
        real_to_node_var = {
            'red': 'c1', 'green': 'c2', 'blue': 'c3', 'vertical': 's1', 'horizontal': 's2'
        }
        real_to_node_initial_facts = {}
        for block, attributes in initial_facts.items():
            updated_attributes = [real_to_node_var.get(attr, attr) for attr in attributes]
            real_to_node_initial_facts[block] = updated_attributes
        return real_to_node_initial_facts, initial_facts



if __name__ == '__main__':
    bridge_world= LegalBridgeDQL()
    test_set = [2, 12, 14, 18, 22, 35, 49, 55, 61, 62, 66, 68, 71, 73, 74, 80, 85, 90, 94, 96, 98, 99, 104, 107, 114, 119, 120, 122, 125, 126, 134, 135, 136, 141, 146, 155, 156, 158,
 164, 166, 169, 172, 179, 181, 184, 188, 192, 197, 199, 218, 221, 226, 230, 242, 246, 257, 267, 268, 270, 272, 275, 287, 296, 305, 307, 310, 313, 317, 318, 330, 335, 342,
 347, 351, 355, 363, 371, 372, 375, 379, 383, 392, 395, 396, 397, 399, 406, 411, 416, 421, 442, 456, 460, 463, 464, 468, 469, 475, 476, 478, 479, 491, 495, 497, 504, 506,
 516, 531, 535, 542, 547, 557, 558, 564, 574, 579, 580, 586, 589, 591, 593, 605, 611]
    len_test_set = len(test_set)
    # print(len_test_set)
    sample_test_set = random.sample(test_set, 20)
    len_sample_test_set = len(sample_test_set)
    print(sample_test_set)
    rws_per_episode, done_count = bridge_world.test(len_sample_test_set, sample_test_set, model=final_model_file)
    accuracy = done_count / len_sample_test_set
    print(done_count)
    print(rws_per_episode)
    print(f'Accuracy: {accuracy * 100:.2f}% ----------- Average reward: {sum(rws_per_episode) / len(rws_per_episode)}')


