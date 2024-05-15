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

data_directory = 'bridgeworld_data_no_pref'
final_model_file = 'bridge_world_dql_no_pref_final.pt'
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

    def test(self, episodes, test_set, model_file):
        #

        env = gym.make('PyReasonBridgeWorld-v0', preferential_constraint = False)
        num_states = 9
        num_actions = 6

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=64, out_actions=num_actions)

        policy_dqn.load_state_dict(torch.load(model_file))

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
                csv_file=f'{data_directory}/{test_set[i]}.csv')
            # print(test_set[i])
            # print(real_to_node_initial_facts)
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
                    # print(action_block_number)

                    if action_block_number == 'b0':
                        # print('Picked b0')
                        reward = -5
                        episode_reward += reward
                        step_count += 1
                        break
                    # print(action_block_number)
                    # print('=======================================================================================')
                # print(policy_actions_slots[0], action_block_number)
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
        # h1_entry = 0
        # h2_entry = 0
        # h3_entry = 0
        # if slots_available['h1'] != 0:
        #     h1_entry = 1
        # if slots_available['h2'] != 0:
        #     h2_entry = 1
        # if slots_available['h3'] != 0:
        #     h3_entry = 1

        # Old approacch
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
        #New approach
        # tensor_values = [
        #     h1_entry,
        #     h2_entry,
        #     h3_entry,
        #     blocks_available.get('red-vertical', 0),
        #     blocks_available.get('red-horizontal', 0),
        #     blocks_available.get('green-vertical', 0),
        #     blocks_available.get('green-horizontal', 0),
        #     blocks_available.get('blue-vertical', 0),
        #     blocks_available.get('blue-horizontal', 0)
        #
        # ]
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
    test_set = [5, 8, 14, 20, 24, 26, 35, 48, 50, 54, 57, 59, 60, 68, 69, 74, 78, 83, 96, 101, 108, 122, 126, 132, 144, 145, 148, 155, 156, 158, 169, 170, 177, 181, 198, 200,
 207, 208, 209, 217, 220, 227, 234, 242, 243, 244, 245, 247, 255, 267, 268, 270, 272, 285, 290, 294, 295, 300, 301, 307, 310, 312, 317, 319, 332, 337, 338, 339,
 344, 346, 351, 358, 360, 361, 367, 369, 371, 378, 379, 386, 397, 413, 416, 426, 429, 438, 439, 460, 463, 465, 475, 476, 477, 478, 482, 485, 487, 488, 494, 496,
 499, 503, 504, 510, 511, 526, 533, 543, 547, 549, 559, 563, 566, 569, 582, 589, 591, 600, 611, 622, 625, 631, 633, 634, 641, 657, 662, 669, 674, 679, 682, 683,
 684, 685, 688, 699, 700, 705, 709, 718, 724, 725, 726, 731, 732, 733, 735, 749, 750, 754, 758, 769, 774, 777, 782, 789, 791, 792, 793, 802, 807, 819, 820, 829,
 834, 847, 849, 852, 853, 854, 860, 863, 865, 869, 870, 877, 885, 887, 888, 891, 894, 905, 908, 913, 914, 916, 917, 923, 947, 948, 949, 955, 965, 970, 971]
    len_test_set = len(test_set)
    # print(len_test_set)
    sample_test_set = random.sample(test_set, 10)
    # sample_test_set = [5]
    len_sample_test_set = len(sample_test_set)
    # print(sample_test_set)
    rws_per_episode, done_count = bridge_world.test(len_sample_test_set, sample_test_set, model_file=final_model_file)
    accuracy = done_count / len_sample_test_set
    # print(done_count)
    # print(rws_per_episode)
    print(f'Accuracy: {accuracy * 100:.2f}% ----------- Average reward: {sum(rws_per_episode) / len(rws_per_episode)}')
    #
    # for rw, t in zip(rws_per_episode, test_set):
    #     print(rw, t)

