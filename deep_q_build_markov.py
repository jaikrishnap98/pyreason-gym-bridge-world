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


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        self.fc1 = nn.Linear(in_states, h1_nodes)
        self.out = nn.Linear(h1_nodes, out_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

class LegalBridgeDQL():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.001  # learning rate (alpha)
    discount_factor_g = 0.99  # discount rate (gamma)
    network_sync_rate = 5  # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 1000  # size of replay memory
    mini_batch_size = 16  # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()  # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None  # NN Optimizer. Initialize later.

    ACTIONS = ['red-vertical', 'red-horizontal','green-vertical', 'green-horizontal','blue-vertical', 'blue-horizontal']

    def train(self, episodes, train_set, test_set, render=False):
        env = gym.make('PyReasonBridgeWorld-v0')
        num_states = 9
        num_actions = 6

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(in_states=num_states, h1_nodes=64, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=64, out_actions=num_actions)

        target_dqn.load_state_dict(policy_dqn.state_dict())

        # print('Policy (random, before training):')
        # self.print_dqn(policy_dqn)

        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr = self.learning_rate_a)

        rewards_per_episode = np.zeros(episodes)


        epsilon_history = []
        step_count = 0

        for i in range(episodes):
            mode_val = len_train_set/10
            if i%mode_val==0:
                len_test_set = len(test_set)
                done_count = blocks_world.test(len_test_set, test_set)
                accuracy = done_count / len_test_set
                print(f'Accuracy: {accuracy*100:.2f}%')
            state = env.reset()[0]
            real_to_node_initial_facts, real_initial_facts = self.get_initial_blocks_dict(csv_file=f'input_data/{train_set[i]}.csv')
            state_dict = env.initialize_facts(real_to_node_initial_facts)
            # print(state_dict)
            input_tensor = self.get_input_tensor_from_state_dict(state_dict)
            # print(input_tensor)
            # available_blocks = self.combine_values(real_initial_facts)
            # print(available_blocks)
            block_availability_list = self.get_block_availability_list(real_initial_facts).copy()
            # print(block_availability_list)


            terminated = False
            truncated = False
            policy_actions_slots = ['h1', 'h2', 'h3']
            reward = 0
            # temp_available_blocks = available_blocks.copy()
            temp_block_availability_list = block_availability_list.copy()
            while(not terminated and not truncated):
                # print(block_availability_list)
                # print(available_blocks)
                if random.random() < epsilon:
                    keys_list = [key for key in temp_block_availability_list.keys() if temp_block_availability_list[key]!=[]]
                    action_string = random.choice(keys_list)
                    # index_to_remove = temp_available_blocks.index(action_string)
                    action_number = self.get_action_number(action_string)
                    action_block_number = self.get_action_block_number(action_number, temp_block_availability_list)
                    # print(action_string, action_block_number)


                else:
                    with torch.no_grad():
                        action_number = policy_dqn(input_tensor).argmax().item()
                        action_string = self.get_action_string(action_number)
                        # if temp_block_availability_list[action_string]!=[]:
                        #     index_to_remove = temp_available_blocks.index(action_string)
                        action_block_number = self.get_action_block_number(action_number, block_availability_list)
                        if action_block_number == 'b0':
                            reward = reward - 50000
                            memory.append((input_tensor, action_number, input_tensor, reward, terminated))
                            step_count += 1
                            continue
                        # print('=======================================================================================')

                # print(policy_actions_slots[0], action_block_number)
                new_state_dict, reward, terminated, truncated, info_dict = env.step((policy_actions_slots[0],action_block_number))
                new_state = self.get_input_tensor_from_state_dict(new_state_dict)
                # print((input_tensor, action_number, new_state_dict, new_state, reward, terminated, info_dict))


                if info_dict['success_step'] == 1:
                    del policy_actions_slots[0]
                    # print(index_to_remove)
                    # print(available_blocks)
                    # del available_blocks[index_to_remove]

                    index_to_remove = temp_block_availability_list[action_string].index(action_block_number)
                    # print(index_to_remove)
                    # print(available_blocks)
                    del block_availability_list[action_string][index_to_remove]
                    # temp_available_blocks = available_blocks.copy()
                    temp_block_availability_list = block_availability_list.copy()
                    # print(policy_actions_slots)
                    # print(available_blocks)
                    # print(block_availability_list)
                else:
                    # temp_available_blocks = [item for item in temp_available_blocks if item != action_string]
                    # index_to_remove = temp_block_availability_list[action_string].index(action_block_number)
                    temp_block_availability_list[action_string]= []
                    new_state = self.update_input_tensor_on_block_availability(new_state, temp_block_availability_list)
                    # print(policy_actions_slots)
                    # print(temp_available_blocks)
                    # print(temp_block_availability_list)


                memory.append((input_tensor, action_number, new_state, reward, terminated))
                # print((input_tensor, action_number, new_state_dict, new_state, reward, terminated, info_dict))
                input_tensor = new_state

                step_count+=1

            rewards_per_episode[i] = reward

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon - 1 / episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

                # Close environment
            env.close()

            # Save policy
            torch.save(policy_dqn.state_dict(), "bridge_world_dql.pt")

            # Create new graph
            plt.figure(1)

            # Plot average rewards (Y-axis) vs episodes (X-axis)
            sum_rewards = np.zeros(episodes)
            for x in range(episodes):
                sum_rewards[x] = np.sum(rewards_per_episode[max(0, x - 100):(x + 1)])
            plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
            plt.plot(sum_rewards)

            # Plot epsilon decay (Y-axis) vs episodes (X-axis)
            plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
            plt.plot(epsilon_history)

            # Save plots
            plt.savefig('bridge_dql.png')


    def test(self, episodes, test_set):
        #

        env = gym.make('PyReasonBridgeWorld-v0')
        num_states = 9
        num_actions = 6

        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=64, out_actions=num_actions)
        policy_dqn.load_state_dict(torch.load("bridge_world_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        # print('Policy (trained):')
        # self.print_dqn(policy_dqn)
        done_count = 0
        step_count = 0
        for i in range(episodes):
            state = env.reset()[0]
            real_to_node_initial_facts, real_initial_facts = self.get_initial_blocks_dict(
                csv_file=f'input_data/{test_set[i]}.csv')
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
                    action_number = policy_dqn(input_tensor).argmax().item()
                    action_string = self.get_action_string(action_number)
                    action_block_number = self.get_action_block_number(action_number, block_availability_list)
                    if action_block_number == 'b0':
                        step_count += 1
                        break
                    # print(action_block_number)
                    # print('=======================================================================================')

                # print(policy_actions_slots[0], action_block_number)
                new_state_dict, reward, terminated, truncated, info_dict = env.step(
                    (policy_actions_slots[0], action_block_number))
                new_state = self.get_input_tensor_from_state_dict(new_state_dict)
                # print((input_tensor, action_number, new_state_dict, new_state, reward, terminated, info_dict))
                if terminated:
                    done_count += 1

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


        env.close()
        return done_count
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated:
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(new_state).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(state)
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(state)
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)

        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
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
            slots_available.get('h1', 0),
            slots_available.get('h2', 0),
            slots_available.get('h3', 0),
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


    def split_train_test(self, total_samples = 1000):


        # Define the total number of samples
        # total_samples = 2000

        # Define the percentage split
        train_percentage = 0.8
        test_percentage = 0.2

        # Calculate the number of samples for each split
        num_train_samples = int(total_samples * train_percentage)
        num_test_samples = total_samples - num_train_samples

        # Generate a list of numbers from 1 to 1000
        numbers = list(range(1, total_samples + 1))

        # Randomly select numbers for the train set
        train_set = random.sample(numbers, num_train_samples)

        # Remove selected numbers from the list to ensure no overlap
        for num in train_set:
            numbers.remove(num)

        # The remaining numbers constitute the test set
        test_set = numbers

        return train_set, test_set


    '''
        Converts an state (int) to a tensor representation.
        For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

        Parameters: state=1, num_states=16
        Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        '''

    # def state_to_dqn_input(self, state: int, num_states: int) -> torch.Tensor:
    #     input_tensor = torch.zeros(num_states)
    #     input_tensor[state] = 1
    #     return input_tensor



    # Print DQN: state, best action, q values
    # def print_dqn(self, dqn):
    #     # Get number of input nodes
    #     num_states = dqn.fc1.in_features
    #
    #     # Loop each state and print policy to console
    #     for s in range(num_states):
    #         #  Format q values for printing
    #         q_values = ''
    #         for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
    #             q_values += "{:+.2f}".format(q) + ' '  # Concatenate q values, format to 2 decimals
    #         q_values = q_values.rstrip()  # Remove space at the end
    #
    #         # Map the best action to L D R U
    #         best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]
    #
    #         # Print policy in the format of: state, action, q values
    #         # The printed layout matches the FrozenLake map.
    #         print(f'{s:02},{best_action},[{q_values}]', end=' ')
    #         if (s + 1) % 4 == 0:
    #             print()  # Print a newline every 4 states


if __name__ == '__main__':
    blocks_world= LegalBridgeDQL()
    train_set, test_set = blocks_world.split_train_test(total_samples=10)
    # train_set = [55, 56,57,58,59,60,61,62,63]
    # test_set = [55, 56,57,58]
    len_test_set = len(test_set)
    len_train_set = len(train_set)
    # print(len_test_set)
    blocks_world.train(len_train_set, train_set, test_set)
    done_count = blocks_world.test(len_test_set, test_set)
    accuracy = done_count / len_test_set
    print(accuracy)