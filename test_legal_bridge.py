import pyreason_gym
import gym
SIMULATOR_PYREASON = True
# DEFAULT = True


env = gym.make('PyReasonBridgeWorld-v0',pyreason_simulator=SIMULATOR_PYREASON)

default = input('Do you want to use default blocks and properties?(y/n): ')
if default=='y':
    initial_facts = {
        'b1': ['red', 'vertical', '2'],
        'b2': ['red', 'vertical', '2'],
        'b3': ['blue', 'horizontal', '6'],
        'b4': ['green', 'horizontal', '4'],
        'b5': ['red', 'vertical', '2'],
        'b6': ['red', 'horizontal', '6'],
        'b7': ['green', 'vertical', '6']
    }
else:
    initial_facts = {
        'b1': ['','',''],
        'b2': ['','',''],
        'b3': ['','',''],
        'b4': ['','',''],
        'b5': ['','',''],
        'b6': ['','',''],
        'b7': ['','',''],
        'b8': ['','',''],
        'b9': ['','',''],
        'b10': ['','','']
    }
    num_blocks = 10
    for i in range(num_blocks):
        color_block = input(f'Select color for block b{i+1}-[red:1, green:2, blue:3]:')
        shape_block = input(f'Select shape for block b{i + 1}-[vertical:1, horizontal:2]:')
        if color_block=='1':
            initial_facts[f'b{i+1}'][0] = 'red'
            initial_facts[f'b{i + 1}'][2] = '2'

        elif color_block=='2':
            initial_facts[f'b{i + 1}'][0] = 'green'
            initial_facts[f'b{i + 1}'][2] = '4'
        elif color_block=='3':
            initial_facts[f'b{i + 1}'][0] = 'blue'
            initial_facts[f'b{i + 1}'][2] = '6'
        else:
            print('Wrong color')
            break
        if shape_block=='1':
            initial_facts[f'b{i + 1}'][1] = 'vertical'
        elif shape_block=='2':
            initial_facts[f'b{i + 1}'][1] = 'horizontal'
        else:
            print('Wrong shape')
            break
# Initial facts in environemnt and incorporated in graph



real_to_node_var ={
    'red': 'c1', 'green': 'c2', 'blue': 'c3', 'vertical': 's1', 'horizontal': 's2'
}
real_to_node_initial_facts = {}
for block, attributes in initial_facts.items():
    updated_attributes = [real_to_node_var.get(attr, attr) for attr in attributes]
    real_to_node_initial_facts[block] = updated_attributes

# print(real_to_node_initial_facts)
#Total combination of block types: Dictionary for counter
block_availability_list = {'red-vertical':[], 'red-horizontal':[], 'green-vertical':[], 'green-horizontal':[], 'blue-vertical':[], 'blue-horizontal':[]}



# Filling up the counter dictionary based on the initial facts of environment
for block, list_properties in initial_facts.items():
    color = list_properties[0]
    shape = list_properties[1]
    block_availability_list[f'{color}-{shape}'].append(block)

# Policy test: Picked blocks
# policy_actions_pick = ['red-square', 'red-square', 'red-square', 'blue-rectangle', 'green-rectangle', 'red-curve', 'green-curve', 'green-square', 'red-curve']
# policy_actions_pick = ['red-vertical', 'red-vertical', 'green-horizontal']
policy_actions_pick = ['green-vertical', 'red-vertical', 'blue-horizontal']
# Policy test: slots assigned, h1-h7 are slots for house and h0 is a trash!
# policy_actions_slots = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h0', 'h7', 'h7']
policy_actions_slots = ['h1', 'h2', 'h3']



# Now let us assign block numbers to the given type of blocks
policy_actions_picked_block = []
for block_type in policy_actions_pick:

    if len(block_availability_list[block_type])>=1:
        block_number = block_availability_list[block_type][0]
        policy_actions_picked_block.append(block_number)
        del block_availability_list[block_type][0]
    else:
        print('No such block available in environemnt')
print(policy_actions_picked_block)


# Reset the environemnt
obs = env.reset()
print(obs)
obs = env.initialize_facts(real_to_node_initial_facts)
print(obs)

# Perform action in two stages:
# First stage is picking up a block
# Second stage: Either assign the block to slot h1-h7 or put it in trash h0 or might just be illegal slot.
for slot, block in zip(policy_actions_slots, policy_actions_picked_block):
    action2 = (slot, block)
    obs = env.step(action2)
    print(obs)

# Function to create a text representation of a block
def block_representation(block_id):
    color, shape, _ = initial_facts[block_id]
    return f"{color.upper()}({shape[0].upper()})"

# start building the house if legal!
if obs[2]:
    slots = obs[0]['slots']
    # Creating the house structure
    bridge_structure = [
        ["     ",block_representation(slots['h3']), "   "],
        [block_representation(slots['h1']), "        ", block_representation(slots['h2'])]
    ]

    # Drawing the house
    for row in bridge_structure:
        print(' '.join(row))
else:
    print('Cannot build legal bridge for this setup!')