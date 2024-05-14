'''
Creating input data for bridgeworld
Authors: Kaustuv Mukherji
Last updated: 05-13-2024
'''
import random

# Remember this will not overwrite previous files.
# Either create new folder everytime or delete contents of folder before running
write_path = "bridgeworld_data_pref_shape_color/bridgeworld_data_pref_horizontal_blue/"

block_colors = ['red', 'green', 'blue']
# type1 = 'square'
type1 = 'vertical'
# type2 = 'rectangle'
type2 = 'horizontal'

### preferential constraint type
# pref_type = 0 # no preference
# pref_type = 1 # colors can't touch
pref_type = 2 # particular block not allowed

### preferential constraint inputs | 'pref_1' must always be a color
## sample for pref_type = 1
# pref_1 = 'red'
# pref_2 = 'blue'

## sample for pref_type = 2
pref_1 = 'blue'
pref_2 = 'horizontal'

#change this as required
block_costs = {'red': 6, 'green': 4, 'blue': 2}

#### Don't need to change anything below this
block_shape = [type1, type2]

input_num = 0
for color in block_colors:
    # legal = False
    if (pref_type == 2 and pref_1 == color and pref_2 == type1):
        continue
    b_left = str(color)+ ',' + type1 + ','+ str(block_costs[color])
    for color2 in block_colors:
        if (pref_type == 2 and pref_1 == color2 and pref_2 == type1):
            continue
        b_right = str(color2)+ ',' + type1 + ','+ str(block_costs[color2])
        for color3 in block_colors:
            if (pref_type == 2 and pref_1 == color3 and pref_2 == type2):
                continue
            if (pref_type == 1 and ((color3 == pref_1 and (color2 == pref_2 or color == pref_2)) or ((color2 == pref_1 or color == pref_1) and color3 == pref_2))):
                continue
            b_top = str(color3)+ ',' + type2 + ','+ str(block_costs[color3])
            # print("Input Num: {}\t{};{};{};{};{};{};{}\n".format(input_num, b1, b2, b3, b4, b5, b6, b7))
            for color4 in block_colors:
                for shape4 in block_shape:
                    b4 = str(color4)+ ',' +str(shape4) + ','+ str(block_costs[color4])
                    for color5 in block_colors:
                        for shape5 in block_shape:
                            b5 = str(color5)+ ',' +str(shape5) + ','+ str(block_costs[color5])
                            input_num += 1
                            blocks = [b_left, b_right, b_top, b4, b5]
                            random.shuffle(blocks)
                            # with open(write_path+"0"+str(input_num)+".csv", 'w+') as writefile:
                            with open(write_path + str(input_num) + ".csv", 'w+') as writefile:
                                for block in blocks:
                                    writefile.write("{}\n".format(block))