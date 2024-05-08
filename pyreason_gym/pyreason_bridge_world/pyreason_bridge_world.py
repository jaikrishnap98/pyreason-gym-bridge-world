

import os
import pyreason as pr
import numpy as np


class PyReasonBridgeWorld:
    def __init__(self, graph, rules):
        self.interpretation = None

        # Keep track of the next timestep to start
        self.next_time = 0
        self.slots_available = {'h1': 0, 'h2': 0, 'h3': 0}
        self.blocks_available = {'red-vertical': 0, 'red-horizontal': 0,
                                 'green-vertical': 0, 'green-horizontal': 0,
                                 'blue-vertical': 0, 'blue-horizontal': 0}
        self.node_to_real_var = {
            'c1': 'red', 'c2': 'green', 'c3': 'blue', 's1': 'vertical', 's2': 'horizontal'
        }
        # Pyreason settings
        pr.settings.verbose = False
        pr.settings.atom_trace = True
        pr.settings.canonical = True
        pr.settings.inconsistency_check = False
        pr.settings.static_graph_facts = False
        pr.settings.save_graph_attributes_to_trace = False
        pr.settings.store_interpretation_changes = True
        current_path = os.path.abspath(os.path.dirname(__file__))


        # Load the graph
        if graph is None:
            pr.load_graphml(f'{current_path}/graph/bridge_world_graph.graphml')
        else:
            pr.load_graph(graph)

        # Load rules
        if rules is None:
            pr.add_rules_from_file(f'{current_path}/yamls/rules_bridge_new.txt', infer_edges=True)
        else:
            pass

    def reset(self):
        # Reason for 1 timestep to initialize everything
        # Certain internal variables need to be reset otherwise memory blows up

        pr.reset()
        self.interpretation = pr.reason(0, again=False)
        # pr.save_rule_trace(self.interpretation)
        self.next_time = self.interpretation.time + 1
        self.slots_available = {'h1': 0, 'h2': 0, 'h3': 0}
        self.blocks_available = {'red-vertical': 0, 'red-horizontal': 0,
                                 'green-vertical': 0, 'green-horizontal': 0,
                                 'blue-vertical': 0, 'blue-horizontal': 0}


    def move(self, action):
        node_facts = []
        edge_facts = []

        fact_add_edge = pr.fact_edge.Fact(f'add_edge_t{self.next_time}', (action[0], action[1]),
                                          pr.label.Label('atLoc'), pr.interval.closed(1, 1), self.next_time,
                                          self.next_time)
        fact_remove_edge = pr.fact_edge.Fact(f'add_edge_t{self.next_time}', (action[0], action[1]),
                                          pr.label.Label('atLoc'), pr.interval.closed(1, 1), self.next_time+1,
                                          self.next_time+1)
        fact_picked_on = pr.fact_node.Fact(f'block_picked_t{self.next_time}', action[1],
                                           pr.label.Label('picked'), pr.interval.closed(1, 1),
                                           self.next_time, self.next_time)
        fact_picked_off = pr.fact_node.Fact(f'block_picked_off_t{self.next_time}', action[1],
                                           pr.label.Label('picked'), pr.interval.closed(0,0),
                                           self.next_time+1, self.next_time+1)
        edge_facts.append(fact_add_edge)
        edge_facts.append(fact_remove_edge)
        node_facts.append(fact_picked_on)
        node_facts.append(fact_picked_off)
        self.interpretation = pr.reason(again=True, node_facts=node_facts, edge_facts=edge_facts)
        # pr.save_rule_trace(self.interpretation)
        # self.interpretation = pr.reason(again=True, edge_facts=edge_facts)
        # pr.save_rule_trace(self.interpretation)
        self.next_time = self.interpretation.time + 1
        # print(self.interpretation.interpretations_edge)
        # print(self.interpretation.interpretations_node)



    def get_obs(self):

        observation = {'costs': {}, 'slots': {},
                       'slots_available': self.slots_available.copy(),
                       'blocks_available': self.blocks_available.copy()
                       }

        # Filter edges that are of the form (h_,b_)
        relevant_edges_slots = [edge for edge in self.interpretation.edges if edge[0].startswith('h') and edge[1].startswith('b')]

        edges_blocks_colors = [edge for edge in self.interpretation.edges if
                                edge[0].startswith('b') and edge[1].startswith('c')]

        edges_blocks_shapes = [edge for edge in self.interpretation.edges if
                                        edge[0].startswith('b') and edge[1].startswith('s')]

        # Select edges that have the atSlot predicate set to [1,1]
        relevant_slot_edges = [edge for edge in relevant_edges_slots if self.interpretation.interpretations_edge[edge].world[pr.label.Label('atSlot')]==pr.interval.closed(1,1)]

        relevant_color_edges = [edge for edge in edges_blocks_colors if self.interpretation.interpretations_edge[edge].world[pr.label.Label('color')]==pr.interval.closed(1,1)]

        relevant_shape_edges = [edge for edge in edges_blocks_shapes if
                                self.interpretation.interpretations_edge[edge].world[
                                    pr.label.Label('shape')] == pr.interval.closed(1, 1)]

        # Filter edges that are of the form (b_,c1) where c1 is a int
        relevant_edges_cost = [edge for edge in self.interpretation.edges if
                                edge[0].startswith('b') and edge[1].isnumeric()]

        # Select edges that have the cost predicate set to [1,1]
        relevant_cost_edges = [edge for edge in relevant_edges_cost if
                               self.interpretation.interpretations_edge[edge].world[
                                   pr.label.Label('pickUpCost')] == pr.interval.closed(1, 1)]


        for edge in relevant_cost_edges:
            observation['costs'][edge[0]] = int(edge[1])
        for edge in relevant_slot_edges:
            observation['slots'][edge[0]] = edge[1]
        temp_dict = {}
        for edge in relevant_color_edges:
            if edge[1]=='c1':
                temp_dict[edge[0]] = 'red'
            elif edge[1]=='c2':
                temp_dict[edge[0]] = 'green'
            elif edge[1]=='c3':
                temp_dict[edge[0]] = 'blue'
        for edge in relevant_shape_edges:
            if edge[1] == 's1':
                temp_dict[edge[0]] = temp_dict[edge[0]]+'-vertical'
            elif edge[1] == 's2':
                temp_dict[edge[0]] = temp_dict[edge[0]]+'-horizontal'
        action_string_dict = {'red-vertical': 1, 'red-horizontal': 2,
                                 'green-vertical': 3, 'green-horizontal': 4,
                                 'blue-vertical': 5, 'blue-horizontal': 6}

        for slot, block in observation['slots'].items():
            block_color_shape = temp_dict[block]
            block_tensor_val = action_string_dict[block_color_shape]
            observation['slots_available'][slot] = block_tensor_val
            observation['blocks_available'][block_color_shape] -= 1



        return observation
    def initialize_env(self, initial_facts):

        node_to_real_initial_facts = {}
        for block, attributes in initial_facts.items():
            updated_attributes = [self.node_to_real_var.get(attr, attr) for attr in attributes]
            node_to_real_initial_facts[block] = updated_attributes


        facts = []
        for block in initial_facts.keys():

            fact_color = pr.fact_edge.Fact(f'block_color_{block[1:]}', (block, initial_facts[block][0]),
                                           pr.label.Label('color'), pr.interval.closed(1, 1), self.next_time, self.next_time)
            fact_shape = pr.fact_edge.Fact(f'block_shape_{block[1:]}', (block, initial_facts[block][1]),
                                           pr.label.Label('shape'),
                                           pr.interval.closed(1, 1), self.next_time, self.next_time)
            fact_cost = pr.fact_edge.Fact(f'block_price_{block[1:]}', (block, initial_facts[block][2]),
                                          pr.label.Label('cost'),
                                          pr.interval.closed(1, 1), self.next_time, self.next_time)
            self.blocks_available[f'{node_to_real_initial_facts[block][0]}-{node_to_real_initial_facts[block][1]}'] += 1
            facts.append(fact_color)
            facts.append(fact_shape)
            facts.append(fact_cost)
        self.interpretation = pr.reason(again=True, edge_facts=facts)
        # pr.save_rule_trace(self.interpretation)
        self.next_time = self.interpretation.time + 1