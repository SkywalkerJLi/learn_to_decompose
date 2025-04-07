"""Tests for learn2decompose_approach.py."""

import random
import pickle
import os
import networkx as nx
import numpy as np
from sympy.utilities.iterables import multiset_partitions
from task_then_motion_planning.planning import TaskThenMotionPlanner

from pybullet_blocks.envs.symbolic_block_stacking_env import (
    SymbolicBlockStackingPyBulletBlocksEnv,
)
from pybullet_blocks.planning_models.action import get_active_operators_and_skills
from pybullet_blocks.planning_models.perception import (
    PREDICATES,
    TYPES,
    SymbolicBlockStackingPyBulletBlocksPerceiver,
)

"""
    Given directed edge links and list of nodes, return the directed connected components of the graph
"""
def find_connected_components(edges, num_nodes):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)

    # Find weakly connected components (ignoring direction)
    weak_components = list(nx.weakly_connected_components(G))

    # Sort each weakly connected component using topological sorting
    sorted_components = []
    for component in weak_components:
        subgraph = G.subgraph(component)  # Extract subgraph
        component = list(nx.topological_sort(subgraph))
        component.reverse()
        sorted_components.append(component)  # Sort preserving order

    # Sort components by the lowest block number for consistency
    sorted_components.sort(key=lambda comp: min(comp))

    return sorted_components

"""
    Convert a partition (list of lists) into a hashable key (tuple of
    frozensets).
"""
def partition_to_key(partition):
    return tuple(frozenset(subset) for subset in partition)

"""
    Efficiently map partitions to index numbers.
"""
def create_partition_dict(partitions):
    return {partition_to_key(partition): i for i, partition in enumerate(partitions)}

"""
    Create index for each possible edge links
"""
def create_edge_dict(num_nodes):
    edge_dict = {}
    count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_dict[tuple((i, j))] = count
            count += 1
            edge_dict[tuple((j, i))] = count
            count += 1
    return edge_dict

"""
    Checks if one list is a subset of another
"""
def is_subset(list1, list2):
    return set(list1).issubset(set(list2))
"""

    Tests Learn2Decompose planning in BlockStackingPyBulletBlocksEnv().
"""
def test_learn2decompose_approach(return_edge_links=True):

    np.random.seed(25)

    elements = [0, 1, 2, 3, 4, 5]
    # Generate all partitions and assign them fixed numbers
    all_partitions = list(
        multiset_partitions(elements, m=None)
    )  # k=None means any number of groups

    print(all_partitions)

    edge_links_to_index = create_edge_dict(6)
    print(edge_links_to_index)

    partition_dict = create_partition_dict(all_partitions)

    env = SymbolicBlockStackingPyBulletBlocksEnv(use_gui=True)
    sim = SymbolicBlockStackingPyBulletBlocksEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/block-stacking-ttmp-test")
    max_motion_planning_time = 0.5  # increase for prettier videos

    perceiver = SymbolicBlockStackingPyBulletBlocksPerceiver(sim)
    operators, skill_types = get_active_operators_and_skills()
    skills = {
        s(sim, max_motion_planning_time=max_motion_planning_time) for s in skill_types
    }

    # Create the planner.
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, operators, skills, planner_id="pyperplan"
    )

    # Run an episode.
    # seed = 99

    # Generate a list of all possible initial configurations
    scene_blocks = ["A", "B", "C", "D", "E", "F"]
    init_configurations = list(multiset_partitions(scene_blocks, m=None))
    goal_pile = [["A", "B", "C", "D", "E", "F"]]

    common_edge_link_patterns = [[1], [1, 11], [1, 11, 19], [1, 11, 19, 25], [1, 11, 19, 25,29]]

    # Number of distinct demonstrations to generate scene data from
    num_demonstrations = 5

    for demo in range(num_demonstrations):

        # Append new demonstration to training dataset
        file_path = "dataset.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                training_dataset = pickle.load(f)
        else:
            training_dataset = []

        print("START")
        seed = np.random.randint(0, 1000)
        rand_index = np.random.randint(0, len(init_configurations))
        init_piles = init_configurations[rand_index]
        for pile in init_piles:
            random.shuffle(pile)
        print(init_piles)

        scene_init = {
            "init_piles": init_piles,
            "goal_piles": goal_pile,
        }

        obs, info = env.reset(seed=seed, options=scene_init)

        init_state = env.get_state()
        init_graph = init_state.to_observation()
        print(init_state)
        print(init_graph)
        nodes = init_graph.nodes
        edge_links = init_graph.edge_links
        print(edge_links)
        edge_index_list = []
        for edge_link in edge_links:
            edge_index_list.append(edge_links_to_index[tuple(edge_link)])
        edge_index_list.sort()

        subgoal_index = 0
        prev_subgoal_graph = init_graph

        ### Only increment index if A is on the bottom, otherwise the robot will have to rebuild anyways from scratch
        if nodes[1][10] == -1:
            while is_subset(common_edge_link_patterns[subgoal_index], edge_index_list):
                subgoal_index += 1

        print(subgoal_index)
        demonstration = []

        connected_components = find_connected_components(edge_links, len(nodes) - 1) # subtract one node for the robot's pose
        scene_subgraph_id = partition_dict[partition_to_key(connected_components)]
        previous_unique_parition_id = scene_subgraph_id

        if return_edge_links:
            for edge_link in edge_links:
                demonstration.append(edge_links_to_index[tuple(edge_link)])
            demonstration.append(-1)
        else:
            demonstration.append(scene_subgraph_id)
            demonstration.append(-1)

        planner.reset(obs, info)

        print("Running demo: ", demo)
        for _ in range(10000):  # should terminate earlier
            action = planner.step(obs)
            obs, reward, done, _, _ = env.step(action)
            state = env.get_state()
            scene_graph = state.to_observation()
            nodes = scene_graph.nodes
            edge_links = scene_graph.edge_links

            connected_components = find_connected_components(edge_links, len(nodes) - 1)
            scene_subgraph_id = partition_dict[partition_to_key(connected_components)]

            if scene_subgraph_id != previous_unique_parition_id: # ensure that no duplicate states are considered
                edge_index_list = []
                for edge_link in edge_links:
                    edge_index_list.append(edge_links_to_index[tuple(edge_link)])
                edge_index_list.sort()
                
                # If the scene graph is at the new subgoal, store the entire scene graph and determine the important object
                important_objects_list = np.zeros(len(nodes) - 1)
                if nodes[1][10] == -1 and is_subset(common_edge_link_patterns[subgoal_index], edge_index_list):
                    print('is subset')
                    for i, (prev_node, curr_node) in enumerate(zip(prev_subgoal_graph.nodes[1:], nodes[1:])):
                        print("Node: ", chr(int(prev_node[8] + 97)).upper())
                        print("Previously node is on: ", prev_node[10])
                        print("Currently node is on: ", curr_node[10])
                        if(prev_node[10] != curr_node[10]):
                            important_objects_list[i] = 1
                    training_dataset.append((prev_subgoal_graph, important_objects_list))
                    print(prev_subgoal_graph)
                    print(important_objects_list)
                    prev_subgoal_graph = scene_graph
                    subgoal_index += 1

                previous_unique_parition_id = scene_subgraph_id
            if done:  # goal reached!
                assert reward > 0
                break
        else:
            assert False, "Goal not reached"

        with open("dataset.pkl", "wb") as f:
            pickle.dump(training_dataset, f)
        
    env.close()
