"""Tests for learn2decompose_approach.py."""

import random
import pickle
import os
import networkx as nx
import numpy as np
from sympy.utilities.iterables import multiset_partitions
from python_research_starter.planners.importance_planning import TaskThenMotionPlannerImportance

from pybullet_blocks.envs.symbolic_block_stacking_env import (
    SymbolicBlockStackingPyBulletBlocksEnv,
)
from pybullet_blocks.planning_models.action import get_active_operators_and_skills
from pybullet_blocks.planning_models.perception import (
    PREDICATES,
    TYPES,
    SymbolicBlockStackingPyBulletBlocksPerceiver,
)

# GNN
import torch
from python_research_starter.subgoal_pipeline.GNN_Models import get_model
from python_research_starter.subgoal_pipeline.GraphPairDataset import GraphPairDataset

print(os.getcwd())
in_channels = 18
hidden_channels = 64
edge_attr_channels = 4

mp_model = get_model("mpnn", in_channels, hidden_channels, edge_attr_channels = edge_attr_channels)

# Load the saved model weights
mp_model.load_state_dict(
    torch.load("/Users/skywalkerli/Desktop/Princeton_2024_2025/Research/learn-to-decompose/src/python_research_starter/subgoal_pipeline/saved_models/mp_graph_importance_checkpoint.pt")['model_state_dict'])
mp_model.eval()  # Set model to evaluation mode


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

    elements = [0, 1, 2, 3, 4, 5]
    # Generate all partitions and assign them fixed numbers
    all_partitions = list(
        multiset_partitions(elements, m=None)
    )  # k=None means any number of groups

    # print(all_partitions)

    edge_links_to_index = create_edge_dict(6)
    # print(edge_links_to_index)

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
    planner = TaskThenMotionPlannerImportance(
        TYPES, PREDICATES, perceiver, operators, skills, planner_id="fd-opt"
    )

    # Run an episode.
    # seed = 99

    # Generate a list of all possible initial configurations
    scene_blocks = ["A", "B", "C", "D", "E", "F"]
    init_configurations = list(multiset_partitions(scene_blocks, m=None))
    goal_pile = [["A", "B", "C", "D", "E", "F"]]

    common_edge_link_patterns = [[1], [1, 11], [1, 11, 19], [1, 11, 19, 25], [1, 11, 19, 25,29]]

    # Number of distinct demonstrations to generate scene data from
    num_demonstrations = 1

    for demo in range(num_demonstrations):

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

        ### TODO
        """
        Implement the paper's strategy of predicting the object importances using less and less
        restrictive importance thresholds. Then these will be the new init piles while keeping the goal
        piles the same? Keep both the init state / pile and goal state the same
        Instead, maybe there is some parameter in the PDDL stream planner that allows for it
        to only look at a certain subset of objects when planning -I can look into that
        And then time how fast each plan runs concurrently?
        I feel like the plan generation is very quick just empircally  
        """

        obs, info = env.reset(seed=seed, options=scene_init)

        init_state = env.get_state()
        init_graph = init_state.to_observation()
        # print(init_state)
        # print(init_graph)
        nodes = init_graph.nodes
        edge_links = init_graph.edge_links
        # print(edge_links)
        edge_index_list = []
        for edge_link in edge_links:
            edge_index_list.append(edge_links_to_index[tuple(edge_link)])
        edge_index_list.sort()

        subgoal_index = 0
        prev_subgoal_graph = init_graph

        ### Only increment index if A is on the bottom, otherwise the robot will have to rebuild anyways from scratch
        if nodes[1][10] == -1:
            print(edge_index_list)
            while is_subset(common_edge_link_patterns[subgoal_index], edge_index_list):
                subgoal_index += 1

        print(subgoal_index)

        ground_truth_subgoal_filepath = ["../subgoal_pipeline/subgoals/ab.pkl", 
                                         "../subgoal_pipeline/subgoals/abc.pkl", 
                                         "../subgoal_pipeline/subgoals/abcd.pkl", 
                                         "../subgoal_pipeline/subgoals/abcde.pkl", 
                                         "../subgoal_pipeline/subgoals/abcdef.pkl"]
        
        subgoal_piles = [['A', 'B'], 
                         ['A', 'B', 'C'], 
                         ['A', 'B', 'C', 'D'], 
                         ['A', 'B', 'C', 'D', 'E'], 
                         ['A', 'B', 'C', 'D', 'E', 'F'],]
        
        file_path = ground_truth_subgoal_filepath[subgoal_index]
        with open(file_path, "rb") as f:
            subgoal_graph = pickle.load(f)

        ### GNN CODE ###
        scene_data = GraphPairDataset.convert_data(init_graph, subgoal_graph)

        # Forward pass
        with torch.no_grad():
            out = mp_model(scene_data.x, scene_data.edge_index, scene_data.edge_attr)
            importance_scores = torch.sigmoid(out)

        print("Input graph: ", scene_data.x)
        print("Input edge_links", scene_data.edge_index)
        print(f"Node importance scores: {importance_scores.numpy()}")
        print(f"One-hot scores: {importance_scores.numpy() > 0.5}" )

        # Given the importance scores, create a new goal pile with a reduced subset of blocks
        # The new goal for the planner is now the subgoal, rather than the entire goal
        new_info = {'goal_piles': [subgoal_piles[subgoal_index]]}

        importance_thresh = 0.9
        planner.reset(obs, new_info, importance_scores.numpy(), importance_thresh)

        print("Running demo: ", demo)
        for _ in range(10000):  # should terminate earlier
            action = planner.step(obs)
            obs, reward, done, _, _ = env.step(action)

            if done:  # goal reached!
                assert reward > 0
                break
        else:
            assert False, "Goal not reached"

    env.close()
