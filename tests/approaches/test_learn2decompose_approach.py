"""Tests for learn2decompose_approach.py."""

import numpy as np
import networkx as nx
import random
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
from python_research_starter.approaches.learn2decompose_approach import (
    Learn2DecomposeApproach,
)

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

def partition_to_key(partition):
    """Convert a partition (list of lists) into a hashable key (tuple of frozensets)."""
    return tuple(frozenset(subset) for subset in partition)

def create_partition_dict(partitions):
    """Efficiently map partitions to index numbers."""
    return {partition_to_key(partition): i for i, partition in enumerate(partitions)}

def create_edge_dict(num_nodes):
    edge_dict = {}
    count = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            edge_dict[tuple(i, j)] = count
            count += 1
            edge_dict[tuple(j, i)] = count
            count += 1
    return edge_dict
def test_learn2decompose_approach():
    """Tests Learn2Decompose planning in BlockStackingPyBulletBlocksEnv()."""

    np.random.seed(2)

    elements = [0, 1, 2, 3, 4, 5]
    # Generate all partitions and assign them fixed numbers
    all_partitions = list(multiset_partitions(elements, m=None))  # k=None means any number of groups
    
    print(all_partitions)



    # # Assign a unique index to each partition
    # index_to_partition = {i: part for i, part in enumerate(all_partitions)}

    # partition_dict = create_partition_dict(all_partitions)

    # env = SymbolicBlockStackingPyBulletBlocksEnv(use_gui=True)
    # sim = SymbolicBlockStackingPyBulletBlocksEnv(env.scene_description, use_gui=False)

    # # from gymnasium.wrappers import RecordVideo
    # # env = RecordVideo(env, "videos/block-stacking-ttmp-test")
    # max_motion_planning_time = 0.5  # increase for prettier videos

    # perceiver = SymbolicBlockStackingPyBulletBlocksPerceiver(sim)
    # operators, skill_types = get_active_operators_and_skills()
    # skills = {
    #     s(sim, max_motion_planning_time=max_motion_planning_time) for s in skill_types
    # }

    # # Create the planner.
    # planner = TaskThenMotionPlanner(
    #     TYPES, PREDICATES, perceiver, operators, skills, planner_id="pyperplan"
    # )

    # # Run an episode.
    # # seed = 99

    # # Generate a list of all possible initial configurations
    # scene_blocks = ["A", "B", "C", "D", "E", "F"]
    # test_blocks = [["B", "A", "C"]]
    # init_configurations = list(multiset_partitions(scene_blocks, m=None))
    # goal_pile = [["A", "B", "C", "D", "E", "F"]]
    # test_goal = [["A", "C", "B"]]

    # # Number of distinct demonstrations to train on
    # num_demonstrations = 100

    # all_demonstrations = []
    # for demo in range(num_demonstrations):
    #     print('START')
    #     seed = np.random.randint(0, 1000)
    #     rand_index = np.random.randint(0, len(init_configurations))
    #     init_piles = init_configurations[rand_index]
    #     for pile in init_piles:
    #         random.shuffle(pile)
    #     print(init_piles)

    #     scene_init = {
    #         "init_piles": init_piles,
    #         "goal_piles": goal_pile,
    #     }
        
    #     obs, info = env.reset(
    #         seed = seed,
    #         options = scene_init
    #     )

    #     # print("OBSERVATIONS")
    #     # print(obs)
    #     init_state = env.get_state()
    #     # print("State representation")
    #     # print(init_state)

    #     #test_partition = partition_to_key([[1,2], [3,4],[5,6]])
    #     #print(partition_dict[test_partition])

    #     init_graph = init_state.to_observation()
    #     nodes = init_graph.nodes
    #     edge_links = init_graph.edge_links
    #     print(edge_links)
    #     connected_components = find_connected_components(edge_links, len(nodes) - 1)
    #     print('ordered components')
    #     print(connected_components)

    #     #env.close()
    #     #print("Graph Representation")
    #     #print(init_graph)
    #     planner.reset(obs, info)

    #     print('Running demo: ', demo)

    #     demonstration = []
    #     for _ in range(10000):  # should terminate earlier
    #         action = planner.step(obs)
    #         obs, reward, done, _, _ = env.step(action)
    #         state = env.get_state()
    #         # print(state)
    #         scene_graph = state.to_observation()
    #         nodes = scene_graph.nodes
    #         edge_links = scene_graph.edge_links
    #         # print("nodes")
    #         # print(nodes)
    #         # print("edge-links")
    #         # print(edge_links)
    #         # print("--------------------------------")
    #         connected_components = find_connected_components(edge_links, len(nodes) - 1)
    #         print(connected_components)
    #         scene_subgraph_id = partition_dict[partition_to_key(connected_components)]

    #         print(connected_components)

    #         # print(scene_subgraph_id)
    #         if len(demonstration) == 0:
    #             demonstration.append(scene_subgraph_id)
    #         elif len(demonstration) > 0 and demonstration[-1] != scene_subgraph_id:
    #             demonstration.append(scene_subgraph_id)
    #         if done:  # goal reached!
    #             assert reward > 0
    #             break
    #     else:
    #         assert False, "Goal not reached"

    #     #print(demonstration)
    #     with open("demonstration.txt", "a", encoding="utf-8") as file:
    #         file.write(", ".join(map(str, demonstration)) + "\n")  # Joins without extra comma

    #     # with open("demonstration.txt", "w", encoding="utf-8") as file:
    #     #     for partition in demonstration:
    #     #         file.write(str(partition) + ", ")
    #     #     #file.write(", ".join(str(demonstration)))  # Join list elements into a single string
    #     #     file.write("\n")
    #     #print()
    #     all_demonstrations.append(demonstration)

    # print("------------------------------------")
    # print(all_demonstrations)
    # # scene_init = {
    # #     "init_piles": [["A","B"], ["C", "D"], ["E", "F"]],
    # #     "goal_piles": [["A", "B", "C", "D", "E", "F"]],
    # # }
    
    # env.close()

    # benchmark = MazeBenchmark(5, 8, 5, 8)
    # approach = Learn2DecomposeApproach(
    #     benchmark.get_actions(),
    #     benchmark.get_next_state,
    #     benchmark.get_cost,
    #     benchmark.check_goal,
    # )
    # rng = np.random.default_rng(123)
    # task = benchmark.generate_tasks(1, "train", rng)[0]
    # plan = approach.generate_plan(task, "test", 1.0, rng)
    # assert plan_is_valid(plan, task, benchmark)
