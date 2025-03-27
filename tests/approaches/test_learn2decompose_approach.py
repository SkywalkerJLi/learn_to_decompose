"""Tests for learn2decompose_approach.py."""

import numpy as np
import networkx as nx
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
    G = nx.Graph()
    
    # Add nodes
    G.add_nodes_from(range(num_nodes))

    # Add edges
    G.add_edges_from(edges)

    # Find connected components
    connected_components = []
    for components in nx.connected_components(G):
        connected_components.append(list(components))
    return connected_components

def partition_to_key(partition):
    """Convert a partition (list of lists) into a hashable key (tuple of frozensets)."""
    return tuple(frozenset(subset) for subset in partition)

def create_partition_dict(partitions):
    """Efficiently map partitions to index numbers."""
    return {partition_to_key(partition): i for i, partition in enumerate(partitions)}

def test_learn2decompose_approach():
    """Tests Learn2Decompose planning in BlockStackingPyBulletBlocksEnv()."""

    np.random.seed(0)

    elements = [0, 1, 2, 3, 4, 5]
    # Generate all partitions and assign them fixed numbers
    all_partitions = list(multiset_partitions(elements, m=None))  # k=None means any number of groups
    print(all_partitions)

    # Assign a unique index to each partition
    index_to_partition = {i: part for i, part in enumerate(all_partitions)}

    partition_dict = create_partition_dict(all_partitions)

    env = SymbolicBlockStackingPyBulletBlocksEnv(use_gui=True)
    sim = SymbolicBlockStackingPyBulletBlocksEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/block-stacking-ttmp-test")
    max_motion_planning_time = 1  # increase for prettier videos

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

    # Number of distinct demonstrations to train on
    num_demonstrations = 50

    all_demonstrations = []
    for demo in range(num_demonstrations):
        print('START')
        seed = np.random.randint(0, 1000)
        rand_index = np.random.randint(0, len(init_configurations) + 1)
        init_piles = init_configurations[rand_index]
        print(init_piles)
        scene_init = {
            "init_piles": init_piles,
            "goal_piles": goal_pile,
        }
        
        obs, info = env.reset(
            seed = seed,
            options = scene_init
        )

        # print("OBSERVATIONS")
        # print(obs)
        init_state = env.get_state()
        # print("State representation")
        # print(init_state)

        #test_partition = partition_to_key([[1,2], [3,4],[5,6]])
        #print(partition_dict[test_partition])

        init_graph = init_state.to_observation()
        print("Graph Representation")
        print(init_graph)
        planner.reset(obs, info)

        print('Running demo: ', demo)

        demonstration = []
        for _ in range(10000):  # should terminate earlier
            action = planner.step(obs)
            obs, reward, done, _, _ = env.step(action)
            state = env.get_state()
            # print(state)
            scene_graph = state.to_observation()
            nodes = scene_graph.nodes
            edge_links = scene_graph.edge_links
            # print("nodes")
            # print(nodes)
            # print("edge-links")
            # print(edge_links)
            # print("--------------------------------")
            connected_components = find_connected_components(edge_links, len(nodes) - 1)
            # print(connected_components)
            scene_subgraph_id = partition_dict[partition_to_key(connected_components)]
            # print(scene_subgraph_id)
            demonstration.append(scene_subgraph_id)
            if done:  # goal reached!
                assert reward > 0
                break
        else:
            assert False, "Goal not reached"

        print(demonstration)
        print()
        all_demonstrations.append(demonstration)

    print("------------------------------------")
    print(all_demonstrations)
    # scene_init = {
    #     "init_piles": [["A","B"], ["C", "D"], ["E", "F"]],
    #     "goal_piles": [["A", "B", "C", "D", "E", "F"]],
    # }
    
    env.close()

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
