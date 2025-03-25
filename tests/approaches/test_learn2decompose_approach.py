"""Tests for learn2decompose_approach.py."""

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


def test_learn2decompose_approach():
    """Tests Learn2Decompose planning in BlockStackingPyBulletBlocksEnv()."""

    env = SymbolicBlockStackingPyBulletBlocksEnv(use_gui=True)
    sim = SymbolicBlockStackingPyBulletBlocksEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/block-stacking-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

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
    obs, info = env.reset(
        seed=99,
    )
    print("OBSERVATIONS")
    print(obs)
    init_state = env.get_state()
    print("State representation")
    print(init_state)

    # if a block's pose is within a certain bound of another, then it is on top
    init_graph = init_state.to_observation()
    print("Graph Representation")
    print(init_graph)
    planner.reset(obs, info)
    for _ in range(10000):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, done, _, _ = env.step(action)
        state = env.get_state()
        # print(state)
        graph = state.to_observation()
        # print(graph)
        if done:  # goal reached!
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

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
