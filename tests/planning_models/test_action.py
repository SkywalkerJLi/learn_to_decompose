"""Tests for action.py."""

import pytest
from pybullet_blocks.envs.block_stacking_env import BlockStackingPyBulletBlocksEnv
from pybullet_blocks.envs.pick_place_env import PickPlacePyBulletBlocksEnv
from pybullet_blocks.planning_models.action import get_active_operators_and_skills
from pybullet_blocks.planning_models.perception import (
    PREDICATES,
    TYPES,
    BlockStackingPyBulletBlocksPerceiver,
    ClearAndPlacePyBulletBlocksPerceiver,
    PickPlacePyBulletBlocksPerceiver,
)
from task_then_motion_planning.planning import TaskThenMotionPlanner


def test_pick_place_pybullet_blocks_action():
    """Tests task then motion planning in PickPlacePyBulletBlocksEnv()."""

    env = PickPlacePyBulletBlocksEnv(use_gui=False)
    sim = PickPlacePyBulletBlocksEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/pick-place-ttmp-test")
    max_motion_planning_time = 5  # increase for prettier videos

    perceiver = PickPlacePyBulletBlocksPerceiver(sim)
    operators, skill_types = get_active_operators_and_skills()
    skills = {
        s(sim, max_motion_planning_time=max_motion_planning_time) for s in skill_types
    }

    # Create the planner.
    planner = TaskThenMotionPlanner(
        TYPES, PREDICATES, perceiver, operators, skills, planner_id="pyperplan"
    )

    # Run an episode.
    obs, info = env.reset(seed=123)
    planner.reset(obs, info)
    for _ in range(10000):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:  # goal reached!
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

    env.close()


def test_block_stacking_pybullet_blocks_action():
    """Tests task then motion planning in BlockStackingPyBulletBlocksEnv()."""

    env = BlockStackingPyBulletBlocksEnv(use_gui=False)
    sim = BlockStackingPyBulletBlocksEnv(env.scene_description, use_gui=False)

    # from gymnasium.wrappers import RecordVideo
    # env = RecordVideo(env, "videos/block-stacking-ttmp-test")
    max_motion_planning_time = 0.1  # increase for prettier videos

    perceiver = BlockStackingPyBulletBlocksPerceiver(sim)
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
        seed=1000,
    )
    planner.reset(obs, info)
    for _ in range(10000):  # should terminate earlier
        action = planner.step(obs)
        obs, reward, done, _, _ = env.step(action)
        if done:  # goal reached!
            assert reward > 0
            break
    else:
        assert False, "Goal not reached"

    env.close()
