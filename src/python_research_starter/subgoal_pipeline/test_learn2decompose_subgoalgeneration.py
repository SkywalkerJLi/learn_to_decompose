"""Tests for learn2decompose_approach.py."""

import pickle

from pybullet_blocks.envs.symbolic_block_stacking_env import (
    SymbolicBlockStackingPyBulletBlocksEnv,
)

"""
    Tests Learn2Decompose planning in BlockStackingPyBulletBlocksEnv().
"""
def test_learn2decompose_approach():

    env = SymbolicBlockStackingPyBulletBlocksEnv(use_gui=True)

    goal_pile = [["B", "A"]]
    init_piles = [["A", "B", "C", "D", "E", "F"]]

    scene_init = {
        "init_piles": init_piles,
        "goal_piles": goal_pile,
    }

    _, _ = env.reset(seed=1, options=scene_init)

    init_state = env.get_state()
    init_graph = init_state.to_observation()
    print(init_graph)
    
    with open("abcdef.pkl", "wb") as f:
            pickle.dump(init_graph, f)
    env.close()
