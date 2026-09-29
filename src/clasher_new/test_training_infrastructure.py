"""Small integration checks for opponent sources and parallel PPO collection."""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import CREnv, random_strategy
from parallel_rollout import ParallelPPO, ParallelVecEnv
from train import CRFeatureExtractor
from training_opponents import OpponentPool


def _make_worker():
    return CREnv(opponent_model=random_strategy)


def test_checkpoint_opponents(tmp_path):
    source = DummyVecEnv([lambda: CREnv(opponent_model=random_strategy) for _ in range(4)])
    checkpoint = tmp_path / "history" / "old.zip"
    checkpoint.parent.mkdir()
    PPO("MultiInputPolicy", source, policy_kwargs={"features_extractor_class": CRFeatureExtractor},
        n_steps=8, batch_size=2, device="cpu", verbose=0).save(checkpoint)
    source.close()

    pool = OpponentPool(script_weight=0, history_weight=1, selfplay_weight=0,
                        history_dir=checkpoint.parent, seed=0)
    env = CREnv(opponent_model=pool)
    pool.bind_env(env)
    env.reset()
    assert pool.source == "history"
    assert pool.current.model.n_envs == 1
    assert pool.current.model.n_steps == 2
    assert pool.current.model.rollout_buffer.observations["grid"].shape[:2] == (2, 1)
    action = pool(env.observe(1))
    assert len(action) == 3 and 0 <= action[0] < 5

    snapshots = tmp_path / "selfplay"
    snapshots.mkdir()
    snapshot = snapshots / "new.zip"
    snapshot.write_bytes(checkpoint.read_bytes())
    selfplay = OpponentPool(script_weight=0, history_weight=0, selfplay_weight=1,
                            selfplay_dir=snapshots, seed=0)
    selfplay_env = CREnv(opponent_model=selfplay)
    selfplay.bind_env(selfplay_env)
    selfplay_env.reset()
    assert selfplay.source == "selfplay"
    assert selfplay.current.model.rollout_buffer.observations["grid"].shape[:2] == (2, 1)


def test_parallel_rollout_update():
    env = ParallelVecEnv([_make_worker, _make_worker])
    try:
        model = ParallelPPO("MultiInputPolicy", env,
                            policy_kwargs={"features_extractor_class": CRFeatureExtractor},
                            n_steps=2, batch_size=4, n_epochs=1, device="cpu", verbose=0)
        model.learn(total_timesteps=4)
        assert model.num_timesteps == 4
        assert np.isfinite(model.rollout_buffer.returns).all()
    finally:
        env.close()
