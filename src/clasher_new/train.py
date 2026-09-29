from environment import CREnv, random_strategy, entity_names

from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from parallel_rollout import ParallelPPO, ParallelVecEnv
from training_opponents import OpponentPool
from training_metrics import TrainingMetricsCallback
import torch.nn as nn
import torch.nn.functional as F
import torch

import random
import numpy as np
import os

from pathlib import Path

# Opponent sources are independent of the policy network and reward function.
# Set these weights and directories to define the desired training mixture.
OPPONENT_WEIGHTS = dict(script_weight=1.0, history_weight=1.0, selfplay_weight=1.0)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_OUTPUT_DIR = Path(os.environ.get("TRAIN_OUTPUT_DIR", str(PROJECT_ROOT / "runs" / "default"))).expanduser()
TENSORBOARD_DIR = Path(os.environ.get("TENSORBOARD_LOG_DIR", str(TRAIN_OUTPUT_DIR / "tensorboard"))).expanduser()
CHECKPOINT_DIR = TRAIN_OUTPUT_DIR / "checkpoints"
FINAL_MODEL = TRAIN_OUTPUT_DIR / "latest.zip"
HISTORY_DIR = PROJECT_ROOT / "opponent_pool"
SELFPLAY_DIR = CHECKPOINT_DIR

class CRFeatureExtractor(BaseFeaturesExtractor):
    """将环境观测编码为 PPO 策略网络使用的特征。"""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.embedding_dim = 8
        self.entity_embedding = nn.Embedding(len(entity_names), self.embedding_dim)
        self.num_frames = int(observation_space["grid"].shape[0])
        self.per_frame_channels = 13 + self.embedding_dim + 4
        self.in_channels = self.num_frames * self.per_frame_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, stride=2), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, self.in_channels, 32, 18)
            cnn_out = self.cnn(dummy).shape[1]
        self.fc = nn.Linear(cnn_out + 5 * self.embedding_dim + 1 + 4 + 1, features_dim)

    def forward(self, observation):
        """
        Gets the observation, use the embedding (dim=8) to expand the channels, then use one-hot to further expand the channels.
        The code is ugly but should do the work.
        """
        grid = observation['grid']  # (B, 8, 32, 18, 15)
        hand = observation['hand'].long()  # (B, 5)
        elixir = observation['elixir']

        card_ids = grid[..., 0].long()
        card_vecs = self.entity_embedding(card_ids)

        rest = grid[..., 1:]  # (B, 32, 18, 14)
        x = torch.cat([rest, card_vecs], dim=-1)  # (B, 32, 18, 14+EMBED)
        card_type = x[..., 0].long()  # (B, 32, 18)
        card_type_oh = F.one_hot(card_type, num_classes=4).float()  # (B, 32, 18, 4)
        rest = x[..., 1:]
        x = torch.cat([rest, card_type_oh], dim=-1)
        x = x.permute(0, 1, 4, 2, 3).contiguous()
        x = x.flatten(1, 2).float()  # (B, frames * channels, 32, 18)

        grid_feat = self.cnn(x)

        hand_feat = self.entity_embedding(hand).flatten(1)  # (B, 5*EMBED)
        phase = observation['phase']
        if phase.ndim == 1 or phase.shape[-1] != 4:
            phase = F.one_hot(phase.long().reshape(-1), num_classes=4)
        phase = phase.float().reshape(phase.shape[0], -1)
        time_left = observation['time_till_next_phase'].float()
        combined = torch.cat([grid_feat, hand_feat, elixir.float(), phase, time_left], dim=1)
        return torch.relu(self.fc(combined))


class WeightsCopyingCallback(BaseCallback):
    """训练期间定期复制策略权重，供对手池采样使用。"""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.num_timesteps % 50000 == 0:
            opponent.policy.load_state_dict(self.model.policy.state_dict())
        return True

class RandomEvalCallback(BaseCallback):
    """训练期间定期对随机对手运行评估回合。"""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps % 50000 == 0:
            rewards = []
            eval_env = CREnv(opponent_model=lambda obs: random_strategy(obs))
            for i in range(5):
                obs, _ = eval_env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs)
                    obs, reward, termination, truncation, info = eval_env.step(action)
                    done = termination or truncation
                    total_reward += reward

                rewards.append(total_reward)
            self.logger.record("eval/mean_reward_vs_random", sum(rewards)/len(rewards))
        return True


def make_env(rank):
    """创建带独立随机种子的并行训练环境工厂函数。"""
    def factory():
        random.seed(10_000 + rank)
        np.random.seed(10_000 + rank)
        torch.set_num_threads(1)
        opponent = OpponentPool(
            **OPPONENT_WEIGHTS,
            history_dir=HISTORY_DIR,
            selfplay_dir=SELFPLAY_DIR,
            seed=10_000 + rank,
        )
        env = CREnv(opponent_model=opponent)
        opponent.bind_env(env)
        return env
    return factory


def main():
    n_envs = 16
    total_timesteps = int(os.environ.get("TRAIN_TOTAL_TIMESTEPS", "5000000"))
    if total_timesteps <= 0:
        raise ValueError("TRAIN_TOTAL_TIMESTEPS must be positive")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Training output: {TRAIN_OUTPUT_DIR.resolve()}", flush=True)
    print(f"TensorBoard logs: {TENSORBOARD_DIR.resolve()}", flush=True)
    env = ParallelVecEnv([make_env(rank) for rank in range(n_envs)])
    n_steps = 8192 // n_envs

    if not FINAL_MODEL.exists():
        print('Previous checkpoint does not exisiting, training new one from scratch.')
        model = ParallelPPO(
            "MultiInputPolicy",
            env,
            policy_kwargs={"features_extractor_class": CRFeatureExtractor},
            n_steps=n_steps,
            # 256 per environment
            batch_size=256,
            learning_rate=1e-4,
            n_epochs=4,
            target_kl=0.03,
            device="cpu",
            seed=0,
            verbose=1,
            tensorboard_log=str(TENSORBOARD_DIR),
        )
    else:
        model = ParallelPPO.load(str(FINAL_MODEL), env=env, device="cpu", learning_rate=1e-4, n_epochs=4,target_kl=0.03,ent_coef = 0.001,tensorboard_log=str(TENSORBOARD_DIR))
    cb = CheckpointCallback(save_freq=20_000 // n_envs, save_path=str(CHECKPOINT_DIR), name_prefix="cr")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            callback=[cb, TrainingMetricsCallback()],
        )
    finally:
        try:
            print('Saving model.')
            model.save(str(FINAL_MODEL))
        finally:
            env.close()


if __name__ == '__main__':
    main()
