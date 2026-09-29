"""TensorBoard metrics for rewards, outcomes, and tower health during training."""

from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class TrainingMetricsCallback(BaseCallback):
    """Log rolling environment metrics once per PPO rollout."""

    INFO_METRICS = (
        "metrics/own_tower_hp_total",
        "metrics/opponent_tower_hp_total",
        "metrics/own_tower_hp_average",
        "metrics/opponent_tower_hp_average",
        "metrics/own_king_tower_hp",
        "metrics/opponent_king_tower_hp",
        "metrics/own_princess_tower_hp_average",
        "metrics/opponent_princess_tower_hp_average",
    )

    def __init__(self, window_size=1000, episode_window=100, verbose=0):
        super().__init__(verbose)
        self.window_size = window_size
        self.episode_window = episode_window
        self.step_rewards = deque(maxlen=window_size)
        self.episode_rewards = deque(maxlen=episode_window)
        self.episode_lengths = deque(maxlen=episode_window)
        self.wins = deque(maxlen=episode_window)
        self.tower_metrics = {
            name: deque(maxlen=window_size) for name in self.INFO_METRICS
        }

    def _on_step(self):
        self.step_rewards.extend(np.asarray(self.locals["rewards"]).reshape(-1).tolist())
        for info in self.locals.get("infos", ()):
            for name, values in self.tower_metrics.items():
                if name in info:
                    values.append(float(info[name]))
            episode = info.get("episode")
            if episode is not None:
                self.episode_rewards.append(float(episode["r"]))
                self.episode_lengths.append(float(episode["l"]))
                if "metrics/win" in info:
                    self.wins.append(float(info["metrics/win"]))
        return True

    def _on_rollout_end(self):
        if self.step_rewards:
            self.logger.record("monitor/step_reward_mean", float(np.mean(self.step_rewards)))
        if self.episode_rewards:
            self.logger.record("monitor/episode_reward_mean", float(np.mean(self.episode_rewards)))
            self.logger.record("monitor/episode_length_mean", float(np.mean(self.episode_lengths)))
        if self.wins:
            self.logger.record("monitor/win_rate", float(np.mean(self.wins)))
        for name, values in self.tower_metrics.items():
            if values:
                self.logger.record(name.replace("metrics/", "monitor/"), float(np.mean(values)))
