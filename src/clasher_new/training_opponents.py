"""Opponent sampling infrastructure independent of the learner policy and reward."""

from collections import OrderedDict
from pathlib import Path
import random

import numpy as np
from stable_baselines3 import PPO

from strategies import make_opponent_pool


class FrozenCheckpointOpponent:
    """Play a checkpoint without updating its weights during training."""

    def __init__(self, model):
        self.model = model

    def __call__(self, observation):
        expected = self.model.observation_space.spaces
        if set(expected) != set(observation):
            raise ValueError(
                "Opponent checkpoint observation keys do not match this environment: "
                f"expected {sorted(expected)}, got {sorted(observation)}"
            )
        action, _ = self.model.predict(observation, deterministic=False)
        return tuple(int(value) for value in np.asarray(action).reshape(-1)[:3])


class OpponentPool:
    """Choose one script, historical checkpoint, or self-play snapshot per game.

    Weights and source directories are supplied by the training entry point.
    Workers rescan the directories at reset, so new checkpoints become eligible
    after a game ends. Loaded models are cached locally in each worker.
    """

    def __init__(self, *, script_weight, history_weight, selfplay_weight,
                 history_dir=None, selfplay_dir=None, seed=None, max_cached=3,
                 scripts=None):
        self.weights = (float(script_weight), float(history_weight), float(selfplay_weight))
        if any(weight < 0 for weight in self.weights) or not any(self.weights):
            raise ValueError("Opponent weights must be nonnegative and not all zero")
        self.history_dir = Path(history_dir) if history_dir else None
        self.selfplay_dir = Path(selfplay_dir) if selfplay_dir else None
        self.rng = random.Random(seed)
        self.scripts = list(make_opponent_pool(spread=0.0, include_scatter=False)
                            if scripts is None else scripts)
        self.max_cached = max_cached
        self.cache = OrderedDict()
        self.incompatible = set()
        self.current = None
        self.source = None
        self.observation_space = None

    def bind_env(self, env):
        self.observation_space = env.observation_space

    @staticmethod
    def _checkpoints(directory):
        if directory is None or not directory.is_dir():
            return []
        return sorted(directory.glob("*.zip"))

    def _load(self, checkpoint):
        key = str(checkpoint.resolve())
        if key in self.incompatible:
            return None
        if key not in self.cache:
            # Saved training checkpoints retain n_envs/n_steps. Loading one for
            # inference would otherwise allocate the full training rollout buffer
            # in every worker (over 2 GiB per model with the current grid).
            model = PPO.load(key, device="cpu", n_envs=1, n_steps=2)
            if self.observation_space is not None and (
                set(model.observation_space.spaces) != set(self.observation_space.spaces)
                or any(model.observation_space[name].shape != space.shape
                       for name, space in self.observation_space.spaces.items())
            ):
                self.incompatible.add(key)
                return None
            self.cache[key] = FrozenCheckpointOpponent(model)
            if len(self.cache) > self.max_cached:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def reset(self):
        history = [path for path in self._checkpoints(self.history_dir)
                   if str(path.resolve()) not in self.incompatible]
        snapshots = [path for path in self._checkpoints(self.selfplay_dir)
                     if str(path.resolve()) not in self.incompatible]
        choices = []
        if self.scripts and self.weights[0]:
            choices.append(("script", self.weights[0]))
        if history and self.weights[1]:
            choices.append(("history", self.weights[1]))
        if snapshots and self.weights[2]:
            choices.append(("selfplay", self.weights[2]))
        if not choices:
            raise ValueError("No opponents available for the configured sources and weights")
        self.source = self.rng.choices(
            [name for name, _ in choices],
            weights=[weight for _, weight in choices],
        )[0]
        if self.source == "script":
            self.current = self.rng.choice(self.scripts)
        elif self.source == "history":
            self.current = self._load(self.rng.choice(history))
        else:
            # The newest snapshot is a frozen copy of the current learner.
            self.current = self._load(max(snapshots, key=lambda path: path.stat().st_mtime_ns))
        if self.current is None:
            if not self.scripts:
                raise ValueError("Selected checkpoint is incompatible and no script fallback exists")
            self.source = "script"
            self.current = self.rng.choice(self.scripts)
        reset = getattr(self.current, "reset", None)
        if callable(reset):
            reset()

    def __call__(self, observation):
        if self.current is None:
            self.reset()
        return self.current(observation)
