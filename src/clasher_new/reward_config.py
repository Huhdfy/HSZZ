from dataclasses import dataclass


@dataclass(frozen=True)
class RewardWeights:
    """Weights used by the environment reward calculation."""

    enemy_tower_destroyed: float = 5.0
    own_tower_destroyed: float = -5.0
    enemy_tower_damage: float = 0.001
    own_tower_damage: float = -0.0012
    win: float = 10.0
    loss: float = -10.0
    invalid_action: float = -0.05


DEFAULT_REWARD_WEIGHTS = RewardWeights()
