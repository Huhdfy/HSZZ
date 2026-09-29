from dataclasses import dataclass

from reward_config import DEFAULT_REWARD_WEIGHTS, RewardWeights


@dataclass(frozen=True)
class _BattleSnapshot:
    own_tower_hp: float
    enemy_tower_hp: float
    own_crowns: int
    enemy_crowns: int


class RewardCalculator:
    """Calculates the reward produced between two environment steps."""

    def __init__(self, player_id=0, weights: RewardWeights = DEFAULT_REWARD_WEIGHTS):
        self.player_id = player_id
        self.opponent_id = 1 - player_id
        self.weights = weights
        self._previous = None

    def reset(self, battle):
        """Start reward tracking for a newly-created battle."""
        self._previous = self._snapshot(battle)

    def calculate(self, battle, action_attempted=False, action_succeeded=True):
        """Return the total reward since the previous call and update the baseline."""
        if self._previous is None:
            raise RuntimeError("RewardCalculator.reset() must be called before calculate().")

        current = self._snapshot(battle)
        previous = self._previous
        weights = self.weights

        reward = (
            weights.enemy_tower_destroyed * (current.enemy_crowns - previous.enemy_crowns)
            + weights.own_tower_destroyed * (current.own_crowns - previous.own_crowns)
            + weights.enemy_tower_damage * (previous.enemy_tower_hp - current.enemy_tower_hp)
            + weights.own_tower_damage * (previous.own_tower_hp - current.own_tower_hp)
        )

        if battle.game_over:
            reward += weights.win if battle.winner == self.player_id else weights.loss

        if action_attempted and not action_succeeded:
            reward += weights.invalid_action

        self._previous = current
        return reward

    def _snapshot(self, battle):
        own_player = battle.players[self.player_id]
        enemy_player = battle.players[self.opponent_id]
        return _BattleSnapshot(
            own_tower_hp=self._total_tower_hp(own_player),
            enemy_tower_hp=self._total_tower_hp(enemy_player),
            own_crowns=own_player.get_crown_count(),
            enemy_crowns=enemy_player.get_crown_count(),
        )

    @staticmethod
    def _total_tower_hp(player):
        return player.king_tower_hp + player.left_tower_hp + player.right_tower_hp
