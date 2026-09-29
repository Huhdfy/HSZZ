from card_utils import Card

# 游戏规则中圣水的上限为 10；目前直接写死在圣水恢复逻辑中。

class PlayerState:
    """保存单个玩家在一局对战中的可变状态。

    ``cycle`` 按出牌循环顺序保存卡牌：前四张是当前手牌，索引 4 是下一张牌。
    """
    def __init__(self, player_id, cycle_queue, elixir, tower_hps=(4824, 3052, 3052)):
        self.player_id = player_id
        # 这里保留传入的队列对象本身；出牌时会原地更新它的顺序。
        self.cycle = cycle_queue
        self.elixir = elixir
        # 塔的顺序与默认生命值对应：国王塔、左侧公主塔、右侧公主塔。
        self.king_tower_hp, self.left_tower_hp, self.right_tower_hp = tower_hps
    
    def regenerate_elixir(self, dt: float, base_regen_time: float = 2.8):
        """按经过时间 ``dt`` 恢复圣水，并确保结果不超过 10。

        ``base_regen_time`` 表示恢复 1 点圣水所需秒数；不同对局阶段可通过
        传入更小的值来模拟双倍或三倍圣水。
        """
        elixir_per_second = 1.0 / base_regen_time
        self.elixir = min(10, self.elixir + elixir_per_second * dt)
    
    def can_play_card(self, card_name):
        """判断一张卡是否能被打出。

        必须同时满足：卡在前四张手牌中、圣水足够，并且己方国王塔尚未被摧毁。
        """
        return (card_name in self.cycle[:4] and
                self.elixir >= Card(card_name).elixir and
                self.king_tower_hp > 0)
    
    def play_card(self, card_name):
        """若出牌合法，扣除圣水并把该牌移到循环队列末尾。"""
        if not self.can_play_card(card_name): return False
        self.elixir -= Card(card_name).elixir
        # 被打出的手牌从当前位置移除，随后放到队尾，下一张牌会自然补入手牌。
        self.cycle.remove(card_name)
        self.cycle.append(card_name)
        return True

    def get_next_card(self):
        """返回循环中的下一张牌（约定为索引 4，要求队列至少有 5 张牌）。"""
        return self.cycle[4]
    
    def get_crown_count(self) -> int:
        """按皇家战争规则计算对手已获得的皇冠数。

        国王塔被摧毁立即计为 3 冠；否则每座被摧毁的公主塔计 1 冠。
        """
        if self.king_tower_hp <= 0:
            return 3
        return int(self.left_tower_hp <= 0) + int(self.right_tower_hp <= 0)
