from card_utils import Card

# 游戏规则中圣水上限为 10。

class PlayerState:
    """保存对局中单位玩家的牌序、圣水与三座塔的生命值。"""
    def __init__(self, player_id, cycle_queue, elixir, tower_hps=(4824, 3052, 3052)):
        self.player_id = player_id
        # 服务端状态复制牌组，避免调用者随后修改原列表影响本局。
        self.cycle = cycle_queue[:]
        self.elixir = elixir
        self.king_tower_hp, self.left_tower_hp, self.right_tower_hp = tower_hps
    
    def regenerate_elixir(self, dt: float, base_regen_time: float = 2.8):
        """按秒恢复圣水；``base_regen_time`` 是恢复 1 点所需时间。"""
        elixir_per_second = 1.0 / base_regen_time
        self.elixir = min(10, self.elixir + elixir_per_second * dt)
    
    def can_play_card(self, card_name):
        """卡必须在前四张手牌中、圣水足够，且国王塔尚未被摧毁。"""
        return (card_name in self.cycle[:4] and
                self.elixir >= Card(card_name).elixir and
                self.king_tower_hp > 0)
    
    def play_card(self, card_name):
        """执行合法出牌：扣除圣水，并将该牌轮转到牌组末尾。"""
        if not self.can_play_card(card_name): return False
        self.elixir -= Card(card_name).elixir
        self.cycle.remove(card_name)
        self.cycle.append(card_name)
        return True

    def get_next_card(self):
        """返回索引 4 的下一张牌（队列约定含至少五张牌）。"""
        return self.cycle[4]
    
    def get_crown_count(self) -> int:
        """国王塔被摧毁为 3 冠，否则每座被摧毁的公主塔为 1 冠。"""
        if self.king_tower_hp <= 0:
            return 3
        return int(self.left_tower_hp <= 0) + int(self.right_tower_hp <= 0)

if __name__ == '__main__':
    deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
    p = PlayerState(0, deck, 10)
    print(p.cycle, p.elixir)
    p.play_card('Knight')
    print(p.cycle, p.elixir)
    p.play_card('Knight')
    print(p.cycle, p.elixir)
