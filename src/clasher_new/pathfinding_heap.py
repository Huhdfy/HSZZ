from pathlib import Path
import math
from core import Position
import heapq

# 静态车道图用字符区分地形；其行顺序与世界坐标的 y 轴方向相反。
grid_path = Path(__file__).with_name('tilemap_lane_grid.txt')
with grid_path.open('r') as f:
    contents = [list(each) for each in f.read().splitlines()]

# 18×32 的场地细分为 36×64 格，即每个寻路格边长为 0.5 世界单位。
# 缓存格子中心与八邻域，避免 A* 每步重复创建完全相同的对象。
cell_cache = {}
neighbor_cache = {}

def position_to_cell(position: Position):
    """把连续世界坐标映射到 0.5 单位的离散网格。

    例如 ``(3.75, 10.1)`` 映射为 ``(7, 20)``。向下取整让每个位置
    都唯一属于一个格子，乘以 2 则对应每单位长度分成两个小格。
    """
    x, y = position.x, position.y
    return math.floor(2*x), math.floor(2*y)

def cell_to_position(cell):
    """把网格索引转换为其中心点的 ``Position``，并缓存转换结果。

    搜索使用整数格坐标，战斗移动使用连续坐标；``+0.5`` 使路径点落在
    格子中央，而非两个格子的边界上。
    """
    if cell not in cell_cache:
        x, y = cell
        cell_cache[cell] = Position((x+0.5)/2, (y+0.5)/2)
    return cell_cache[cell]

def get_neighboring_points(x, y):
    """返回一个格子的最多八个合法邻居，包含对角线方向。

    邻接关系只依赖地图尺寸，所以可以缓存；某个邻居是否可走，会在主搜索
    循环中依据单位半径和建筑阻挡另行判断。
    """
    if (x, y) in neighbor_cache: return neighbor_cache[(x,y)]
    result = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            new_x, new_y = x+dx, y+dy
            if dx == dy == 0: continue
            if new_x < 0 or new_y < 0 or new_x >= 36 or new_y >= 64: continue
            result.append((new_x, new_y))
    neighbor_cache[(x,y)] = result
    return result


class EntityPathfinder:
    """使用最小堆实现的 A* 寻路器。

    终点不是目标实体的中心，而是单位进入攻击范围后可站立的格子；这使
    近战单位会停在目标旁边攻击，不会尝试走进目标的碰撞区域。
    """
    def __init__(self, entity, target, battle_state):
        # 复制坐标，避免实体在计算期间移动而改变本次搜索的起点或终点。
        self.start_position = Position(entity.position.x, entity.position.y)
        self.target_position = Position(target.position.x, target.position.y)
        self.target = target
        self.entity = entity
        self.start_cell = position_to_cell(self.start_position)
        self.battle = battle_state
        # ``goals`` 保存所有候选攻击格，随后从中挑选唯一的搜索终点 ``goal``。
        self.goals = set()
        self.goal = None

    def heuristic(self, cell):
        """估计当前格到终点的剩余代价（Chebyshev 距离 × 10）。

        因为允许八方向移动，Chebyshev 距离能近似剩余的最少步数。它只用于
        搜索优先级；实际已经走过的成本由 ``g`` 精确记录。
        """
        x, y = cell
        gx, gy = self.goal
        return 10 * max(abs(x - gx), abs(y - gy))

    def calculate(self):
        """计算到目标攻击范围内的可走路径，返回按行进顺序排列的坐标列表。

        过程是：枚举候选攻击格、选定终点、执行堆式 A* 搜索、沿 ``parent``
        回溯路径。返回列表包含起点与终点，供单位逐点移动。
        """
        self.goals = set()

        radius = self.target.data.collision_radius + self.entity.data.range
        # 攻击距离必须加上目标的碰撞半径，否则单位会试图走到目标中心。

        target_cell = position_to_cell(self.target_position)
        # 坐标缩放为两倍，因此扫描半径也乘以 2；额外一格避免边缘漏检。
        scan_radius = math.ceil(radius*2) + 1
        for x in range(target_cell[0]-scan_radius, target_cell[0]+scan_radius):
            for y in range(target_cell[1]-scan_radius, target_cell[1]+scan_radius):
                distance = cell_to_position((x, y)).distance_to(self.target_position)
                # 格子中心未必落在真实攻击边界。放宽半径可消除近战单位的离散误差。
                if distance < radius+0.375 and self.battle.pathfind_ground_walkable(cell_to_position((x, y)), self.entity.data.collision_radius):
                    self.goals.add((x, y))
        # 优先选择同时靠近目标和起点的候选格，以减少绕远路的可能。
        self.goal = min(self.goals, key=lambda c: cell_to_position(c).distance_to(self.target_position)+cell_to_position(c).distance_to(self.start_position))

        # g[n] 为起点到 n 的实际最小代价；f[n] = g[n] + 启发式估计。
        # parent[n] 记录以当前最优路线到达 n 的前驱，用于稍后回溯路径。
        g = {}
        f = {}
        parent = {}
        closed_set = set()
        g[self.start_cell] = 0
        f[self.start_cell] = self.heuristic(self.start_cell)
        # 不在堆内原地修改旧优先级；发现更优路线时压入新条目，弹出时过滤旧条目。
        open_heap = [(f[self.start_cell], self.start_cell)]

        while open_heap:
            current_f, current = heapq.heappop(open_heap)
            # 同一节点可能被多次压入堆：已展开或过期的条目均跳过。
            if current in closed_set:
                continue
            if current_f > f[current]:
                continue
            if current == self.goal:
                break
            closed_set.add(current)
            for neighbor in get_neighboring_points(current[0], current[1]):
                if neighbor in closed_set: continue
                neighbor_position = cell_to_position(neighbor)
                if not self.battle.pathfind_ground_walkable(neighbor_position, self.entity.data.collision_radius):
                    continue
                nx, ny = neighbor
                px, py = current
                # 文件第 0 行是地图顶端，因此 y 索引需要翻转。
                tile_char = contents[63-ny][nx]
                if tile_char == 'W':
                    # 飞行或具备跳河能力的单位可以低成本越过水域。
                    if self.entity.data.is_air_unit or self.entity.data.jump_speed:
                        tile_cost = 7
                    else:
                        tile_cost = 50
                elif tile_char == '.':
                    # 普通陆地的基础地形成本。
                    tile_cost = 8
                else:
                    # 桥等特殊地形成本更低，以引导单位选择正确渡河点。
                    tile_cost = 5
                if nx != px and ny != py:
                    # 14≈sqrt(2)×10，近似对角线移动比直线移动更长。
                    geo_cost = 14
                else:
                    geo_cost = 10
                step_cost = tile_cost * geo_cost
                tentative_g = g[current] + step_cost
                # 只有首次到达或找到更短路线时才更新前驱和堆优先级。
                if neighbor not in g or tentative_g < g[neighbor]:
                    g[neighbor] = tentative_g
                    parent[neighbor] = current
                    f[neighbor] = g[neighbor] + self.heuristic((nx, ny))
                    heapq.heappush(open_heap, (f[neighbor], neighbor))
        # 从终点沿前驱链反向回溯到起点，再翻转为真实行进方向。
        path = [current]
        while path[-1] != self.start_cell:
            path.append(parent[path[-1]])
        path.reverse()

        # 将离散格路径转换回战斗系统使用的连续世界坐标。
        positions = [cell_to_position(each) for each in path]
        return positions

if __name__ == '__main__':
    from battle import BattleState
    from player import PlayerState

    player_0_deck = ['Knight', 'MiniPekka', 'Arrows', 'Minions', 'Musketeer', 'Fireball', 'Giant', 'Archer']
    player_1_deck = ['Minions', 'Archer', 'MiniPekka', 'Musketeer', 'Giant', 'Fireball', 'Arrows', 'Knight']
    battle = BattleState(PlayerState(0, player_0_deck, 10), PlayerState(1, player_1_deck, 10))
    battle.deploy_card(0, 'Knight', Position(10.5, 10.5))

    pathfind = EntityPathfinder(battle.entities[7], battle.entities[2], battle)
    print(pathfind.calculate())






