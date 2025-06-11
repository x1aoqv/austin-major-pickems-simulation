"""
CS2 Major Pick'Em 模拟器配置文件
包含：参数配置、队伍类定义、胜率计算、胜率矩阵生成等功能
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Tuple, List, Dict
import json
import sys

if TYPE_CHECKING:
    from pathlib import Path

# 评分系统权重配置
VRS_WEIGHT = 1.0  # Valve评分系统权重
HLTV_WEIGHT = 0.0  # HLTV评分系统权重
SIGMA = 349.2  # Valve默认sigma值，用于Elo公式计算

@dataclass(frozen=True)
class Team:
    """
    队伍类，存储队伍的基本信息和评分
    Attributes:
        id: 队伍唯一标识符
        name: 队伍名称
        seed: 种子排名
        rating: 评分元组 (VRS评分, HLTV评分)
    """
    id: int
    name: str
    seed: int
    rating: Tuple[int, ...]

    def __str__(self) -> str:
        return str(self.name)

    def __hash__(self) -> int:
        return self.id

@lru_cache(maxsize=None)
def win_probability(a: Team, b: Team, sigma: Tuple[int, ...] = (SIGMA, SIGMA)) -> float:
    """
    计算队伍a战胜队伍b的概率
    
    Args:
        a: 队伍a
        b: 队伍b
        sigma: Elo公式的sigma值，默认为(SIGMA, SIGMA)
    
    Returns:
        float: 队伍a的胜率 (0-1之间)
    """
    # 获取两支队伍的VRS和HLTV评分
    v1, h1 = a.rating[0], a.rating[1]
    v2, h2 = b.rating[0], b.rating[1]

    # 使用Elo公式计算VRS胜率
    d = sigma[0]
    p_vrs = 1 / (1 + 10 ** ((v2 - v1) / d))

    # 基于比例计算HLTV胜率
    p_hltv = h1 / (h1 + h2)

    # 计算加权平均胜率
    p = VRS_WEIGHT * p_vrs + HLTV_WEIGHT * p_hltv
    p /= (VRS_WEIGHT + HLTV_WEIGHT) if (VRS_WEIGHT + HLTV_WEIGHT) > 0 else 1

    return p

def calculate_win_matrix(teams: List[Team], sigma: Tuple[int, ...] = (SIGMA, SIGMA)) -> Dict[str, Dict[str, float]]:
    """
    计算所有队伍之间的胜率矩阵
    
    Args:
        teams: 队伍列表
        sigma: Elo公式的sigma值，默认为(SIGMA, SIGMA)
    
    Returns:
        Dict[str, Dict[str, float]]: 胜率矩阵，格式为 {队伍A: {队伍B: A胜B的概率}}
    """
    win_matrix = {}
    
    for team1 in teams:
        win_matrix[team1.name] = {}
        for team2 in teams:
            if team1 != team2:
                win_matrix[team1.name][team2.name] = win_probability(team1, team2, sigma)
    
    return win_matrix

def print_win_matrix(win_matrix: Dict[str, Dict[str, float]], teams: List[Team]) -> None:
    """
    打印胜率矩阵
    
    Args:
        win_matrix: 胜率矩阵
        teams: 队伍列表
    """
    print("胜率矩阵（行队名 vs. 列队名 -> 行队名获胜概率）:")
    
    # 固定列宽
    COLUMN_WIDTH = 10  # 每列固定10个字符宽度
    
    # 打印表头
    header = "队伍".center(COLUMN_WIDTH)
    for team in teams:
        # 如果队伍名太长，截断并添加省略号
        team_name = team.name
        if len(team_name) > COLUMN_WIDTH - 2:
            team_name = team_name[:COLUMN_WIDTH-3] + "..."
        header += team_name.center(COLUMN_WIDTH)
    print(header)
    
    # 打印分隔线
    separator = "-" * COLUMN_WIDTH
    for _ in teams:
        separator += "-" * COLUMN_WIDTH
    print(separator)
    
    # 打印每一行
    for team1 in teams:
        # 如果队伍名太长，截断并添加省略号
        team1_name = team1.name
        if len(team1_name) > COLUMN_WIDTH - 2:
            team1_name = team1_name[:COLUMN_WIDTH-3] + "..."
        row = team1_name.center(COLUMN_WIDTH)
        
        for team2 in teams:
            if team1 == team2:
                row += "-".center(COLUMN_WIDTH)
            else:
                win_rate = win_matrix[team1.name][team2.name]
                row += f"{win_rate:.2f}".center(COLUMN_WIDTH)
        print(row)

def load_teams(file_path: str) -> List[Team]:
    """
    从JSON文件加载队伍数据
    
    Args:
        file_path: JSON文件路径
    
    Returns:
        List[Team]: 队伍列表
    
    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON格式错误
    """
    with open(file_path) as file:
        data = json.load(file)
    
    teams = []
    for i, (team_name, team_data) in enumerate(data["teams"].items()):
        rating = tuple(
            (eval(sys_v))(team_data[sys_k])  # noqa: S307
            for sys_k, sys_v in data["systems"].items()
        )
        teams.append(Team(
            id=i,
            name=team_name,
            seed=team_data["seed"],
            rating=rating
        ))
    
    return teams

def main():
    """主函数，用于测试配置和函数"""
    # 加载队伍数据
    file_path = "2025_austin_stage_2.json"
    try:
        teams = load_teams(file_path)
        
        # 计算并打印胜率矩阵
        win_matrix = calculate_win_matrix(teams)
        print_win_matrix(win_matrix, teams)

    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
    except json.JSONDecodeError:
        print(f"错误：{file_path} 不是有效的JSON文件")
    except Exception as e:
        print(f"发生错误：{str(e)}")

if __name__ == "__main__":
    main() 