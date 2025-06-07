"""
CS2 Major Pick'Em 模拟器主程序
实现：瑞士轮比赛模拟、多进程加速、结果统计等功能

主要功能：
1. 模拟瑞士轮比赛过程
2. 计算队伍间的胜率
3. 统计不同战绩组合的出现频率
4. 多进程并行计算以提高效率
"""

from __future__ import annotations

import json
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import lru_cache, reduce
from multiprocessing import Pool
from os import cpu_count
from random import random
from statistics import median
from time import perf_counter_ns
from typing import TYPE_CHECKING
import numpy as np
import tqdm

from config import VRS_WEIGHT, HLTV_WEIGHT, SIGMA, win_probability

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

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
    rating: tuple[int, ...]

    def __str__(self) -> str:
        return str(self.name)

    def __hash__(self) -> int:
        return self.id


@dataclass
class Record:
    """
    记录队伍在比赛中的表现

    Attributes:
        wins: 胜场数
        losses: 负场数
        teams_faced: 已对阵的队伍集合
    """
    wins: int
    losses: int
    teams_faced: set[Team]

    @staticmethod
    def new() -> Record:
        """创建新的比赛记录"""
        return Record(wins=0, losses=0, teams_faced=set())

    @property
    def diff(self) -> int:
        """计算胜负场差"""
        return self.wins - self.losses


@dataclass
class Result:
    """
    记录模拟结果

    Attributes:
        three_zero: 3-0战绩的次数
        advanced: 3-1/3-2战绩的次数
        zero_three: 0-3战绩的次数
        pickem_results: 记录每个预测组合的出现次数
    """
    three_zero: int
    advanced: int
    zero_three: int
    pickem_results: dict[str, int]

    @staticmethod
    def new() -> Result:
        """创建新的结果记录"""
        return Result(
            three_zero=0,
            advanced=0,
            zero_three=0,
            pickem_results={}
        )

    def __add__(self, other: Result) -> Result:
        """
        合并两个结果记录

        Args:
            other: 另一个结果记录

        Returns:
            Result: 合并后的结果记录
        """
        # 合并pickem_results
        combined_pickems = {}
        for key in set(self.pickem_results.keys()) | set(other.pickem_results.keys()):
            combined_pickems[key] = self.pickem_results.get(key, 0) + other.pickem_results.get(key, 0)

        return Result(
            three_zero=self.three_zero + other.three_zero,
            advanced=self.advanced + other.advanced,
            zero_three=self.zero_three + other.zero_three,
            pickem_results=combined_pickems
        )


@dataclass
class SwissSystem:
    """
    瑞士轮系统

    Attributes:
        sigma: Elo公式的sigma值
        records: 所有队伍的比赛记录
        remaining: 仍在比赛中的队伍集合
    """
    sigma: tuple[int, ...]
    records: dict[Team, Record]
    remaining: set[Team]

    def __init__(self, sigma: tuple[int, ...], records: dict[Team, Record], remaining: set[Team]):
        self.sigma = sigma
        self.records = records
        self.remaining = remaining

    def seeding(self, team: Team) -> tuple[int, int, int]:
        """
        计算队伍的种子排名
        基于：胜负场差、Buchholz难度、初始种子排名

        Args:
            team: 目标队伍

        Returns:
            tuple[int, int, int]: (胜负场差, Buchholz难度, 种子排名)
        """
        # 计算Buchholz分数（所有对手的胜负场差之和）
        buchholz = sum(self.records[opp].diff for opp in self.records[team].teams_faced)

        # 第一轮使用初始种子排名
        if not self.records[team].teams_faced:
            return (0, 0, team.seed)
        else:
            # 中期使用计算出的排名
            # 1. 首先按胜负场差分组（1-0组、0-1组等）
            # 2. 在组内按Buchholz分数从高到低排序
            # 3. 同Buchholz分数时，按初始种子排名排序
            return (-self.records[team].diff, -buchholz, team.seed)

    def simulate_match(self, team_a: Team, team_b: Team) -> None:
        """
        模拟单场比赛
        """
        # 判断是否为BO3
        is_bo3 = self.records[team_a].wins == 2 or self.records[team_a].losses == 2

        # 计算单图胜率
        p = win_probability(team_a, team_b, self.sigma)

        # 生成随机数
        if is_bo3:
            random_values = np.random.random(3)
            first_map = p > random_values[0]
            second_map = p > random_values[1]
            team_a_win = p > random_values[2] if first_map != second_map else first_map
        else:
            random_values = np.random.random(1)
            team_a_win = p > random_values[0]

        # 更新队伍记录
        if team_a_win:
            self.records[team_a].wins += 1
            self.records[team_b].losses += 1
        else:
            self.records[team_a].losses += 1
            self.records[team_b].wins += 1

        # 记录对阵队伍
        self.records[team_a].teams_faced.add(team_b)
        self.records[team_b].teams_faced.add(team_a)

        # 处理晋级/淘汰
        if is_bo3:
            for team in [team_a, team_b]:
                if self.records[team].wins == 3 or self.records[team].losses == 3:
                    self.remaining.remove(team)

    def simulate_round(self, round_num: int) -> None:
        """
        模拟一轮比赛
        根据当前战绩将队伍分组，并安排对阵

        Args:
            round_num: 当前轮次
        """
        # 按战绩分组
        groups = {}  # 用字典存储不同战绩的组
        for team in self.remaining:
            diff = self.records[team].diff
            if diff not in groups:
                groups[diff] = []
            groups[diff].append(team)

        # 在每个组内按Buchholz分数从高到低排序，同分时按Initial Seed从低到高排序
        for diff in groups:
            groups[diff].sort(key=lambda t: (
                -sum(self.records[opp].diff for opp in self.records[t].teams_faced),  # Buchholz分数从高到低
                t.seed  # 同分时按Initial Seed从低到高排序（因为Initial Seed越小表示排名越高）
            ))

        def try_match_teams(teams: list[Team], round_num: int) -> list[tuple[Team, Team]]:
            """
            尝试按照优先级匹配队伍，同时考虑种子排名和回避原则

            Args:
                teams: 需要匹配的队伍列表
                round_num: 当前轮次

            Returns:
                list[tuple[Team, Team]]: 匹配结果列表
            """
            if not teams:
                return []

            n = len(teams)
            if n < 2:
                return []

            # 第一轮特殊处理
            if round_num == 1:
                matches = []
                # 高排名对阵低排名
                for i in range(n // 2):
                    matches.append((teams[i], teams[n - 1 - i]))
                return matches

            # 第二、三轮：高排名对阵低排名，但需要考虑回避原则
            if round_num <= 3:
                matches = []
                used_indices = set()

                # 从最高排名开始，寻找最低排名的未对阵队伍
                for i in range(n):
                    if i in used_indices:
                        continue

                    # 从最低排名开始寻找对手
                    for j in range(n-1, i, -1):
                        if j in used_indices:
                            continue

                        # 检查是否已经对阵过
                        if teams[j] not in self.records[teams[i]].teams_faced:
                            matches.append((teams[i], teams[j]))
                            used_indices.add(i)
                            used_indices.add(j)
                            break

                return matches

            # 第四轮和第五轮：使用优先级模式
            # 定义优先级匹配模式
            priority_patterns = [
                [(0,5), (1,4), (2,3)],  # 1v6 2v5 3v4
                [(0,5), (1,3), (2,4)],  # 1v6 2v4 3v5
                [(0,4), (1,5), (2,3)],  # 1v5 2v6 3v4
                [(0,4), (1,3), (2,5)],  # 1v5 2v4 3v6
                [(0,3), (1,5), (2,4)],  # 1v4 2v6 3v5
                [(0,3), (1,4), (2,5)],  # 1v4 2v5 3v6
                [(0,5), (1,2), (3,4)],  # 1v6 2v3 4v5
                [(0,4), (1,2), (3,5)],  # 1v5 2v3 4v6
                [(0,2), (1,5), (3,4)],  # 1v3 2v6 4v5
                [(0,2), (1,4), (3,5)],  # 1v3 2v5 4v6
                [(0,3), (1,2), (4,5)],  # 1v4 2v3 5v6
                [(0,2), (1,3), (4,5)],  # 1v3 2v4 5v6
                [(0,1), (2,5), (3,4)],  # 1v2 3v6 4v5
                [(0,1), (2,4), (3,5)],  # 1v2 3v5 4v6
                [(0,1), (2,3), (4,5)],  # 1v2 3v4 5v6
            ]

            # 尝试每种优先级模式
            for pattern in priority_patterns:
                matches = []
                used_indices = set()
                valid_pattern = True

                for i, j in pattern:
                    if i >= n or j >= n or i in used_indices or j in used_indices:
                        valid_pattern = False
                        break

                    # 检查是否已经对阵过
                    if teams[j] in self.records[teams[i]].teams_faced:
                        valid_pattern = False
                        break

                    matches.append((teams[i], teams[j]))
                    used_indices.add(i)
                    used_indices.add(j)

                if valid_pattern:
                    return matches

            # 如果没有找到合适的优先级模式，尝试任意匹配
            matches = []
            used_indices = set()

            # 高排名对阵低排名
            for i in range(n):
                if i in used_indices:
                    continue

                for j in range(n-1, i, -1):
                    if j in used_indices:
                        continue

                    if teams[j] not in self.records[teams[i]].teams_faced:
                        matches.append((teams[i], teams[j]))
                        used_indices.add(i)
                        used_indices.add(j)
                        break

            return matches

        # 第一轮特殊处理（使用初始种子排名）
        if len(groups.get(0, [])) == len(self.records):
            # 按初始种子排名排序
            teams = sorted(groups[0], key=lambda t: t.seed)
            for a, b in zip(teams, teams[len(teams) // 2 :]):
                self.simulate_match(a, b)
        else:
            # 每组内按排名匹配
            for diff in sorted(groups.keys(), reverse=True):
                matches = try_match_teams(groups[diff], round_num)
                for team_a, team_b in matches:
                    self.simulate_match(team_a, team_b)

    def simulate_tournament(self) -> None:
        """模拟整个比赛阶段，直到所有队伍都完成比赛"""
        round_num = 1
        while self.remaining and round_num <= 5:  # 限制最多5轮
            self.simulate_round(round_num)
            round_num += 1


class Simulation:
    """
    模拟器主类

    Attributes:
        sigma: Elo公式的sigma值
        teams: 参赛队伍集合
    """
    sigma: tuple[int, ...]
    teams: set[Team]

    def __init__(self, filepath: Path) -> None:
        """
        从JSON文件加载数据并初始化模拟器
        """
        with open(filepath) as file:
            data = json.load(file)

        def id_generator() -> Generator[int]:
            i = 0
            while True:
                yield i
                i += 1

        auto_id = id_generator()
        self.sigma = (SIGMA, SIGMA)
        self.teams = {
            Team(
                id=next(auto_id),
                name=team_k,
                seed=team_v["seed"],
                rating=tuple(
                    (eval(sys_v))(team_v[sys_k])
                    for sys_k, sys_v in data["systems"].items()
                ),
            )
            for team_k, team_v in data["teams"].items()
        }

    def batch(self, n: int) -> dict[Team, Result]:
        """
        运行n次模拟
        """
        results = {team: Result.new() for team in self.teams}
        all_combinations = {}

        # 预生成随机数
        random_values = np.random.random(n * 10)  # 预生成足够的随机数
        random_index = 0

        # 添加进度条
        with tqdm.tqdm(
            total=n,
            desc="模拟进度",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            position=0,
            leave=True
        ) as pbar:
            for sim_id in range(n):
                ss = SwissSystem(
                    sigma=self.sigma,
                    records={team: Record.new() for team in self.teams},
                    remaining=set(self.teams),
                )

                ss.simulate_tournament()

                # 记录当前模拟的结果
                three_zero_teams = []
                advanced_teams = []
                zero_three_teams = []

                for team, record in ss.records.items():
                    if record.wins == 3:
                        if record.losses == 0:
                            results[team].three_zero += 1
                            three_zero_teams.append(team.name)
                        else:
                            results[team].advanced += 1
                            advanced_teams.append(team.name)
                    elif record.wins == 0:
                        results[team].zero_three += 1
                        zero_three_teams.append(team.name)

                # 记录这次模拟的结果
                sim_result = {
                    '3-0': sorted(three_zero_teams),
                    '3-1/3-2': sorted(advanced_teams),
                    '0-3': sorted(zero_three_teams)
                }

                # 记录这个组合
                key = f"3-0: {', '.join(sim_result['3-0'])} | 3-1/3-2: {', '.join(sim_result['3-1/3-2'])} | 0-3: {', '.join(sim_result['0-3'])}"
                all_combinations[key] = all_combinations.get(key, 0) + 1

                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({'当前组合数': len(all_combinations)})

        # 将组合结果添加到第一个队伍的结果中
        results[list(self.teams)[0]].pickem_results = all_combinations

        return results

    def run(self, n: int, k: int) -> dict[Team, Result]:
        """
        使用k个进程运行n次模拟
        """
        # 计算每个进程的迭代次数
        iterations_per_process = n // k
        remaining_iterations = n % k
        
        # 创建任务列表
        tasks = []
        for i in range(k):
            # 最后一个进程处理剩余的迭代次数
            iterations = iterations_per_process + (remaining_iterations if i == k-1 else 0)
            tasks.append(iterations)
        
        # 创建进度条
        with tqdm.tqdm(
            total=n,
            desc="模拟进度",
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
            position=0,
            leave=True
        ) as pbar:
            # 创建进程池
            with Pool(k) as pool:
                # 使用imap处理任务，这样可以实时获取结果
                results = []
                for result in pool.imap(self.batch, tasks):
                    results.append(result)
                    # 更新进度条
                    pbar.update(iterations_per_process)
                    pbar.set_postfix({'当前组合数': len(result[list(self.teams)[0]].pickem_results)})

        # 合并所有进程的结果
        def _f(acc: dict[Team, Result], results: dict[Team, Result]) -> dict[Team, Result]:
            for team, result in results.items():
                acc[team] += result
            return acc

        return reduce(_f, results)


def format_results(results: dict[Team, Result], n: int, run_time: float) -> list[str]:
    """
    格式化模拟结果
    
    Args:
        results: 模拟结果
        n: 模拟次数
        run_time: 运行时间
    
    Returns:
        list[str]: 格式化的结果字符串列表
    """
    out = [f"已进行 {n:,} 次瑞士轮模拟"]

    # 输出组合统计
    all_combinations = list(results.values())[0].pickem_results
    sorted_combinations = sorted(all_combinations.items(), key=lambda x: x[1], reverse=True)
    filename = f"{VRS_WEIGHT:.4f}_{HLTV_WEIGHT:.4f}_{SIGMA:.4f}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for combination, count in sorted_combinations:
            f.write(f"{combination}: {count}/{n} ({count/n*100:.4f}%)\n")

    out.append(f"\n运行耗时: {run_time:.4f} 秒")
    return out


if __name__ == "__main__":
    # 直接使用2025_austin_stage_1.json文件
    file_path = "2025_austin_stage_2.json"
    n_iterations = 1000000  # 增加迭代次数以获得更准确的结果
    n_cores = max(1, cpu_count() - 1)  # 保留一个核心给系统使用

    # 运行模拟并打印格式化结果
    start = perf_counter_ns()
    results = Simulation(file_path).run(n_iterations, n_cores)
    run_time = (perf_counter_ns() - start) / 1_000_000_000
    print("\n".join(format_results(results, n_iterations, run_time)))