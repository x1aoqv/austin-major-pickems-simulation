"""
CS2 Major Pick'Em 贪心算法
用于寻找最优的预测组合
"""

from collections import defaultdict
import re
import itertools
from typing import List, Dict, Tuple
from config import Team, win_probability, SIGMA

file_path = "0.5000_0.5000_349.2000.txt"  # 替换为您的文件路径

def parse_simulation_results(file_path: str) -> dict:
    """
    解析模拟结果文件，返回每个组合及其出现频率的字典
    格式: {('3-0': set, '3-1/3-2': set, '0-3': set): frequency}
    """
    results = defaultdict(int)
    total_simulations = 0
    pattern = r"3-0: (.*?) \| 3-1/3-2: (.*?) \| 0-3: (.*?): (\d+)/\d+"

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(pattern, line)
            if match:
                three_zero = set(t.strip() for t in match.group(1).split(','))
                three_one_two = set(t.strip() for t in match.group(2).split(','))
                zero_three = set(t.strip() for t in match.group(3).split(','))
                count = int(match.group(4))

                key = (frozenset(three_zero), frozenset(three_one_two), frozenset(zero_three))
                results[key] += count
                total_simulations += count

    return results, total_simulations


def calculate_team_probabilities(teams: List[Team], sigma: Tuple[int, ...] = (SIGMA, SIGMA)) -> Dict[str, float]:
    """
    计算每个队伍获得3-0、3-1/3-2、0-3战绩的概率

    Args:
        teams: 队伍列表
        sigma: Elo公式的sigma值

    Returns:
        Dict[str, float]: 每个队伍获得不同战绩的概率
    """
    probabilities = {}

    for team in teams:
        # 计算3-0概率
        three_zero_prob = 1.0
        for other in teams:
            if other != team:
                three_zero_prob *= win_probability(team, other, sigma)

        # 计算0-3概率
        zero_three_prob = 1.0
        for other in teams:
            if other != team:
                zero_three_prob *= (1 - win_probability(team, other, sigma))

        # 计算3-1/3-2概率
        advanced_prob = 1 - three_zero_prob - zero_three_prob

        probabilities[team.name] = {
            '3-0': three_zero_prob,
            '3-1/3-2': advanced_prob,
            '0-3': zero_three_prob
        }

    return probabilities


def generate_candidate_combinations(team_stats: dict, results: dict, top_n: int = 5) -> list:
    """生成候选预测组合"""
    candidates = []
    teams = list(team_stats.keys())

    # 1. 贪心策略组合
    greedy_3_0 = sorted(teams, key=lambda x: team_stats[x]['3-0'], reverse=True)[:2]
    greedy_0_3 = sorted(teams, key=lambda x: team_stats[x]['0-3'], reverse=True)[:2]
    remaining = [t for t in teams if t not in greedy_3_0 and t not in greedy_0_3]
    greedy_adv = sorted(remaining, key=lambda x: team_stats[x]['3-1/3-2'], reverse=True)[:6]

    candidates.append({
        '3-0': set(greedy_3_0),
        '3-1/3-2': set(greedy_adv),
        '0-3': set(greedy_0_3)
    })

    # 2. 3-0组变种：选择3-0概率最高的top_n队伍中的组合
    top_3_0 = sorted(teams, key=lambda x: team_stats[x]['3-0'], reverse=True)[:top_n]
    for combo in itertools.combinations(top_3_0, 2):
        if set(combo) == set(greedy_3_0):
            continue
        remaining = [t for t in teams if t not in combo and t not in greedy_0_3]
        adv = sorted(remaining, key=lambda x: team_stats[x]['3-1/3-2'], reverse=True)[:6]
        candidates.append({
            '3-0': set(combo),
            '3-1/3-2': set(adv),
            '0-3': set(greedy_0_3)
        })

    # 3. 0-3组变种：选择0-3概率最高的top_n队伍中的组合
    top_0_3 = sorted(teams, key=lambda x: team_stats[x]['0-3'], reverse=True)[:top_n]
    for combo in itertools.combinations(top_0_3, 2):
        if set(combo) == set(greedy_0_3):
            continue
        remaining = [t for t in teams if t not in greedy_3_0 and t not in combo]
        adv = sorted(remaining, key=lambda x: team_stats[x]['3-1/3-2'], reverse=True)[:6]
        candidates.append({
            '3-0': set(greedy_3_0),
            '3-1/3-2': set(adv),
            '0-3': set(combo)
        })

    # 4. 3-1/3-2组变种：替换1-2个队伍
    top_adv = sorted(remaining, key=lambda x: team_stats[x]['3-1/3-2'], reverse=True)
    if len(top_adv) > 6:
        # 替换概率最低的1-2个队伍
        for to_remove in range(1, 3):
            new_adv = greedy_adv[:-to_remove] + top_adv[len(greedy_adv):len(greedy_adv) + to_remove]
            candidates.append({
                '3-0': set(greedy_3_0),
                '3-1/3-2': set(new_adv),
                '0-3': set(greedy_0_3)
            })

    # 5. 高概率组合：基于模拟结果中的高概率组合
    for (three_zero, three_one_two, zero_three), count in list(results.items())[:50]:
        candidates.append({
            '3-0': set(three_zero),
            '3-1/3-2': set(three_one_two),
            '0-3': set(zero_three)
        })

    # 去重
    unique_candidates = []
    seen = set()
    for cand in candidates:
        key = (frozenset(cand['3-0']), frozenset(cand['3-1/3-2']), frozenset(cand['0-3']))
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    return unique_candidates


def evaluate_candidate(candidate: dict, results: dict) -> float:
    """评估候选组合在模拟结果中的表现，返回正确数≥5的概率"""
    correct_ge5_count = 0
    total_simulations = sum(results.values())

    for (three_zero, three_one_two, zero_three), count in results.items():
        correct = 0

        # 检查3-0组
        for team in candidate['3-0']:
            if team in three_zero:
                correct += 1

        # 检查3-1/3-2组
        for team in candidate['3-1/3-2']:
            if team in three_one_two:
                correct += 1

        # 检查0-3组
        for team in candidate['0-3']:
            if team in zero_three:
                correct += 1

        if correct >= 5:
            correct_ge5_count += count

    return correct_ge5_count / total_simulations


def find_optimal_combination(file_path: str, teams: List[Team]) -> tuple:
    """
    寻找预测正确数≥5概率最大的组合

    Args:
        file_path: 模拟结果文件路径
        teams: 队伍列表

    Returns:
        tuple: (最优组合, 最优概率, 所有评估结果)
    """
    # 解析模拟结果
    results, total_simulations = parse_simulation_results(file_path)

    # 计算队伍概率
    team_stats = calculate_team_probabilities(teams)

    # 生成候选组合
    candidates = generate_candidate_combinations(team_stats, results)

    # 评估所有候选组合
    best_combination = None
    best_probability = -1
    evaluation_results = []

    for candidate in candidates:
        prob = evaluate_candidate(candidate, results)
        evaluation_results.append((candidate, prob))

        if prob > best_probability:
            best_probability = prob
            best_combination = candidate

    # 按概率排序
    evaluation_results.sort(key=lambda x: x[1], reverse=True)

    return best_combination, best_probability, evaluation_results


def main():
    """主函数，用于测试贪心算法"""
    from config import load_teams

    # 加载队伍数据
    teams = load_teams("2025_austin_stage_1.json")

    # 寻找最优组合
    best_combination, best_probability, all_results = find_optimal_combination(file_path, teams)

    # 打印前10个组合
    print("最佳竞猜组合:")
    for i, (candidate, prob) in enumerate(all_results[:10], 1):
        print(f"\nNO.{i}: 预测正确数 >= 5 的概率 = {prob:.4f}")
        print(f"  3-0 晋级: {', '.join(sorted(candidate['3-0']))}")
        print(f"  3-1/3-2 晋级: {', '.join(sorted(candidate['3-1/3-2']))}")
        print(f"  0-3 淘汰: {', '.join(sorted(candidate['0-3']))}")
        print()


if __name__ == "__main__":
    main()