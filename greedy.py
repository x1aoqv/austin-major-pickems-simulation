"""
CS2 Major Pick'Em 贪心算法
用于寻找最优的预测组合
"""

from collections import defaultdict
import re
import itertools
from typing import List, Dict, Tuple
from config import Team, win_probability, SIGMA
import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
from scipy import stats

matplotlib.use('TkAgg')

file_path = "0.0000_1.0000_349.2000.txt"  # 替换为您的文件路径

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


def generate_candidate_combinations(team_stats: dict, results: dict, top_n: int = 4) -> list:
    """生成候选预测组合"""
    candidates = []
    teams = list(team_stats.keys())
    
    # 1. 获取3-0概率最高的4个队伍
    top_3_0_teams = sorted(teams, key=lambda x: team_stats[x]['3-0'], reverse=True)[:3]
    
    # 2. 获取0-3概率最高的4个队伍
    top_0_3_teams = sorted(teams, key=lambda x: team_stats[x]['0-3'], reverse=True)[:3]
    
    # 3. 获取3-1/3-2概率最高的12个队伍
    top_adv_teams = sorted(teams, key=lambda x: team_stats[x]['3-1/3-2'], reverse=True)[:9]
    
    # 4. 生成所有可能的组合
    # 4.1 生成3-0组的组合（4选2）
    for three_zero in itertools.combinations(top_3_0_teams, 2):
        # 4.2 生成0-3组的组合（4选2）
        for zero_three in itertools.combinations(top_0_3_teams, 2):
            # 4.3 生成3-1/3-2组的组合（12选6）
            for adv in itertools.combinations(top_adv_teams, 6):
                # 检查是否有重复的队伍
                if (set(three_zero) & set(zero_three) or 
                    set(three_zero) & set(adv) or 
                    set(zero_three) & set(adv)):
                    continue
                
                candidates.append({
                    '3-0': set(three_zero),
                    '3-1/3-2': set(adv),
                    '0-3': set(zero_three)
                })

    # 6. 去重
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


def evaluate_combination_batch(combos: List[dict], results: dict) -> List[Tuple[dict, float, List[int]]]:
    """
    评估一批组合
    
    Args:
        combos: 要评估的组合列表
        results: 模拟结果字典
    
    Returns:
        List[Tuple[dict, float, List[int]]]: (组合, 概率, 正确数列表)
    """
    batch_results = []
    
    for combo in combos:
        correct_counts = []
        # 计算每个模拟结果中的正确数
        for (three_zero, three_one_two, zero_three), count in results.items():
            correct = 0
            # 检查3-0组
            correct += len(set(combo['3-0']) & set(three_zero))
            # 检查3-1/3-2组
            correct += len(set(combo['3-1/3-2']) & set(three_one_two))
            # 检查0-3组
            correct += len(set(combo['0-3']) & set(zero_three))
            # 添加正确数到列表
            correct_counts.extend([correct] * count)
        
        # 计算正确数>=5的概率
        correct_counts = np.array(correct_counts)
        prob = np.mean(correct_counts >= 5)
        
        # 只保留概率大于0的组合
        if prob > 0:
            batch_results.append((combo, prob, correct_counts.tolist()))
    
    return batch_results


def process_chunk(args):
    """处理一组组合"""
    chunk, results, shared_dict = args
    local_best_combo = None
    local_best_prob = -1
    local_correct_counts = None
    
    for combo in chunk:
        correct_counts = []
        # 计算每个模拟结果中的正确数
        for (three_zero, three_one_two, zero_three), count in results.items():
            correct = 0
            correct += len(set(combo['3-0']) & set(three_zero))
            correct += len(set(combo['3-1/3-2']) & set(three_one_two))
            correct += len(set(combo['0-3']) & set(zero_three))
            correct_counts.extend([correct] * count)
        
        # 计算正确数>=5的概率
        correct_counts = np.array(correct_counts)
        prob = np.mean(correct_counts >= 5)
        
        if prob > local_best_prob:
            local_best_combo = combo
            local_best_prob = prob
            local_correct_counts = correct_counts.tolist()
        
        # 更新进度
        with shared_dict['lock']:
            shared_dict['processed'] += 1
            if shared_dict['processed'] % 1000 == 0:  # 每处理1000个组合更新一次进度
                shared_dict['last_update'] = time.time()
    
    return local_best_combo, local_best_prob, local_correct_counts


def find_optimal_combination(file_path: str, teams: List[Team]) -> tuple:
    """
    寻找预测正确数≥5概率最大的组合
    
    Args:
        file_path: 模拟结果文件路径
        teams: 队伍列表
    
    Returns:
        tuple: (最优组合, 最优概率, 所有评估结果, 正确数分布)
    """
    # 解析模拟结果
    results, total_simulations = parse_simulation_results(file_path)
    tqdm.tqdm.write(f"已加载 {total_simulations} 个模拟结果")
    
    # 计算每个队伍在不同组别的概率
    team_stats = calculate_team_probabilities(teams)
    
    # 生成候选组合
    candidates = generate_candidate_combinations(team_stats, results)
    total_candidates = len(candidates)
    tqdm.tqdm.write(f"共生成 {total_candidates} 个候选组合")
    
    # 创建共享字典和锁
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['processed'] = 0
    shared_dict['best_prob'] = -1.0
    shared_dict['best_combo'] = None
    shared_dict['best_counts'] = None
    shared_dict['lock'] = manager.Lock()
    shared_dict['last_update'] = time.time()
    
    # 使用多进程处理
    start_time = time.time()
    n_cores = max(1, cpu_count() - 1)  # 保留一个核心给系统使用
    tqdm.tqdm.write(f"使用 {n_cores} 个CPU核心进行处理")
    
    # 创建进度条
    pbar = tqdm.tqdm(
        total=total_candidates,
        desc="评估进度",
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        position=0,
        leave=True
    )
    
    def update_progress():
        last_processed = 0
        while shared_dict['processed'] < total_candidates:
            current_processed = shared_dict['processed']
            if current_processed > last_processed:
                pbar.n = current_processed
                if shared_dict['best_combo']:
                    try:
                        current_combo = shared_dict['best_combo']
                        pbar.set_postfix({
                            '最佳概率': f"{shared_dict['best_prob']:.4f}",
                            '当前组合': f"3-0: {', '.join(sorted(current_combo['3-0']))}"
                        })
                    except:
                        pass
                pbar.refresh()
                last_processed = current_processed
            time.sleep(0.1)
    
    # 启动进度更新线程
    import threading
    progress_thread = threading.Thread(target=update_progress, daemon=True)
    progress_thread.start()
    
    try:
        # 将候选组合平均分配给每个进程
        chunk_size = total_candidates // n_cores
        chunks = [candidates[i:i + chunk_size] for i in range(0, total_candidates, chunk_size)]
        
        tqdm.tqdm.write("开始处理...")
        with Pool(n_cores) as pool:
            results = pool.map(process_chunk, [(chunk, results, shared_dict) for chunk in chunks])
        
        # 合并所有进程的结果
        best_combination = None
        best_probability = -1
        all_correct_counts = None
        
        for combo, prob, counts in results:
            if prob > best_probability:
                best_combination = combo
                best_probability = prob
                all_correct_counts = counts
        
        tqdm.tqdm.write("处理完成")
        
    except Exception as e:
        tqdm.tqdm.write(f"处理过程中出现错误: {str(e)}")
        raise
    finally:
        pbar.close()
    
    end_time = time.time()
    tqdm.tqdm.write(f"\n评估完成，耗时: {end_time - start_time:.2f} 秒")
    
    # 只返回最优组合
    evaluation_results = [(best_combination, best_probability)]
    
    return best_combination, best_probability, evaluation_results, all_correct_counts


def evaluate_combination(combo: dict, results: dict) -> Tuple[List[int], float]:
    """
    评估单个组合在模拟结果中的表现
    
    Args:
        combo: 要评估的组合
        results: 模拟结果字典
    
    Returns:
        Tuple[List[int], float]: (正确数列表, 正确数>=5的概率)
    """
    correct_counts = []
    total_simulations = sum(results.values())
    
    for (three_zero, three_one_two, zero_three), count in results.items():
        correct = 0
        correct += len(set(combo['3-0']) & set(three_zero))
        correct += len(set(combo['3-1/3-2']) & set(three_one_two))
        correct += len(set(combo['0-3']) & set(zero_three))
        correct_counts.extend([correct] * count)
    
    correct_counts = np.array(correct_counts)
    prob_ge5 = np.mean(correct_counts >= 5)
    
    return correct_counts, prob_ge5


def plot_distribution(correct_counts: List[int], best_probability: float):
    """
    绘制正确数分布函数
    
    Args:
        correct_counts: 所有组合的正确数列表
        best_probability: 最优组合的概率
    """
    plt.figure(figsize=(10, 6))
    
    # 计算分布
    counts, bins = np.histogram(correct_counts, bins=range(11), density=True)
    
    # 计算期望
    expected_value = np.mean(correct_counts)
    
    # 绘制直方图，>=5的部分用绿色，<5的部分用红色
    colors = ['red' if x < 5 else 'green' for x in bins[:-1]]
    plt.bar(bins[:-1], counts, width=0.8, alpha=0.7, color=colors)
    
    # 添加最优概率线和期望线
    plt.axvline(x=5, color='r', linestyle='--', label=f'P(X>=5)={best_probability:.4f}')
    plt.axvline(x=expected_value, color='blue', linestyle=':', 
                label=f'E[X]={expected_value:.2f}')
    
    # 设置图表属性
    plt.xlabel('X')
    plt.ylabel('Freq')
    plt.xticks(range(11))
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存为SVG格式
    plt.savefig('correct_counts_distribution.svg', format='svg')
    plt.close()


def main():
    """主函数，用于测试贪心算法"""
    from config import load_teams
    
    # 加载队伍数据
    teams = load_teams("2025_austin_stage_2.json")
    
    # 寻找最优组合
    best_combination, best_probability, all_results, correct_counts = find_optimal_combination(file_path, teams)
    
    # 绘制分布函数
    plot_distribution(correct_counts, best_probability)
    
    # 打印最优组合
    tqdm.tqdm.write("\n最佳竞猜组合:")
    candidate, prob = all_results[0]
    tqdm.tqdm.write(f"预测正确数 >= 5 的概率 = {prob:.4f}")
    tqdm.tqdm.write(f"  3-0 晋级: {', '.join(sorted(candidate['3-0']))}")
    tqdm.tqdm.write(f"  3-1/3-2 晋级: {', '.join(sorted(candidate['3-1/3-2']))}")
    tqdm.tqdm.write(f"  0-3 淘汰: {', '.join(sorted(candidate['0-3']))}")
    tqdm.tqdm.write("")


if __name__ == "__main__":
    main()