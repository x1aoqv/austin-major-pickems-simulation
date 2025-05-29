# 2025 Austin Major Pick'Em Simulator
# 2025 奥斯汀 Major 竞猜模拟器

This is a Python program for simulating the 2025 CS2 Austin Major. The program uses the Swiss system format to predict team performances through match simulations.

这是一个用于模拟 2025 CS2 Austin Major 的 Python 程序。该程序基于瑞士轮赛制,通过模拟比赛来预测各支队伍的表现。

## Project References
## 项目参考

This project is developed based on the following open-source projects:

本项目基于以下开源项目开发:

- [Major Pick'ems Simulator](https://github.com/ndunnett/major-pickems-sim) - Reference Project / 参考项目
- [CS2 Major Rules](https://github.com/ValveSoftware/counter-strike_rules_and_regs) - Swiss System Rules / 瑞士轮规则
- [CS2 Regional Standings](https://github.com/ValveSoftware/counter-strike_regional_standings) - VRS System / VRS系统

## Features
## 功能特点

- Monte Carlo Simulation / 蒙特卡洛模拟
  - Large-scale tournament simulation / 大规模比赛模拟
  - Probability-based match outcomes / 基于概率的比赛结果
  - Statistical analysis of results / 结果统计分析

- Greedy Algorithm / 贪心算法
  - Optimal Pick'Em combination search / 最优竞猜组合搜索
  - Probability-based team selection / 基于概率的队伍选择
  - Top 10 combinations ranking / Top 10 组合排名

- Swiss System Tournament / 瑞士轮赛制
  - Buchholz system implementation / Buchholz系统实现
  - Seeding-based matchmaking / 基于种子的对阵
  - Round-by-round progression / 逐轮晋级机制

- Multi-process Computing / 多进程计算
  - Parallel simulation execution / 并行模拟执行
  - CPU core utilization / CPU核心利用
  - Performance optimization / 性能优化

- Customizable Parameters / 可自定义参数
  - VRS/HLTV rating weights / VRS/HLTV评分权重
  - Sigma value adjustment / Sigma值调整
  - Team data configuration / 队伍数据配置

## Installation
## 安装说明

```bash
pip install -r requirements.txt
```

## Usage
## 使用方法

1. Configure Parameters / 配置参数:
   - Set team information and weights in `config.py` / 在 `config.py` 中设置队伍信息和权重
   - Adjust `VRS_WEIGHT` and `HLTV_WEIGHT` parameters / 调整 `VRS_WEIGHT` 和 `HLTV_WEIGHT` 参数
   - Set `SIGMA` value / 设置 `SIGMA` 值

2. Run Simulation / 运行模拟:
```bash
python simulate.py
```

3. View Results / 查看结果:
   - File naming format / 文件名格式: `VRS_WEIGHT_HLTV_WEIGHT_SIGMA.txt`

4. Solve Pick'Em Combinations / 竞猜组合求解:
```bash
python greedy.py
```

## Parameters
## 参数说明

- `VRS_WEIGHT`: VRS Rating Weight (Default: 0.7) / VRS 评分权重 (默认: 0.7)
- `HLTV_WEIGHT`: HLTV Rating Weight (Default: 0.3) / HLTV 评分权重 (默认: 0.3)
- `SIGMA`: Standard Deviation Parameter (Default: 349.2) / 标准差参数 (默认: 349.2)

## License
## 许可证

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 