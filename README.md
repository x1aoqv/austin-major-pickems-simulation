# CS2 Major Pick'Em 模拟器

这是一个用于模拟 2025 CS2 Austin Major 的 Python 程序。该程序基于瑞士轮赛制,通过模拟比赛来预测各支队伍的表现。

## 项目参考

本项目基于以下开源项目开发:

- [Major Pick'ems Simulator](https://github.com/ndunnett/major-pickems-sim) - 参考项目
- [CS2 Major 规则](https://github.com/ValveSoftware/counter-strike_rules_and_regs) - 瑞士轮规则
- [CS2 区域排名系统](https://github.com/ValveSoftware/counter-strike_regional_standings) - VRS系统

## 功能特点

- 支持瑞士轮赛制模拟
- 多进程并行计算
- 基于队伍历史数据的胜率预测
- 生成详细的比赛结果统计
- 支持自定义参数配置

## 安装说依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

1. 配置参数:
   - 在 `config.py` 中设置队伍信息和权重
   - 调整 `VRS_WEIGHT` 和 `HLTV_WEIGHT` 参数
   - 设置 `SIGMA` 值

2. 运行模拟:
```bash
python simulate.py
```

3. 查看结果:
   - 文件名格式: `VRS_WEIGHT_HLTV_WEIGHT_SIGMA.txt`

4. 竞猜组合求解
```bash
python greedy.py
```
## 参数说明

- `VRS_WEIGHT`: VRS 评分权重 (默认: 0.7)
- `HLTV_WEIGHT`: HLTV 评分权重 (默认: 0.3)
- `SIGMA`: 标准差参数 (默认: 349.2)

## 贡献指南

欢迎提交 Pull Request 或创建 Issue 来改进项目。

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。 