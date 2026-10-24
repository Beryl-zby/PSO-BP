# 原始PSO-BP方法存档

本目录保留原始普通PSO-BP、受约束PSO-BP及旧图表。以下为输入工作簿“方法比较”页保存的历史结果，本次目录整理没有重新训练旧模型。

| 原始方法 | 固定MAPE (%) | 滚动MAPE (%) |
|---|---:|---:|
| 普通PSO-BP | 19.795580 | 4.931283 |
| 受约束PSO-BP | 6.086767 | 11.340811 |
| 岭回归 | 3.347435 | 1.949635 |

结构为3→4→1，每个网络21参数，正式集成200成员；受约束版本增加连接权重非负限制。

历史图表位于[results/图表](results/图表)。输入和历史结果来源：[原始工作簿](../data/需水预测模型对照.xlsx)。

若要重跑旧流程，在根目录执行：

```matlab
R = main_Water_ModelComparison([], [], true, true);
```

默认写入本目录`results/`，需要Optimization及Parallel Computing Toolbox。

当前改进版本见[01_PSO_BP_Ensemble](../01_PSO_BP_Ensemble/README.md)；其余新方案与测试结果见[根README](../README.md)。
