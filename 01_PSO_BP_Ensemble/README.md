# 正则PSO-BP集成

## 实测结果

| 检验 | MAPE (%) | MAE | RMSE | MSE |
|---|---:|---:|---:|---:|
| 固定起点 | 0.805875 | 1.336132 | 1.654967 | 2.738917 |
| 逐年滚动 | 1.035417 | 1.714478 | 2.076226 | 4.310715 |

两项MAPE均值：**0.920646%**。结果来源：[result.json](results/result.json)。

2019年训练所选配置：`lambda=0.01`；2023年全历史所选配置：`lambda=0.03`。

未来105个检查点：农业反向0、工业反向0、人口反向105、非正预测0、节水路径反序0。人口方向为统计响应，不作因果解释。

## 方法与运行

3→1→1网络；6个参数；12成员等权集成；PSO初始化和解析Jacobian训练；权重、偏置共同正则。

依赖：基础MATLAB。核心网络、训练代码及改进对照流程均放在本目录。

进入本目录，在MATLAB运行：

```matlab
R = run_PSO_BP_Ensemble;
```

从项目根目录运行：

```matlab
setup_water_paths;
R = run_PSO_BP_Ensemble;
```

输入统一取自`../data/需水预测模型对照.xlsx`；共享回测和数据读取位于`../shared/`。

## 结果与验证

- [详细报告](results/结果报告.md)：逐年预测、参数选择与情景检查。
- [保存的模型](results/result.mat)：固定、滚动及最终模型。
- 图表：[时间检验](results/时间检验.png)、[因素响应](results/因素响应.png)、[未来情景](results/未来情景.png)。
- [统一对比报告](../results/three_methods/三种方法对比报告.md)与[验证记录](../results/three_methods/verification.json)：保存模型复现、56项误差重算、样本范围及前缀隔离检查通过。

这是2020—2023已知数据上的开发回测，不是新增独立测试；所有方法使用相同输入、时间划分和实际值锚定。

本目录还保留`main_Water_ImprovedComparison.m`和`water_improved_backtest.m`；详细消融及5组随机种子记录位于[results/improved](../results/improved/模型改进报告.md)。
