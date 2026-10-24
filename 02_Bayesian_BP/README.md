# 贝叶斯正则神经网络

## 实测结果

| 检验 | MAPE (%) | MAE | RMSE | MSE |
|---|---:|---:|---:|---:|
| 固定起点 | 6.531570 | 10.782969 | 10.815006 | 116.964362 |
| 逐年滚动 | 1.993328 | 3.313452 | 4.927905 | 24.284251 |

两项MAPE均值：**4.262449%**。结果来源：[result.json](results/result.json)。

2019年训练所选配置：`hidden=2`；2023年全历史所选配置：`hidden=1`。

未来105个检查点：农业反向0、工业反向0、人口反向105、非正预测0、节水路径反序0。人口方向为统计响应，不作因果解释。

## 方法与运行

隐藏节点1/2/4内验证选择；每个结构12成员等权平均；trainbr自动平衡误差与参数惩罚。这是贝叶斯正则化训练，不是完整贝叶斯神经网络。

依赖：MATLAB Deep Learning Toolbox（fitnet、trainbr）。

进入本目录，在MATLAB运行：

```matlab
R = run_Bayesian_BP;
```

从项目根目录运行：

```matlab
setup_water_paths;
R = run_Bayesian_BP;
```

输入统一取自`../data/需水预测模型对照.xlsx`；共享回测和数据读取位于`../shared/`。

## 结果与验证

- [详细报告](results/结果报告.md)：逐年预测、参数选择与情景检查。
- [保存的模型](results/result.mat)：固定、滚动及最终模型。
- 图表：[时间检验](results/时间检验.png)、[因素响应](results/因素响应.png)、[未来情景](results/未来情景.png)。
- [统一对比报告](../results/three_methods/三种方法对比报告.md)与[验证记录](../results/three_methods/verification.json)：保存模型复现、56项误差重算、样本范围及前缀隔离检查通过。

这是2020—2023已知数据上的开发回测，不是新增独立测试；所有方法使用相同输入、时间划分和实际值锚定。

技术说明：[MathWorks trainbr](https://www.mathworks.com/help/deeplearning/ref/trainbr.html)。初始化间差异不解释为贝叶斯后验预测区间。
