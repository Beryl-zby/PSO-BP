# 高斯过程回归

## 实测结果

| 检验 | MAPE (%) | MAE | RMSE | MSE |
|---|---:|---:|---:|---:|
| 固定起点 | 1.442153 | 2.381443 | 2.414765 | 5.831088 |
| 逐年滚动 | 1.226647 | 2.026428 | 2.585431 | 6.684454 |

两项MAPE均值：**1.334400%**。结果来源：[result.json](results/result.json)。

2019年训练所选配置：`squaredexponential/linear`；2023年全历史所选配置：`squaredexponential/linear`。

未来105个检查点：农业反向0、工业反向19、人口反向105、非正预测0、节水路径反序0。人口方向为统计响应，不作因果解释。

## 方法与运行

精确GPR；平方指数/Matérn 3/2核配常数/线性均值，共4种候选；训练期拟合核和噪声参数，内验证选择候选。

依赖：MATLAB Statistics and Machine Learning Toolbox（fitrgp）。

进入本目录，在MATLAB运行：

```matlab
R = run_Gaussian_Process;
```

从项目根目录运行：

```matlab
setup_water_paths;
R = run_Gaussian_Process;
```

输入统一取自`../data/需水预测模型对照.xlsx`；共享回测和数据读取位于`../shared/`。

## 结果与验证

- [详细报告](results/结果报告.md)：逐年预测、参数选择与情景检查。
- [保存的模型](results/result.mat)：固定、滚动及最终模型。
- 图表：[时间检验](results/时间检验.png)、[因素响应](results/因素响应.png)、[未来情景](results/未来情景.png)。
- [统一对比报告](../results/three_methods/三种方法对比报告.md)与[验证记录](../results/three_methods/verification.json)：保存模型复现、56项误差重算、样本范围及前缀隔离检查通过。

这是2020—2023已知数据上的开发回测，不是新增独立测试；所有方法使用相同输入、时间划分和实际值锚定。

JSON额外保存未锚定预测的模型标准差，未把它直接解释为锚定后预测的95%区间。工业响应存在反向点，节水情景解释需结合详细报告。技术说明：[fitrgp](https://www.mathworks.com/help/stats/fitrgp.html)。
