# 需水预测方法与测试结果

各方法代码按目录存放。统一使用2010—2023年数据：固定起点检验以2010—2019年训练、预测2020—2023年；滚动检验每次仅用此前历史。所有方法采用相同三个输入和末年实际值锚定，在训练期末3年选择参数。

## 方法测试结果

MAPE单位为%，越低越好；“均值”是两项MAPE的描述性平均。

| 方法 | 固定起点MAPE | 逐年滚动MAPE | 两项均值 |
|---|---:|---:|---:|
| [正则PSO-BP集成](01_PSO_BP_Ensemble/README.md) | 0.8059 | 1.0354 | 0.9206 |
| [高斯过程回归](03_Gaussian_Process/README.md) | 1.4422 | 1.2266 | 1.3344 |
| [普通线性回归](04_Linear_Baselines/README.md) | 1.4380 | 1.2363 | 1.3372 |
| [岭回归（加密网格）](04_Linear_Baselines/README.md) | 2.2425 | 1.7773 | 2.0099 |
| [岭回归（原网格）](04_Linear_Baselines/README.md) | 3.3474 | 1.9496 | 2.6485 |
| [贝叶斯正则神经网络](02_Bayesian_BP/README.md) | 6.5316 | 1.9933 | 4.2624 |
| [最近实际值](04_Linear_Baselines/README.md) | 7.0258 | 1.9801 | 4.5029 |

当前正则PSO-BP集成在两项总体误差上最好。GPR与普通线性回归接近；GPR在105个未来检查点中有19个工业响应方向异常。贝叶斯正则网络本轮误差较大。

**以上是已知数据上的开发回测，2020—2023年已参与模型探索，不是新增独立测试。** 两套检验复用了同四个年份，不能视为8个独立样本。

## 目录

```text
00_Legacy_PSO_BP/       原始普通/受约束PSO-BP、历史图表与结果说明
01_PSO_BP_Ensemble/     改进PSO-BP完整代码、单独入口、独立结果
02_Bayesian_BP/         贝叶斯正则神经网络、单独入口、独立结果
03_Gaussian_Process/    高斯过程回归、单独入口、独立结果
04_Linear_Baselines/    岭回归、普通线性回归、持平基准及测试结果
data/                  原始输入工作簿
shared/                共用数据读取、回测、选参和报告代码
tests/                 两套已有验证程序
results/improved/      PSO-BP详细对照、消融、五种子测试记录
results/three_methods/ 三种方案与线性基准的统一报告和图表
```

每个方法目录的README列出了该方法的实际测试结果、依赖和运行命令。当前打开的[模型改进报告](results/improved/模型改进报告.md)和[comparison.json](results/improved/comparison.json)保留原位置。

## 运行

在MATLAB中将当前目录设为项目根目录：

```matlab
S = main_ThreeMethodComparison;   % 运行三个方案并统一比较
V = test_ThreeMethodComparison;   % 核对指标、模型复现和数据隔离
```

只运行改进PSO-BP及线性对照：

```matlab
R = main_Water_ModelComparison;
V = test_Water_ImprovedComparison;
```

根目录入口会自动加载方法和共享目录。若只查看现有模型或单独执行测试，先运行`setup_water_paths`。也可进入各方法目录执行其`run_*.m`入口。

## 已有测试记录

- [三种方法验证](results/three_methods/verification.json)：56个误差值重算、保存模型复现、训练样本范围、前缀隔离、既有PSO和线性结果一致性均通过。
- [PSO-BP验证](results/improved/verification.json)：解析梯度、误差重算、前缀隔离、5组随机种子、常量输入及未来情景检查通过。
- [三种方法完整对比报告](results/three_methods/三种方法对比报告.md)：逐年预测、所选参数和未来方向检查。

## 数据与依赖

输入位于[data/需水预测模型对照.xlsx](data/需水预测模型对照.xlsx)，整理时未改动工作簿内容。工作簿内原有结果是旧模型记录，最新结果以各方法`results/`和统一对比报告为准。

PSO-BP改进路径只需要基础MATLAB；贝叶斯正则网络需要Deep Learning Toolbox；GPR需要Statistics and Machine Learning Toolbox。原始存档流程需要Optimization和Parallel Computing Toolbox。本机测试版本为MATLAB R2024a。
