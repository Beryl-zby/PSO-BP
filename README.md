# 需水预测方法与测试结果

各方法代码按目录存放。统一使用2010—2023年数据：固定起点检验以2010—2019年训练、预测2020—2023年；滚动检验每次仅用此前历史。所有方法采用相同三个输入和末年实际值锚定，数值模型在训练期末3年选择参数；千问智能体读取训练前缀内验证误差和方向诊断，决定数值专家组合权重。

## 网页工作台（本地千问）

新增 [06_WaterLab_Web](06_WaterLab_Web/README.md)：上传数据、浏览器 PSO-BP 预测、交互图表、标准问答、多智能体工作流搜索与 MatPlotAgent Python 绘图。新增“Agent预测”通用入口：单变量或0–20驱动、最多20000条记录，Agent根据开发期验证调整预测模型并冻结后评价末段。支持连接已有本地千问 OpenAI 兼容服务，不需要云端密钥。按目录内的中文说明配置 `.env.local` 并运行 `pnpm dev`。

网页 Multi-agent 模式优化的是解释流程；与 `05_Multi_Agent` 中对数值专家分配权重的实验不同。EvoFlow/AFlow 为领域适配，不是完整论文复现；搜索评分不表示预测精度提升。

## 方法测试结果

MAPE单位为%，越低越好；“均值”是两项MAPE的描述性平均。

| 方法 | 固定起点MAPE | 逐年滚动MAPE | 两项均值 |
|---|---:|---:|---:|
| [正则PSO-BP集成](01_PSO_BP_Ensemble/README.md) | 0.8059 | 1.0354 | 0.9206 |
| [三个专家等权平均](05_Multi_Agent/README.md) | 0.9852 | 1.1661 | 1.0757 |
| [内验证误差倒数加权](05_Multi_Agent/README.md) | 1.0089 | 1.1647 | 1.0868 |
| [千问单智能体](05_Multi_Agent/README.md) | 1.0313 | 1.1474 | 1.0894 |
| [千问多智能体协作](05_Multi_Agent/README.md) | 1.1680 | 1.1629 | 1.1655 |
| [高斯过程回归](03_Gaussian_Process/README.md) | 1.4422 | 1.2266 | 1.3344 |
| [普通线性回归](04_Linear_Baselines/README.md) | 1.4380 | 1.2363 | 1.3372 |
| [岭回归（加密网格）](04_Linear_Baselines/README.md) | 2.2425 | 1.7773 | 2.0099 |
| [岭回归（原网格）](04_Linear_Baselines/README.md) | 3.3474 | 1.9496 | 2.6485 |
| [贝叶斯正则神经网络](02_Bayesian_BP/README.md) | 6.5316 | 1.9933 | 4.2624 |
| [最近实际值](04_Linear_Baselines/README.md) | 7.0258 | 1.9801 | 4.5029 |

当前按两项MAPE均值排序，最好的是正则PSO-BP集成。GPR与普通线性回归接近；GPR在105个未来检查点中有19个工业响应方向异常。贝叶斯正则网络本轮误差较大。千问多智能体为真实本地Qwen推理，单智能体、等权和误差倒数加权均作为对照；详细权重、原始回复及限制见[多智能体报告](05_Multi_Agent/results/结果报告.md)。

**以上是已知数据上的开发回测，2020—2023年已参与模型探索，不是新增独立测试。** 两套检验复用了同四个年份，不能视为8个独立样本。

## 目录

```text
00_Legacy_PSO_BP/       原始普通/受约束PSO-BP、历史图表与结果说明
01_PSO_BP_Ensemble/     改进PSO-BP完整代码、单独入口、独立结果
02_Bayesian_BP/         贝叶斯正则神经网络、单独入口、独立结果
03_Gaussian_Process/    高斯过程回归、单独入口、独立结果
04_Linear_Baselines/    岭回归、普通线性回归、持平基准及测试结果
05_Multi_Agent/         本地千问角色协作、单智能体及规则加权对照
06_WaterLab_Web/        上传、预测、可视化、本地千问与多智能体网页工作台
data/                  原始输入工作簿
shared/                共用数据读取、回测、选参和报告代码
tests/                 数值模型与多智能体验证程序
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

多智能体按[独立目录的运行步骤](05_Multi_Agent/README.md)先导出MATLAB证据，再在本地MLX环境执行千问和Python验证。

## 已有测试记录

- [多智能体验证](05_Multi_Agent/results/verification.json)：25次真实千问输出回放、训练期证据范围、4种策略的预测与误差重算。
- [三种方法验证](results/three_methods/verification.json)：56个误差值重算、保存模型复现、训练样本范围、前缀隔离、既有PSO和线性结果一致性均通过。
- [PSO-BP验证](results/improved/verification.json)：解析梯度、误差重算、前缀隔离、5组随机种子、常量输入及未来情景检查通过。
- [三种方法完整对比报告](results/three_methods/三种方法对比报告.md)：逐年预测、所选参数和未来方向检查。

## 数据与依赖

输入位于[data/需水预测模型对照.xlsx](data/需水预测模型对照.xlsx)，整理时未改动工作簿内容。工作簿内原有结果是旧模型记录，最新结果以各方法`results/`和统一对比报告为准。

PSO-BP改进路径只需要基础MATLAB；贝叶斯正则网络需要Deep Learning Toolbox；GPR需要Statistics and Machine Learning Toolbox。原始存档流程需要Optimization和Parallel Computing Toolbox。本机测试版本为MATLAB R2024a。

多智能体复用本地MLX版Qwen3-VL-4B-Instruct-4bit，依赖Apple Silicon和Python环境；模型权重及羽毛球项目本身不上传仓库。

## 最新WaterLab工作台与协作

网页已接入真实苏北需水输入、五种SSP与三条节水路径，以及本地千问Agent规划试算和同划分规则对照。朋友克隆本仓库后按 [WaterLab启动与协作说明](06_WaterLab_Web/README.md) 操作。基础Excel已在仓库data目录，机器专用配置和模型权重不随仓库分发。
