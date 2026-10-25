# WaterLab

面向需水预测研究的网页工作台。上传年度数据，运行正则 PSO-BP 集成，查看回测、未来假设情景与驱动响应，并向千问分析助手提问。

## 本地启动（给使用者）

不需要 MATLAB，不需要 ChatGPT 登录，不需要云端千问密钥。先安装 **Node.js 24 LTS** 与 **pnpm 11.25**，并启动你已有的千问推理服务。

```bash
git clone https://github.com/Beryl-zby/PSO-BP.git
cd PSO-BP/06_WaterLab_Web
npm install -g pnpm@11.25.0
pnpm install
```

把本目录 `local-model.env.example` 复制为 `.env.local`，填写本地服务地址和**服务实际返回的模型名**，然后：

```bash
pnpm dev
```

浏览器打开终端显示的本机地址（默认 http://127.0.0.1:5173）。首次由本地开发登录自动进入，不会要求 ChatGPT 账号。无需把模型权重放进这个仓库。

配置示例：

```dotenv
QWEN_PROVIDER=local
QWEN_BASE_URL=http://127.0.0.1:11434/v1
QWEN_MODEL=填写你已有模型的名称
QWEN_TIMEOUT_MS=180000
```

| 已有服务 | QWEN_BASE_URL | 模型名怎么查 |
| --- | --- | --- |
| Ollama | `http://127.0.0.1:11434/v1` | `ollama list` |
| LM Studio | `http://127.0.0.1:1234/v1` | Developer 页或 `/v1/models` |
| vLLM / OpenAI 兼容服务 | `http://127.0.0.1:8000/v1`（按实际端口） | `/v1/models` |

如果本地服务设置了鉴权，额外填 `QWEN_API_KEY`。不设置鉴权时留空；程序不会把继承的云端密钥发给本地服务。服务若不支持 `response_format`，可加 `QWEN_JSON_MODE=off`，此时依赖模型按提示输出 JSON。只有模型权重文件还不能调用，需要先用已有推理工具启动 HTTP 服务。

**图像审查需要支持图片输入的 Qwen-VL 等视觉模型。** 纯文本模型可用于问答、多智能体与绘图代码生成；MatPlotAgent 完整视觉闭环需要视觉模型，能力不支持时会明确报错。绘图首次需联网从 jsDelivr 下载 Pyodide/Matplotlib；模型推理走本机，但首次运行并非完全离线。搜索约 30 次以上模型调用，4B 模型的耗时取决于机器和上下文设置。

启动无密钥也可上传数据、运行 PSO-BP、查看固定图表。修改 `.env.local` 后重启 `pnpm dev`。该命令是本机运行方式；`pnpm start` 面向原托管构建，不能代替此处的本地启动。此版本默认只监听本机，不需要给公网开放模型端口。

服务协议参考：[Ollama](https://docs.ollama.com/api/openai-compatibility)、[LM Studio](https://lmstudio.ai/docs/developer/openai-compat)、[vLLM](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/)。

## Agent 预测：通用数据入口

页面顶部选择 **Agent 预测**。原标准 PSO-BP 与 Multi-agent 分析模式保留，原上传入口仍适用于原三驱动年度数据。

通用入口读取第一张工作表，支持 CSV / XLSX / XLS（25MB以内）：

- 选择时间与目标列，可不选驱动，也可选1–20个数值驱动；支持负数和零目标。
- 时间可为等间隔数值、YYYY-MM、日期或 ISO 时间戳；支持12–20000条历史、1–120步预测。
- 历史目标必须完整；驱动缺失由候选配置选择训练期中位数或前值填补。只有末尾连续空目标行作为用户未来行。不会自动填补中间缺失目标。
- 不等间隔记录按记录步建模；没有用户未来时间时显示“未来第N期”，不声称预测了实际日历频率。日/月/小时的周期仅是可修改建议，不代表检测到了真实季节性。

执行流程：**开发期概要 → Agent提出候选 → 实际训练与多步验证 → Agent按验证反馈调整 → 冻结验证最优方案 → 末段测试 → 全历史重拟合与未来预测**。两次规划调用最多新增5个Agent候选，连同持平/漂移/适用季节基线，最多8个实际候选。没有配置模型服务时可运行“内置候选对照”，页面明确显示没有调用Agent。

支持最近实际值、线性漂移、季节持平、指数平滑、带滞后/可选外生驱动的岭回归，以及适合三驱动小样本的 ZBY PSO-BP。Agent可变更模型、滞后阶数、训练窗口、正则、平滑系数与驱动缺失处理。这里是受限实验规划与一次反馈迭代，未宣称实现任意 AutoML、深度网络搜索或原论文完整流程。PSO-BP是候选之一，不强制用于所有数据。

默认按时间保留最后20%（2–24步）作为本次搜索末段测试；之前的开发期再保留最后20%（2–24步）作验证，至少6条初始训练。每个候选只用初训拟合并递推预测整段验证，不用验证目标更新滞后。Agent只收到开发期统计及验证反馈；末段目标/统计不发送给规划接口。选中后先用开发期拟合并预测整段测试，再重拟合全历史预测未来；测试结果不会回流调参。历史数据上的时间留出仍不是新采集的独立数据集，选中模型也可能差于持平。

驱动预测可选训练末值或训练末24步线性趋势（均在每次训练截止处计算）；选择“用户提供”时，验证/测试使用对应时点驱动，假定它们事前已知，误差不包含外生驱动预测的不确定性，未来必须提供完整驱动。该假设在界面明确显示。

选择指标为 RMSE、MAE 或可处理零/负值的 sMAPE。均为原始目标单位上的误差（sMAPE为%）。没有概率校准区间。可导出全部候选、训练配置、验证/测试结果与输入的实验JSON，以及未来CSV；新入口的分析问答和 MatPlotAgent 按实际选中模型与时间字段工作。图表历史最多均匀抽取约300点，训练与数据导出保留全量。

验证命令：

```bash
node tests/adaptive.test.mjs
node tests/model-config.test.mjs
node --experimental-strip-types tests/multiagent.test.mjs
pnpm exec tsc --noEmit
pnpm build
```

新增测试包含单变量月度、四驱动负值/缺失数据、常量数据、20000条记录、无效候选，以及修改验证/测试标签后的信息隔离检查。

## 已实现

- CSV / XLSX / XLS 上传、字段映射、数据预览和数值校验。
- 识别原 Beryl-zby/PSO-BP 工作簿的“历史数据 / 未来指标 / 模型设置”布局，按原代码提取三个历史驱动。
- 浏览器 Worker 中的真实 PSO-BP：3-1-1 网络、12 成员、24 粒子、40 代、解析 Jacobian 局部优化、权重和偏置 L2 正则。
- 训练前缀最后 3 年内验证选择 λ；固定起点和逐年滚动回测；末年实测锚定。线性回归和最近实际值对照。
- 自定义未来驱动，或近 5 年驱动线性趋势；基准和农工驱动期末 ±5% 的假设情景。
- 趋势图、回测图、驱动响应图；代码计算越界、非正和响应方向诊断。
- 千问真实问答、自然语言图表切换、当前图表 PNG + 数值证据的多模态审查。
- CSV / JSON / PNG / 可打印 HTML 报告导出；当前浏览器中保留最近四次实验。

## 范围

标准模式使用固定交互图表；Multi-agent 模式实现 EvoFlow/AFlow 的受限领域适配，以及 MatPlotAgent 的浏览器 Python 执行流程。适配范围见文末，不是原论文完整复现。

原算法来自 https://github.com/Beryl-zby/PSO-BP 。主要对应：
01_PSO_BP_Ensemble/model_SmallSample_PSO_BP.m 和 shared/water_method_backtest.m。
浏览器使用 Mulberry32 随机序列，与 MATLAB Twister 不同，不保证逐位一致。
内置示例曾参与原项目开发，不是新增独立测试。两种回测使用相同年份。

## 数据格式

首行为标题。必需字段：年份、三个非负驱动、正需水量；至少 9 年历史、最多 400 行。字段名可在上传后映射。未来行的需水量留空，三个驱动仍需提供，年份必须晚于历史记录；最多 15 年。

public/waterlab-template.csv 是可直接导入的示例。普通文件读取第一张工作表；原项目工作簿只提取历史数据，原工作簿的 SSP 未来指标不在本版自动导入，页面会明确显示未来驱动来源。

年度间隔缺失时回测仍按记录顺序扩展；未来驱动趋势以实际数值年份拟合。本版不提供概率校准的预测区间，也不把统计驱动关系解释为因果效应。

## 运行

需要 Node.js 22.13+ 和 pnpm 11.25。

    pnpm run install:ci
    pnpm dev

服务端运行环境：

- DASHSCOPE_API_KEY：千问密钥；
- QWEN_BASE_URL：OpenAI 兼容接口的 /v1 根地址；
- QWEN_MODEL：例如 qwen3.5-flash。

托管密钥通过 Sites 的环境变量功能设置，不能放在浏览器代码或源仓库。本地预览直接读取启动进程的环境变量，开发配置仅在 serve 时注入。Sites 自带登录和私有访问控制；本地预览使用 starter 的本地模拟登录。

    node tests/engine.test.mjs
    pnpm exec tsc --noEmit
    pnpm build

测试覆盖解析 Jacobian、局部训练目标下降、常量输入、无效/重复记录、指标重算、未来标签隔离和保存的 MATLAB 开发指标对照。data/demo-result.json 由真实浏览器移植引擎计算生成。

上传数据在浏览器内解析并计算。提问时发送必要的结果摘要给配置的千问服务；图表审查额外发送当前图像。实验记录保存在浏览器 localStorage，不跨设备同步。

## Multi-agent 模式（新增）

网页顶部切换到 **Multi-agent**。预测仍用相同的 PSO-BP 引擎；此模式新增可运行、可检查的协作分析与科学绘图。

- **EvoFlow 领域适配**：两个初始角色图，按问题标签检索父代，通过千问生成交叉/变异图；实际执行候选，按精简/完整角色数保留生态位。问题阶段按标签从保留种群路由。所有角色目前使用同一个 Qwen 模型，未实现异构模型变异或原论文完整 Pareto/语义生态位机制。
- **AFlow 领域适配**：以 EvoFlow 的候选为起点，执行 3 轮 UCT 选择、受限图扩展、真实运行、评分和向祖先回传奖励。图可改变角色、分析并行结构和三种提示策略；不是原仓库的任意 Python 工作流代码搜索。
- **评价**：三个字段问题用于候选选择；冻结后两个不同字段问题用于复核，且不再据复核结果重新选择。字段通过率（相对容差 0.1%，最小绝对容差 0.0001）减去调用数与错误引用惩罚。该目标仅衡量结构化数值引用及调用数量，不证明回答整体质量、独立泛化、因果或预测精度提升。每个候选、输入图、评分检查、角色原文、耗时和 Tokens 均可下载。
- **MatPlotAgent 浏览器移植**：需求扩写→Qwen生成真实Python→Pyodide执行→失败时一次修复→多模态视觉审查→如有必要修订→对比择优。非固定图表切换。可下载真实 PNG、最终 Python 和对应 data.json。运行源码需 Python + numpy/pandas/matplotlib，或浏览器内 Pyodide 0.28.3。

MatPlotAgent 的执行运行在 `sandbox="allow-scripts"` 的不透明来源 iframe 中，内部 Worker 的 CSP 仅允许 Pyodide CDN 网络连接。宿主文件、主页面 DOM 和服务端密钥不传入该环境。Python AST 限制用于运行兼容性，不作为完整安全沙箱；浏览器的来源隔离与网络 CSP 是执行边界。运行超时或用户停止时会终止 Worker 并移除 iframe。首次从 jsDelivr 下载 Pyodide 与科学计算包；失败会明确报错，不用静态图冒充成功。单个图最大 2000×1600，PNG ≤1.7MB。

搜索/问答把当前计算证据提交给配置的千问；绘图另提交绘图数据、源码与生成图片用于审查。停止会取消当前客户端请求，已到达提供方的 LLM 调用可能仍计费。切换数据、重新预测或切换模式会取消并清理当前多智能体任务。候选搜索和图片当前仅在当前组件会话中保存，下载后可保留；原有实验记录仍保存数值预测。

参考：

- EvoFlow: https://arxiv.org/abs/2502.07373 （论文链接的代码仓库当前访问返回404，按公开论文机制实现受限适配）
- AFlow: https://github.com/FoundationAgents/AFlow
- MatPlotAgent: https://aclanthology.org/2024.findings-acl.701/
- STC 所用本地执行流程参考: https://github.com/cyt2023/MatplotAgent-fixed/blob/main/local_run.py （重写为浏览器执行流程，未直接运行原仓库CLI）
- Pyodide Worker: https://pyodide.org/en/0.28.3/usage/webworker.html

    node --experimental-strip-types tests/multiagent.test.mjs

新增测试覆盖数值引用评分、缺失/错误引用惩罚、调用成本、生态位保留、标签路由、UCT 探索与选择/复核字段分离。
