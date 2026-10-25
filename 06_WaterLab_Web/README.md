# WaterLab

面向需水预测研究的网页工作台。上传年度数据，运行正则 PSO-BP 集成，查看回测、未来假设情景与驱动响应，并向千问分析助手提问。

## 给协作者：克隆后启动最新版

需要 Node.js 24 LTS、pnpm 11.25 和可用的 Ollama（或其他OpenAI兼容本地模型服务）。模型权重不放GitHub，每位使用者在自己电脑安装。

```bash
git clone https://github.com/Beryl-zby/PSO-BP.git
cd PSO-BP/06_WaterLab_Web
pnpm install --frozen-lockfile
ollama pull qwen3.5:4b-q4_K_M
```

复制 `local-model.env.example` 为 `.env.local`。Windows PowerShell用 `Copy-Item local-model.env.example .env.local`，macOS/Linux用 `cp local-model.env.example .env.local`。确认Ollama已启动（没有运行时，在另一个终端执行 `ollama serve`），然后运行：

```bash
pnpm dev
```

打开终端显示的本机地址，默认 `http://127.0.0.1:5173`。每次启动或构建会从仓库的 `data/需水预测模型对照.xlsx` 生成网页输入副本，不需要任何原作者的F盘路径。`.env.local`、`node_modules`、模型权重和运行缓存不提交Git。

- **论文SSP预测**：原正则PSO-BP主模型，5种SSP×3条效率路径，105个年度值，图表和对照导出。
- **Agent预测**：自动载入同一真实研究数据和任务背景，千问提议候选、一次反馈调整，冻结后末段评价，另计算同划分规则候选对照与15组合。单次实测并未优于规则PSO-BP，不能直接当成预测精度改进。
- **标准分析 / Multi-agent**：原有功能保留；这些入口的演示情景不能冒充论文SSP。

修改基础数据请编辑仓库的源Excel，重新启动；也可在论文SSP页面上传同格式文件。生成的 `public/research-input.json` 不参与版本管理。若需把新输入用于Agent模块，先更新仓库Excel并重启，再点击“加载我的需水研究数据”。最近Agent试算在各自浏览器保存，输入指纹一致才恢复；朋友不会自动获得你浏览器中的会话记录。

已有克隆先 `git pull --ff-only`。有自己的未提交修改时先提交或暂存，再拉取并处理冲突。修改后跑测试，再提交到协作分支并发起Pull Request。不要提交密钥、权重、缓存和机器专用绝对路径。

```bash
node scripts/sync-research-input.mjs
node tests/research.test.mjs
node tests/research-adaptive.test.mjs
node tests/adaptive.test.mjs
node tests/model-config.test.mjs
node --experimental-strip-types tests/multiagent.test.mjs
pnpm exec tsc --noEmit
```

普通数值预测不需要LLM。Agent规划、问答需要模型服务；MatPlotAgent首次仍需联网下载浏览器Python运行库，完整视觉闭环没有作为本次发布的成功验证项。

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

## 论文SSP预测（本地接入）

默认进入“论文SSP预测”。启动时从模型目录的 `data/需水预测模型对照.xlsx` 读取“历史数据、未来指标、模型设置、口径与政策依据”，同步为浏览器可读副本。输入Excel保持不变；每次启动重新同步，运行期间也可上传同格式新Excel。“重新读取内置输入”只读最近同步副本。

点击“运行15种组合”，以相同正则PSO-BP最终模型计算SSP1—SSP5与三条路径，输出2024—2030共105个年度值。三驱动、工业指数拟合、2023锚定、2025比较基年与2030门槛均沿用 `shared/water_suite_data.m` 公式。人口、GDP、三产业、面积、工业增加值等保留在输入预览和明细。

- 效率保持：农业与工业强度固定为2023值，发展规模按SSP变化。
- 历史趋势：农业采用输入趋势，工业以2010—2023年2023价强度拟合指数速度，乘各SSP既定速度系数。
- 规划参考：农业同历史趋势；工业2025后取趋势与参考门槛较小值，2030较该SSP预测2025值下降至少10%（当前输入设置）。历史趋势已达标时两路径如实重合。

默认2023起点图，可切换2010历史起点、同SSP三路径或同路径五SSP。可下载PNG、年份纵向15组合横向的汇总CSV、输入预测明细CSV、发展节水对照CSV和完整实验JSON。CSV可用Excel打开。

单位换算同MATLAB；播种面积代理灌溉面积，工业价格采用省GDP平减代理，火电口径未完全统一。情景差值非因果政策效果。规划参考不认定苏北考核达标，2025比较值为预测而非实测。不输出未经验证的实际部门节水量。

模型为3-1-1、12成员、24粒子、40代，训练前缀内选L2正则；表内旧200成员/4隐节点设置不覆盖当前主模型。读取表的随机种子与政策参数。浏览器随机数不同于MATLAB，不保证逐位一致。2020—2023为已经用于探索的条件开发回测，非新增独立测试。

全部105组输入与独立复算一致。当前浏览器105个预测与已有MATLAB最大绝对差约0.00000402亿m³。通用Agent和标准分析入口保留，它们的示例不会作为论文输入。当前本地SSP接入尚未推送GitHub。

```bash
node scripts/sync-research-input.mjs
node tests/research.test.mjs
```

### Agent使用真实需水数据

Agent入口默认读取相同论文输入，自动设置7步、无季节周期、用户提供未来驱动，并带入苏北五市研究背景。可选择SSP和路径。规划仅接收开发期概要、研究任务与验证反馈，不包含末段目标。候选冻结后，另按相同模型推演全部15组合，并计算相同时间划分下的内置规则候选对照。无外生驱动模型的15组合相同，页面明确提示不能解读为SSP无影响。

默认14年数据划分为10年初训、2年验证、2年末段评价。与论文主模型原2020—2023回测口径不同，不能直接比较这两套误差。两者均属于已经探索过的历史数据，非新增独立验证。原主模型不被替换。

2026-10-26单次实测（不按测试结果重试）：本地4B千问两轮规划约48.8秒，选中线性漂移，末段RMSE约4.115亿m³；内置规则候选选择PSO-BP，RMSE约0.503亿m³。该次Agent未改善结果，全部候选和原始建议按实际显示；原始文字存在误判，以数值执行表为准。最近一次真实研究试算保存在本机浏览器，输入SHA256一致时恢复，可导出完整实验JSON及15组合CSV。

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
