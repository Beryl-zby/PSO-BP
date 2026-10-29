# WaterLab 需水预测工作台

当前指定模型为**正则BP集成（无PSO）**，版本 `regularized-bp-v2`。正则参数与末年、无锚定、末三年残差中位数三种校准方式在训练期内联合选择；回测和未来情景共用所选公式。工作台统一使用苏北五市研究输入，支持五种SSP×三条用水效率路径，预测2024—2030年需水量。

## 四个模块

| 模块 | 负责什么 |
|---|---|
| ① 数据与研究设置 | 输入Excel、研究背景、历史数据和未来情景 |
| ② Agent预测 | 数据诊断、候选方案与参数建议、程序验证、验证选优 |
| ③ BP预测 | 使用指定正则BP集成生成15组预测 |
| ④ 结果分析与绘图 | 选择BP结果、Agent结果或两者对比，查看与导出图表、数值和AI解读 |

第四模块只保留一套图表。用户决定预测方法、记录、SSP与路径；AI根据问题切换允许的图型并解释结果。输入版本不一致时不能直接对比；验证口径不同的误差不作排名。

## 首次本地启动

准备Node.js 24 LTS、pnpm和Ollama（或已有的OpenAI兼容本地模型服务）。

```bash
git clone https://github.com/Beryl-zby/PSO-BP.git
cd PSO-BP/06_WaterLab_Web
pnpm install --frozen-lockfile
npm run install:langgraph
ollama pull qwen3.5:4b-q4_K_M
```

将 `local-model.env.example` 复制为 `.env.local`。Windows PowerShell：

```powershell
Copy-Item local-model.env.example .env.local
powershell -ExecutionPolicy Bypass -File scripts/setup-plot-fonts.ps1
```

字体脚本从本机Windows字体目录配置宋体和Times New Roman；其他系统请将获授权的对应字体放入 `public/plot-fonts/`。字体文件不随仓库分发。

确认Ollama已启动；如未启动，在另一个终端运行 `ollama serve`。然后启动工作台：

```bash
pnpm dev
```

打开终端显示的地址，默认 [http://127.0.0.1:5173](http://127.0.0.1:5173)。普通BP数值预测不需要大语言模型；Agent规划与AI解读需要模型服务。修改 `.env.local` 后重启工作台。

## 模型服务配置

```dotenv
QWEN_PROVIDER=local
QWEN_BASE_URL=http://127.0.0.1:11434/v1
QWEN_MODEL=qwen3.5:4b-q4_K_M
QWEN_TIMEOUT_MS=600000
QWEN_REASONING_EFFORT=none
```

`QWEN_MODEL`应与服务实际提供的模型名称一致，可用 `ollama list` 核对。其他兼容服务按实际端口配置：LM Studio通常为 `http://127.0.0.1:1234/v1`；vLLM通常为 `http://127.0.0.1:8000/v1`。服务启用鉴权时填写 `QWEN_API_KEY`。`.env.local`、密钥、模型权重与依赖目录不提交Git。

## 数据和结果

- 源工作簿：[../data/需水预测模型对照.xlsx](../data/需水预测模型对照.xlsx)。上传使用程序支持的研究工作簿结构。
- 最新MATLAB结果：[../results/三路径预测_BP_最新](../results/三路径预测_BP_最新)。
- 启动或构建时自动生成 `public/research-input.json`；该文件不提交Git。
- `public/research-latest-result.json`保存与输入指纹匹配的MATLAB内置结果。浏览器重算与MATLAB可能因随机数实现而有微小差异。
- 工作台运行记录保存在各自浏览器中，Git同步不包含浏览器会话。更换输入后按输入指纹检查结果是否匹配。

三条路径为近期平均效率维持、历史趋势延续、江苏政策参考约束。政策作用于农业、工业强度；生活驱动没有额外政策约束。总量折算参考用于结果校核，不截断预测。当前BP开发回测指标与口径见[项目首页](../README.md)。

## 智能体与编排

| 角色 | 职责 |
|---|---|
| 建模规划智能体 | 根据数据诊断与背景提出模型、参数方案，读取验证反馈 |
| 预测执行智能体 | 请求调用受限数值工具，由程序完成训练、验证和预测 |
| 结果分析与可视化智能体 | 读取所选结果，解释数值并请求切换统一图表 |

各角色通过同一Qwen服务调用，职责由提示词和程序权限区分。LangGraph 1.5.3编排步骤，运行时由 `vendor/langgraph-runtime/package-lock.json`锁定。可选AFlow为受约束的工作流搜索适配，调整审核、反馈等步骤，以开发验证结果评价；并非完整原论文复现。旧MultiAgentPanel、EvoFlow和MatPlotAgent相关源码作为历史兼容保留，不构成当前第四模块的另一套绘图入口。

## 更新和检查

已有克隆更新：

```bash
git pull --ff-only
cd 06_WaterLab_Web
pnpm install --frozen-lockfile
npm run install:langgraph
pnpm dev
```

有本地修改时先提交或暂存，再拉取。相关检查：

```bash
node --test tests/bp-primary.test.mjs tests/result-view.test.mjs tests/research-figures.test.mjs tests/workspace-records.test.mjs
pnpm exec tsc --noEmit
pnpm build
```

`pnpm dev`用于本地开发启动。`pnpm start`使用托管构建配置，不替代上述首次启动步骤。网页服务未运行时访问127.0.0.1会提示连接被拒绝。
