# ChartGround-Edit 开发计划

原则：每个 Phase 单独提交、可独立验收；未满足当前 Phase 的验收条件，不进入下一阶段。任何完整训练都必须先通过 32 样本过拟合门槛。

## Phase 0：仓库理解与环境确认

### 目标

建立真实的 Sa2VA 源码地图，确认本机可用环境与资源边界，形成隔离式子项目骨架。

### 任务

- 阅读根 README、Sa2VA README、微调指南、核心模型、数据、配置、训练和评测入口。
- 记录数据 → MLLM → `[SEG]` hidden state → SAM2 → loss 的训练链路及 HF 推理链路。
- 只读检查 Python/CUDA/GPU、已有虚拟环境和已有本地权重；不得触发下载或安装。
- 建立根目录项目文档与 `projects/chartground_edit/` 目录。
- 列出模型选择、数据语义和编辑定义的待决问题。

### 产物

- `AGENTS.md`
- `PROJECT.md`
- `PLAN.md`
- `EXPERIMENTS.md`
- `projects/chartground_edit/README.md` 与空工程目录

### 验收条件

- 源码映射中的每项都有真实路径、类/函数名且可由 `rg` 定位。
- `git diff` 仅包含规划和骨架文件，无上游模型改动。
- 未启动训练、未下载模型/数据、未安装依赖。

## Phase 1：最小数据格式与可视化

### Phase 1A 实际状态（2026-09-15）

已完成本阶段的数据协议、32 条确定性合成数据、JSONL 校验与 Reader、同步 resize/crop/horizontal flip，以及原图/mask/overlay/目标裁剪和 contact sheet 闭环。默认 seed 为 `20260915`，16 个 `chart_type × referring_type` 组合各 2 条；相关测试为 9/9 通过。生成结果位于 `projects/chartground_edit/data/synthetic_v0/`（本地生成物受根 `.gitignore` 的 `data/` 规则影响，不作为大型数据提交）。Sa2VA 接入、模型推理和训练均未运行；后续 Phase 未标记完成。

### Phase 1B 实际状态（2026-09-15）

已完成独立于模型的 GT mask 编辑后端，支持 `highlight`、`recolor`、`extract`、`remove`，以及参数校验、直接运行的 CLI、四类图表展示 gallery 和像素级测试。Phase 1B 新增测试 31/31 通过，Phase 1A+1B 全量回归 40/40 通过。展示资产为 `projects/chartground_edit/assets/editing_v0_gallery.png`。本阶段仅验证 ground-truth mask；Sa2VA 接入、预测 mask、模型推理和训练均未运行。原计划 Phase 5 中与模型预测、编辑评测和完整 API/Demo 集成有关的工作仍未标记完成。

### 目标

定义可版本化的数据 schema，并用 32 个本地小样本验证图像、表达、mask 与元数据的一致性。

### 任务

- 冻结 v0 schema、mask 语义、类别词表、指代类型和 train/val/test 字段。
- 实现 schema 校验器与最小 dataset reader，不接入训练。
- 制作或生成 32 个轻量样本，覆盖四种图表和四种指代类型；不下载大型数据集。
- 实现原图、mask overlay、目标裁剪的可视化脚本。
- 实现共享几何变换的单元测试，包括 resize/crop/flip 后图像与 mask 对齐。
- 建立数据卡草案，记录许可证、来源、生成器版本和划分规则。

### 产物

- schema 示例及版本说明
- 32 样本 manifest
- 数据校验与可视化脚本
- 几何一致性测试和可视化检查图

### 验收条件

- 32/32 样本通过 schema、路径、尺寸、值域、非空 mask 校验。
- 固定 seed 下可重复读取/生成相同标注。
- 几何变换测试通过，人工抽查 overlay 无可见错位。
- 尚未加载 Sa2VA 权重或运行训练。

## Phase 2：Sa2VA 推理基线

### Phase 2A 实际状态（2026-09-15）

已完成只读的 Sa2VA 推理路径、硬件、环境、本地资源和官方 checkpoint 元数据审计，并形成 `projects/chartground_edit/docs/phase2_baseline_plan.md`。首个零样本候选规划为固定 revision 的 `ByteDance/Sa2VA-1B`（InternVL2.5、BF16），备选为 `ByteDance/Sa2VA-InternVL3-2B`；实际峰值显存仍待单样本验证。当前 `chartground` 环境缺少 PyTorch/transformers，本地也没有完整 Sa2VA checkpoint，因此尚不具备运行条件。本阶段没有安装依赖、下载权重、运行推理/训练或修改 Sa2VA 核心代码；Phase 2 推理未标记完成。

### Phase 2B-0 实际状态（2026-09-15）

用户已将首选固定为 `ByteDance/Sa2VA-InternVL3-2B` revision `15837dcaecc304714a1f0f069e74f47e47521c7f`，取代 Phase 2A 当时的 Sa2VA-1B 候选排序。已按 `setup_env.sh`、`projects/sa2va/pyproject.toml` 和 `uv.lock` 创建 `projects/sa2va/.venv`（物理目录 `/tmp/sa2va_env`），使用 InternVL3 对应的 `latest` 组，并完成关键依赖导入、7 张 RTX 3090 的轻量 CUDA 通信、checkpoint JSON/index/safetensors 文件头和本地 revision metadata 校验。ChartGround-Edit 回归为 40/40 通过，`compileall` 通过；没有加载模型、执行推理/训练或下载任何 checkpoint。环境和 checkpoint 细节见 `projects/chartground_edit/docs/phase2_environment.md`。当前 7 张 GPU 的显存占用均约 22.96 GiB，Phase 2B-1 必须等待可用 GPU 后再做 BF16 单样本实测；Phase 2 推理仍未标记完成。

### Phase 2B-1 实际状态（2026-09-15）

已在 `projects/chartground_edit/` 内实现模型无关的 `PredictionResult`、严格 mask 后处理、IoU/Dice/空预测/成功判定、懒加载的 `Sa2VAInternVL3Backend` 和单样本 CLI；没有修改 Sa2VA 上游源码。轻量回归为 65/65 通过。GPU 门槛复核后仅在物理 GPU 2（进程内逻辑 `cuda:0`）执行一次固定样本 `cge_bar_category_01`：模型文本为 `Sure, [SEG].<|im_end|>`，返回 1 个 `(1, 320, 480)` bool mask，非空且编辑成功，但与 GT 无交集，IoU/Dice 均为 0.0。模型加载 31,683.58 ms、`predict_forward` 1,223.36 ms，PyTorch 峰值 allocated memory 4,917.64 MiB。真实对比资产为 `projects/chartground_edit/assets/sa2va_2b_smoke.png`。这只证明单样本调用链可运行，不代表模型质量或完整 test split 已完成；Phase 2 baseline 仍未标记完成。

### Phase 2B-2 实际状态（2026-09-15）

已使用同一固定 checkpoint、BF16、Prompt 和 mask 后处理，对 `synthetic_v0` 的 4 条 test 样本进行一次顺序零样本运行；一个 backend 实例只加载模型一次，没有重试、Prompt 调优或 GT mask 编辑。4/4 推理调用成功并返回一个非空 `(1, 320, 480)` bool mask，编辑均成功；2/4 与 GT 有交集，2/4 为 nonempty disjoint。macro mean IoU/Dice 为 0.035667/0.064015，micro IoU/Dice 为 0.006158/0.012241；加载 9,586.27 ms，平均 `predict_forward` 782.74 ms，PyTorch peak allocated 5,004.33 MiB。结果与限制见 `projects/chartground_edit/docs/phase2_zeroshot_results.md`，真实总览为 `projects/chartground_edit/assets/sa2va_2b_zeroshot_test.png`。这是 4 条合成样本的诊断性基线，不代表最终统计结果；完整 Phase 2 和训练仍未标记完成。

### Phase 2C 实际状态（2026-09-16）

已完成 synthetic_v0 split 审计，确认生成顺序使 train/val/test 分别固定覆盖“非 category / category-highlight / category-recolor”，不具备平衡泛化评测条件。新增不读取 `target_attributes` 的 synthetic-v0 指令拆分、三个预注册 Prompt、val-only CLI 防护、配对汇总和 gallery；test/train 未运行。物理 GPU 1 上同一 Sa2VA-InternVL3-2B BF16 实例只加载一次，完成 4 条 val × 3 Prompt 的 12 次单次推理。三个 variant 均为 100% inference/SEG success、0% empty、25% nonempty-disjoint；P0/P1/P2 macro IoU 分别为 0.392236/0.340420/0.351172，macro Dice 为 0.437576/0.391375/0.428990。删除编辑动作没有表现出一致改善，中文 wrapper 不影响 `[SEG]` 输出但显著改变部分 mask 几何。P0 只能作为 future balanced synthetic_v1 val 的候选，不能视为最终 Prompt。详见 `projects/chartground_edit/docs/synthetic_v0_split_audit.md` 和 `projects/chartground_edit/docs/phase2_prompt_diagnostic.md`。本轮未生成 synthetic_v1、未训练，也未将 Phase 2 标记为全部完成。

### 目标

在不训练的前提下跑通单图指代分割基线，保存结构化输出和失败信息。

### 任务

- 经用户确认后选择本机已有或允许下载的一个 Sa2VA HF backbone。
- 实现薄适配器调用 `predict_forward`，不修改上游核心逻辑。
- 规范 prompt、`[SEG]` 检测、mask 数量匹配、原尺寸恢复和异常处理。
- 在 Phase 1 的 32 样本上运行定性推理，并实现可复现的 mIoU/cIoU 计算。
- 记录无 `[SEG]`、多 mask、空 mask、尺寸错误和 OOM 等失败类型。

### 产物

- 推理配置与 CLI
- 结构化预测文件、mask 和 overlay
- 基线评测脚本及实验记录

### 验收条件

- 单条和批量 32 样本均能完成，输出可追溯到 sample ID 和配置。
- mask 与原图尺寸一致，指标由脚本从预测和 GT 自动计算。
- 失败不会静默吞掉，且不宣称未测能力已经完成。

## Phase 3：科学图表指代分割数据集

### Phase 3A 实际状态（2026-09-16）

已实现并严格审计 `synthetic_v1`：320 条、train/val/test=192/64/64，16 个
`chart_type × referring_type` 组合各自为 12/4/4，四种 action 在组合内部严格平衡。
v1 schema/Reader 直接保存完整指令和指代表达，保持 v0 API 向后兼容；easy/medium/hard
全局为 106/107/107。seed、scene/content ID、style family 与 instruction template family
按 split 隔离，独立审计的硬失败、精确 image/mask 重复、空/全一/语义错误 mask 均为
0。低分辨率阈值 0.01 报告 97 对近重复人工复核候选，不自动删除。批量 PNG/manifest
受 `.gitignore` 保护，只提交 16 组合 gallery。本阶段未加载模型、未运行 synthetic_v1
val/test、未训练；Phase 3B Prompt benchmark、训练和最终 test 评测均未标记完成。

### 目标

建立规模可控、分布明确、可复现且许可清晰的训练/验证/测试集。

### 任务

- 实现四类图表的参数化合成与精确图元 mask 导出。
- 构造类别、外观、图例、趋势四类表达及等价改写，控制歧义。
- 增加交叉曲线、相近颜色、细线、重叠 band、压缩等难例。
- 设计按底层数据/模板族隔离的划分，做近重复和泄漏检查。
- 抽样人工复核真实图表子集，记录许可和标注规范。
- 输出数据统计、数据卡和版本号。

### 产物

- 数据生成器/转换器、manifest、数据卡
- train/val/test 划分和统计报告
- 自动校验、泄漏检查与可视化抽检报告

### 验收条件

- 同一底层数据或模板族不跨划分。
- 所有样本通过自动校验，抽检达到预先约定的标注一致性门槛。
- 四类图表、四类指代均有明确覆盖，分布统计可复现。

## Phase 4：LoRA/轻量微调

### 目标

以最小可训练参数验证模型能学习 ChartGround-Edit 的指代分割。

### 任务

- 创建独立配置，显式打印并保存 trainable/frozen 参数清单。
- 先做 1 样本 smoke test，再做 32 样本过拟合。
- 检查 `[SEG]` 数量、LLM loss、mask loss、Dice loss、梯度和 mask 可视化。
- 只有 32 样本可稳定过拟合后，提交完整训练预算供用户确认。
- 对比零样本、仅投影/decoder、LoRA 等受控设置；每次只改一个变量。

### 产物

- 轻量微调配置、启动/检查脚本
- 32 样本过拟合曲线、checkpoint 与预测可视化
- 参数清单、实验记录和可复现命令

### 验收条件

- 1 样本前后向无 NaN/Inf，目标模块有非零梯度。
- 32 样本达到预先写入实验记录的过拟合阈值，而非事后调整口径。
- 未经用户确认不运行完整训练。

## Phase 5：图表编辑模块

### 目标

将预测或真值 mask 转换为可控、确定性、可单测的四类编辑结果。

### 任务

- 定义 `highlight`、`recolor`、`extract`、`remove` 的参数与边界行为。
- 分离 segmentation adapter 与 editor，使编辑器可直接用 GT mask 单测。
- 处理 anti-alias、细线、mask feather/dilate 的显式可选策略。
- 保持非目标区域不变，并保存透明度、颜色和背景策略元数据。
- 实现 before/mask/after 拼图和 CLI/API 层。

### 产物

- 编辑器模块与四类操作
- 单元测试、示例结果和操作 schema
- 分割失败与编辑失败的独立错误报告

### 验收条件

- 四个操作均通过合成图的像素级测试。
- 非目标区域变化满足预设容差，输出尺寸/格式正确。
- 使用 GT mask 时编辑结果正确；模型误分割不得被误归因于编辑器。

## Phase 6：评测、Demo、README 和开源整理

### 目标

形成可复现、可解释、适合简历和开源展示的完整交付。

### 任务

- 固化分层评测、失败案例与消融表生成脚本。
- 构建轻量 Demo，展示图表、指令、mask、操作参数和前后对比。
- 完善安装、数据准备、推理、评测、局限性和许可证说明。
- 清理路径、密钥、私有数据引用和大文件，增加最小 CI。
- 从干净环境按文档复现 smoke test；检查全部 git diff 和仓库体积。

### 产物

- Demo、最终 README、数据卡/模型卡、评测报告
- 自动化 smoke test/CI、发布检查清单
- 简历项目描述与精选可视化案例

### 验收条件

- 新用户按 README 能完成不依赖私有路径的最小推理或 mock 演示。
- 指标、表格和图片均能追溯到 commit、数据版本、模型配置和脚本。
- 开源包不含权重、大型数据、密钥或许可证不允许再分发的内容。
