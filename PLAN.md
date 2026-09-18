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

### Phase 3B 实际状态（2026-09-16）

已按冻结协议在 synthetic_v1 的 64 条 balanced val 上完成 P0/P1/P2 共 192 次真实
Sa2VA-InternVL3-2B BF16 推理；单进程、单 GPU、模型只加载一次，全部 attempt 完成且
protocol hash 前后相同。P0/P1/P2 的 16-group Macro IoU 为
0.164503/0.179093/0.182199，按预注册 primary metric 选择 P2 `target_only_zh` 作为后续
frozen-test 的唯一全局 Prompt。P2−P1 的 paired bootstrap 95% CI 为
[-0.006147, 0.015220]，包含 0，因此只称为 selected on validation，不声称显著优势。
所有文本均含 `[SEG]`，但 empty rate 为 43.75%–45.31%，协议成功不等于分割准确。
本阶段未运行 synthetic_v1 train/test，未训练或微调；Phase 3C frozen test baseline
尚未开始、未标记完成。

### Phase 3C 实际状态（2026-09-16）

已在独立冻结协议下，仅使用 Phase 3B 选定的全局 P2 `target_only_zh`，对
`synthetic_v1` 64 条 test 各执行一次 Sa2VA-InternVL3-2B BF16 推理。模型加载一次，
backend 调用 64/64，未比较 P0/P1、未重跑 val/train、未选择性重试，protocol 前后
SHA-256 均为 `3e2aa95825b369746db5bd667912bbfd74d0288c5c9a5ddb9c75ce3bb3593478`。
execution、mask contract 和 `[SEG]` 均为 64/64；空预测 27/64，nonempty-disjoint
7/64。16-group Macro IoU/Dice 为 0.198033/0.242366，Micro IoU/Dice 为
0.225301/0.367748。37 个非空预测全部由 predicted mask 成功执行真实 action/parameters，
27 个空预测明确跳过。64 个保存 mask 的二值性、尺寸、IoU/Dice 和编辑输入来源独立
复核通过。该结果只建立 frozen synthetic test baseline，不触发 Prompt 修改；训练与
微调仍未开始。

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

### Phase 4A 实际状态（2026-09-16）

已完成官方 InternVL3-2B/LoRA、dataset、`[SEG]` hidden state、SAM2 与 loss/save/convert
路径的源码审计；冻结 train-only smoke1（1 条）与 overfit32（16 组合各 2 条），并新增
不加载模型的纯数据契约与测试。推荐 Phase 4B 首先只训练 `text_hidden_fcs`。本轮没有
加载 checkpoint、构建模型、执行 forward/backward/optimizer 或访问 val/test 推理。
审计同时发现冻结 P2 Prompt 与 official assistant target 各含 `[SEG]`，现有
`Sa2VAModel.forward` 会选中两者并在 `check_obj_number` 中截断/重复；在 labels-aware
唯一 assistant token 对齐获得批准和测试前，不具备进入 Phase 4B 的条件。

### Phase 4B-0 实际状态（2026-09-17）

已用本地真实 tokenizer、官方 conversation encoding/图像预处理和 collator 在 smoke1
复现双 `[SEG]`：user/assistant positions=1823/1840，labels=-100/151674。已加入显式
labels-aware token selection 和 strict one-to-one policy；strict 在 MLLM forward 前检查并
绕过 `fix_number=5`，默认 legacy 与推理路径不变。P2 未修改。已创建只训练
`text_hidden_fcs` 的 parse-only 单步配置、正式 dataset/collator bridge、fail-fast/runtime
metadata 和专项测试；没有构建模型、加载权重、执行 forward/backward/optimizer 或
val/test。

### Phase 4B-1 实际状态（2026-09-17）

已固定并验证 `OpenGVLab/InternVL3-2B` revision
`899155015275a9b7338c7f4677e19c784e0e5a21`，并使用官方 `convert_to_pth.py` 将固定
Sa2VA HF revision 转成 4.63 GB、1,589 tensor 的完整 BF16 PTH。物理 GPU 5 上真实构建
模型后，只有四个 `text_hidden_fcs.*` tensor 可训练，共 2,754,304 parameters；仅对
smoke1 执行了一次 forward、backward 和 optimizer.step。三项 loss 与 total loss 均
finite，4/4 projection tensors 获得非零梯度，冻结参数无梯度，step 后参数真实改变；
峰值 allocated/reserved 为 7,211.72/7,570.0 MiB，无 OOM、无重试。projection-only
`iter_1.pth` 仅含四个目标 tensor，并已通过清零后正式 loader 的逐 tensor 精确重载。
Phase 4B-1 门禁通过；本轮未运行 32 样本、val/test 或完整 HF 导出。

### Phase 4C 实际状态（2026-09-17）

固定 synthetic_v1 train-only overfit32、P2 和 seed `20260916`，只训练四个
`text_hidden_fcs.*` tensor（2,754,304 parameters），完成 10 epoch、320 optimizer
steps；每条样本恰好出现 10 次。沿用预注册协议的 AdamW `lr=4e-5`、weight decay
`0.05`、5% linear warmup + cosine，其余模块全部冻结。训练无 OOM、无重试，三份
projection-only checkpoint 位于 `<WORK_DIR>`。

生产推理路径在同一次 HF 模型加载中依次评测 baseline、step32、step128、step320；
baseline 与训练前逐 mask 哈希完全一致。16-group Macro IoU 从 `0.164664` 提升至
`0.359869 / 0.509776 / 0.569468`，step320 相对 baseline `+0.404804`，达到预注册的
明显成功阈值；empty rate 从 `43.75%` 降到 `0%`，16/16 组合均提升。该结论仅证明
固定 32 条训练样本上的 projection-only learnability，不代表泛化能力。进入完整 train
前仍需单独批准训练预算；本阶段没有访问 val/test，也没有自动切换 LoRA 或 SAM2。

### Phase 5A 实际状态（2026-09-17）

strategy A 已在完整 synthetic_v1 train 192 条上完成固定 10 epoch/1920 steps；每条
样本恰好出现 10 次，仍只训练 2,754,304 个 `text_hidden_fcs` 参数。训练无 OOM、无
重试，五份 checkpoint 均为 projection-only。随后同一次 HF 模型加载只评测 64 条
val：zero-shot baseline 精确复现 Phase 3B P2，step192/576/960/1344/1920 的
16-group Macro IoU 分别为 `0.265243/0.385975/0.426042/0.417203/0.406564`。按预注册
Primary 选择 step960，相对 baseline `0.182199` 提升 `+0.243843`；empty rate 从
43.75% 降到 0%。本阶段未访问 test，具备另行批准 Phase 5B 单次 fine-tuned test 的
工程条件。

### Phase 5B 实际状态（2026-09-17）

已将 Phase 5A 唯一选中的 step960 projection 持久化并以 SHA-256
`64c0d109d2985893ba1f2ba4c4fe7acc4265dc758e5d56ecb6ac8e2aa791f41e`
冻结。随后只加载一次正式 HF 模型，对 synthetic_v1 test 64 条各执行一次 P2 推理；
未重新运行 zero-shot、未比较其他 checkpoint、未训练或重试。Fine-tuned 16-group
Macro IoU/Dice 为 `0.428948/0.534238`，相对 Phase 3C 保存的 zero-shot
`0.198033/0.242366` 提升 `+0.230916/+0.291872`；empty rate 从 `42.1875%`
降至 `0%`。15/16 组合提升、1 组持平、0 组下降。64 个预测 mask 均通过原尺寸、
二值和离线指标复算，64/64 predicted-mask 编辑成功。最终无 GT CLI 复用冻结 P2、
strict projection loader 和既有 editor；本阶段没有根据 test 修改模型。

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

### synthetic_v2 / saved-output visualization 实际状态（2026-09-18）

在独立分支 `experiment/synthetic-v2-peft` 完成两项不涉及模型的后续工作：从 Phase 5B
冻结输出离线重建四张定量图、四动作产品图和失败案例图；新增独立 v2 schema 与确定性
1,600 条生成器，train/val/test=`960/320/320`，16 组合各 100 条、action 精确平衡。
联合 mask/泄漏/重复 audit 为 0 hard failure，v1 及历史结果未修改。本阶段未训练、推理或
启动 A/B/C 消融。

### Phase 6A 实际状态（2026-09-17）

已将 ChartGround-Edit 整理为 release candidate：项目 README 聚焦能力、架构、最终
结果、Quick Start、数据/训练复现与局限；根 README 增加 Sa2VA 扩展入口；新增
projection Model Card；无 GT Demo 的参数、四种编辑动作与输出契约均由 mock/unit test
覆盖。版本化结果中的数值字段保持不变，本机用户名、绝对模型路径、固定物理 GPU 编号
和临时目录已替换为公开占位符；历史协议只做路径脱敏，相关发布版 hash metadata 同步，
Phase 5B protocol 未改变。未运行模型、训练、val/test 或新增实验。

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

## Phase 7：synthetic_v2 parameter-efficient adaptation

### Phase 7A 实际状态（2026-09-18）

已完成唯一一次 synthetic_v2 Strategy A 训练：仅更新 2,754,304 个
`text_hidden_fcs` 参数，960 条 train × 5 epoch，共 4,800/4,800 steps；未访问 test。
训练后在固定 320 条 val 上公平比较 zero-shot、v1 step960 迁移和五个 v2 checkpoint，
按预注册规则选择 step4800。其 16-group Macro IoU 为 `0.210365`，相对 zero-shot
提高 `0.112790`，15/16 组提升，empty rate 为 `4.0625%`；相对 v1 step960 迁移仅
提高 `0.031550`，未达到预注册的 `0.05` 门槛。下一阶段可单独预注册小型 LoRA
Strategy B 消融，但不得据本次 val 结果重跑或调整 Strategy A。

### Phase 7B 实际状态（2026-09-18）

已完成唯一一次固定 Strategy B：从原始 Sa2VA full PTH 初始化，同时
训练 2,754,304 个 projection 参数和 LLM 最后 8 层 attention q/k/v/o 的
1,245,184 个 rank-16 LoRA 参数，总可训练 3,999,488（0.1726%）。smoke 和
4,800/4,800 steps 正式训练均通过；960 个 train ID 各出现 5 次，正式
进程无 OOM，未访问 test。五个 B checkpoint 只在固定 320 条 val 上
评测，按预注册规则选择 step4800；其 16-group Macro IoU/Dice 为
`0.273399/0.364193`，相对冻结 Strategy A step4800 提升
`+0.063034/+0.072073`，empty rate 从 `4.0625%` 降至 `1.25%`。三项预注册
简约选择条件全部通过，因此保留 Strategy B。下一主线是持久化并冻结
B step4800，随后只做一次 synthetic_v2 frozen test；不需要先开启 Strategy C。
