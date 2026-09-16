# ChartGround-Edit

ChartGround-Edit 是基于 Sa2VA 的科学图表自然语言指代分割与可控编辑子项目。输入图表和指令，模型定位被指代的曲线、柱体、散点或置信区间等元素并输出 mask；编辑模块再用该 mask 执行 `highlight`、`recolor`、`extract` 或 `remove`。

当前状态：Phase 1A 数据闭环、Phase 1B ground-truth mask 编辑后端和 Phase 2B-1 单样本 Sa2VA 推理闭环均已实现。Sa2VA-InternVL3-2B 已真实运行 1 个固定 smoke 样本；完整 test split、训练和微调均未运行。

## MVP

- 图表：折线图、柱状图、散点图、带置信区间的科学曲线图
- 指代：类别、外观、图例、趋势
- 编辑：`highlight`、`recolor`、`extract`、`remove`
- 暂不支持：字符级 OCR 分割、数学公式理解、复杂三维图表和开放域编辑

## 当前目录结构

```text
projects/chartground_edit/
├── README.md
├── assets/                 # README 使用的版本化展示资产
├── docs/                   # 标注、JSONL、数据卡与编辑语义规范
├── data/                   # 本地生成的 synthetic_v0（生成物被 gitignore）
├── scripts/                # 数据、校验、编辑 CLI 与 gallery 入口
├── tests/                  # 数据、几何、编辑器与 CLI 测试
└── chartground_edit/       # Python 包
    ├── datasets/           # schema、reader、合成生成器、同步变换
    ├── editing/            # GT/predicted mask 均可复用的确定性编辑后端
    ├── inference/          # Sa2VA backend、结果类型、mask 后处理与指标
    └── visualization/      # 原图/mask/overlay/目标裁剪及 contact sheet
```

## Phase 1A 快速使用

从仓库根目录运行；命令只生成本地小图，不下载数据或模型：

```bash
PYTHONPATH=projects/chartground_edit python \
  projects/chartground_edit/scripts/generate_synthetic_v0.py \
  --output-dir projects/chartground_edit/data/synthetic_v0 \
  --seed 20260915 --clean
```

生成后可打开：

- `projects/chartground_edit/data/synthetic_v0/annotations.jsonl`
- `projects/chartground_edit/data/synthetic_v0/gallery.png`
- `projects/chartground_edit/data/synthetic_v0/visualizations/*.png`

运行测试：

```bash
python -m pytest projects/chartground_edit/tests -q
```

可单独验证已有 manifest：

```bash
PYTHONPATH=projects/chartground_edit python \
  projects/chartground_edit/scripts/validate_jsonl_v0.py \
  projects/chartground_edit/data/synthetic_v0/annotations.jsonl \
  --expected-count 32
```

协议详见 `docs/annotation_spec_v0.md` 与 `docs/jsonl_schema_v0.md`，数据限制详见 `docs/data_card_synthetic_v0.md`。

## Phase 1B：使用 mask 编辑

编辑接口只接收 RGB Pillow 图像、二值 mask、动作名和参数，不依赖 Sa2VA 或 annotation 数据结构。完整契约见 `docs/editing_spec_v0.md`。

```python
from chartground_edit.editing import edit

edited = edit(image, mask, "recolor", {"color": "#E63946"})
```

CLI 可从仓库根目录直接运行：

```bash
python projects/chartground_edit/scripts/edit_with_mask.py \
  --image projects/chartground_edit/data/synthetic_v0/images/cge_bar_category_01.png \
  --mask projects/chartground_edit/data/synthetic_v0/masks/cge_bar_category_01.png \
  --action recolor \
  --output /tmp/chartground_edit_recolor.png \
  --color "#E63946"
```

CLI 同时提供 `--highlight-strength` 和 `--remove-fill-mode {color,neighbor}`；运行 `--help` 可查看完整参数。四种动作的 v0 语义为：

- `highlight`：mask 内保持原样，mask 外按强度变暗并降低饱和度；这是唯一预期修改 mask 外的动作。
- `recolor`：仅修改 mask 内色相/饱和度，并保留原像素 HSL lightness。
- `extract`：返回 RGBA，mask 外 alpha 为 0，mask 内保留原 RGB。
- `remove`：仅在 mask 内使用固定色或确定性邻域中位数填充，不进行内容恢复或生成式修复。

![Phase 1B ground-truth mask editing gallery](assets/editing_v0_gallery.png)

gallery 可复现生成：

```bash
python projects/chartground_edit/scripts/generate_editing_gallery.py \
  --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl \
  --output projects/chartground_edit/assets/editing_v0_gallery.png
```

## Phase 2 环境准备

Sa2VA-InternVL3-2B 使用仓库 `latest` 依赖组。环境入口由官方脚本创建在
`projects/sa2va/.venv`，物理目录位于 `/tmp/sa2va_env`：

```bash
bash setup_env.sh sa2va latest
source projects/sa2va/.venv/bin/activate
```

checkpoint 路径不得硬编码；ChartGround-Edit CLI 使用必填
`--checkpoint PATH` 参数。当前环境版本、完整安装记录、离线约定、checkpoint
校验结果和已知依赖 metadata 冲突见
[`docs/phase2_environment.md`](docs/phase2_environment.md)。

## Phase 2B-1：单样本 Sa2VA baseline

CLI 从 JSONL 读取原图、GT 和未经补充的原始 instruction，固定构造一个 Prompt，
再将模型的第一个预测 mask 交给指标与现有编辑器。`--skip-model-run` 可在无 GPU
时验证输入和 Prompt，且不会加载模型：

```bash
projects/sa2va/.venv/bin/python \
  projects/chartground_edit/scripts/run_sa2va_baseline.py \
  --checkpoint /path/to/Sa2VA-InternVL3-2B \
  --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl \
  --sample-id cge_bar_category_01 \
  --device cuda:0 --dtype bfloat16 \
  --edit-action recolor --edit-color "#E63946" \
  --output-dir /tmp/chartground_edit_phase2b1_smoke \
  --skip-model-run
```

2026-09-15 的唯一真实 smoke 使用 checkpoint revision
`15837dcaecc304714a1f0f069e74f47e47521c7f`。模型生成
`Sure, [SEG].<|im_end|>` 并返回一个非空 bool mask，但预测了错误柱体，实际
IoU/Dice 均为 0.0。编辑链路成功不代表分割正确；尚未运行完整 test split。

![Sa2VA-InternVL3-2B single-sample smoke result](assets/sa2va_2b_smoke.png)

## Phase 2B-2：固定 test split 零样本基线

批量入口在一个进程中创建一个 backend，并按 manifest 顺序处理 split；模型只加载
一次，单样本失败不会触发重试。以下命令固定选择 4 条 test 样本，仍使用 Phase
2B-1 的唯一 Prompt 和第一个预测 mask：

```bash
CUDA_VISIBLE_DEVICES=2 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
projects/sa2va/.venv/bin/python \
  projects/chartground_edit/scripts/run_sa2va_split.py \
  --checkpoint /path/to/Sa2VA-InternVL3-2B \
  --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl \
  --split test --device cuda:0 --dtype bfloat16 \
  --expected-count 4 --continue-on-sample-error \
  --output-dir /tmp/chartground_edit_phase2b2_test
```

2026-09-15 的真实运行中，4/4 inference success、0/4 空预测、2/4 nonempty
disjoint；macro mean IoU/Dice 为 0.035667/0.064015，micro IoU/Dice 为
0.006158/0.012241。模型只加载一次。所有编辑图均由 predicted mask 驱动；错误预测
也完整保留在下图中。

![Sa2VA-InternVL3-2B synthetic_v0 test zero-shot results](assets/sa2va_2b_zeroshot_test.png)

这只是 4 条合成样本的诊断性 baseline，不是最终统计结果。test split 未用于 Prompt
选择；后续 Prompt 诊断只能在 val split 上进行。逐样本输出、文本、耗时、显存和
结果解释见 [`docs/phase2_zeroshot_results.md`](docs/phase2_zeroshot_results.md)。

## Phase 2C：val-only Prompt 诊断

split 审计确认 synthetic_v0 存在生成顺序偏置：train 只有 appearance/legend/trend，
val 和 test 都只有 category；val 全是 highlight，test 全是 recolor。具体交叉表与适用
边界见 [`docs/synthetic_v0_split_audit.md`](docs/synthetic_v0_split_audit.md)。原 manifest
没有修改。

Phase 2C 将 synthetic-v0 instruction 确定性拆成 `referring_expression`、
`edit_action` 和显式 `edit_parameters`，并只在 val 上比较三个预注册 Prompt：

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
projects/sa2va/.venv/bin/python \
  projects/chartground_edit/scripts/run_prompt_diagnostic.py \
  --checkpoint /path/to/Sa2VA-InternVL3-2B \
  --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl \
  --split val \
  --prompt-variants full_instruction,target_only_en,target_only_zh \
  --device cuda:0 --dtype bfloat16 --expected-count 4 \
  --output-dir /tmp/chartground_edit_phase2c_prompt_diagnostic
```

脚本明确拒绝 `--split test`。真实 4×3 运行中三个 Prompt 都达到 100% inference
success 和 100% `[SEG]` output rate，含义只是全部样本成功产生协议合法输出，**不代表
100% segmentation accuracy**。P0/P1/P2 macro IoU 分别为
0.392236/0.340420/0.351172；删除编辑动作没有显示一致改善。中文 wrapper 保持
`[SEG]` 输出，但改变了部分 mask 几何。

![Phase 2C val-only Prompt diagnostic](assets/phase2c_prompt_diagnostic.png)

本诊断只有 4 条 category/highlight 合成 val 样本；P0 最多只能作为 balanced
synthetic_v1 val 的候选。test 零样本 macro IoU 仍只有 0.035667，且本轮没有重跑或
用于调 Prompt。完整配对结果、限制和 synthetic_v1 规范见
[`docs/phase2_prompt_diagnostic.md`](docs/phase2_prompt_diagnostic.md)。

## 开发路线

1. Phase 0：完成 Sa2VA 源码地图、环境边界和工程骨架。
2. Phase 1：冻结最小数据格式，用 32 个样本验证 mask 和可视化。
3. Phase 2：不训练地跑通 Sa2VA 单图推理基线。
4. Phase 3：建设可复现的科学图表指代分割数据集。
5. Phase 4：先通过 32 样本过拟合，再申请完整 LoRA/轻量训练。
6. Phase 5：实现并独立测试四类 mask 驱动编辑。
7. Phase 6：完成分层评测、Demo、文档和开源发布检查。

详细范围、源码映射和验收条件见仓库根目录的 `PROJECT.md` 与 `PLAN.md`；实验必须记录在 `EXPERIMENTS.md`。

## 开发约束

- 默认不修改 `projects/sa2va/` 的核心模型逻辑，优先通过适配器复用。
- 图像与 mask 的随机几何变换必须共享同一组参数。
- 未经确认不下载权重或大型数据，不启动完整训练。
- 所有结果必须来自可复现脚本，失败和未运行状态要明确记录。
