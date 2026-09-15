# ChartGround-Edit

ChartGround-Edit 是基于 Sa2VA 的科学图表自然语言指代分割与可控编辑子项目。输入图表和指令，模型定位被指代的曲线、柱体、散点或置信区间等元素并输出 mask；编辑模块再用该 mask 执行 `highlight`、`recolor`、`extract` 或 `remove`。

当前状态：Phase 1A 的数据与可视化闭环、Phase 1B 的 ground-truth mask 可控编辑后端均已实现并通过测试。Phase 2B-0 已按仓库 lock 建立 Sa2VA-InternVL3-2B 环境并校验本地 checkpoint；尚未加载模型或接入 Sa2VA，模型推理、训练和模型评测均未运行。

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

checkpoint 路径不得硬编码；Phase 2B-1 的 ChartGround-Edit CLI 将使用必填
`--checkpoint PATH` 参数。当前环境版本、完整安装记录、离线约定、checkpoint
校验结果和已知依赖 metadata 冲突见
[`docs/phase2_environment.md`](docs/phase2_environment.md)。本阶段尚未运行模型。

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
