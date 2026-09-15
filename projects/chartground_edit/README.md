# ChartGround-Edit

ChartGround-Edit 是基于 Sa2VA 的科学图表自然语言指代分割与可控编辑子项目。输入图表和指令，模型定位被指代的曲线、柱体、散点或置信区间等元素并输出 mask；编辑模块再用该 mask 执行 `highlight`、`recolor`、`extract` 或 `remove`。

当前状态：Phase 1A 的 v0 标注协议、JSONL 校验器、最小 Reader、32 条确定性合成数据、同步几何变换和可视化闭环已实现并通过测试。尚未接入 Sa2VA；模型推理、训练、编辑器和模型评测均未运行。

## MVP

- 图表：折线图、柱状图、散点图、带置信区间的科学曲线图
- 指代：类别、外观、图例、趋势
- 编辑：`highlight`、`recolor`、`extract`、`remove`
- 暂不支持：字符级 OCR 分割、数学公式理解、复杂三维图表和开放域编辑

## 当前目录结构

```text
projects/chartground_edit/
├── README.md
├── docs/                   # v0 标注规范与 JSONL schema
├── data/                   # 数据卡与本地生成的 synthetic_v0（生成物被 gitignore）
├── scripts/                # Phase 1A 数据生成入口
├── tests/                  # schema、Reader、确定性、几何与可视化测试
└── chartground_edit/       # Python 包
    ├── datasets/           # schema、reader、合成生成器、同步变换
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
