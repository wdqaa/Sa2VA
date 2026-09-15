# ChartGround-Edit

ChartGround-Edit 是基于 Sa2VA 的科学图表自然语言指代分割与可控编辑子项目。输入图表和指令，模型定位被指代的曲线、柱体、散点或置信区间等元素并输出 mask；编辑模块再用该 mask 执行 `highlight`、`recolor`、`extract` 或 `remove`。

当前状态：仅完成仓库理解、项目规划和工程骨架初始化。数据读取、模型适配、训练、编辑器、评测和 Demo 均尚未实现。

## MVP

- 图表：折线图、柱状图、散点图、带置信区间的科学曲线图
- 指代：类别、外观、图例、趋势
- 编辑：`highlight`、`recolor`、`extract`、`remove`
- 暂不支持：字符级 OCR 分割、数学公式理解、复杂三维图表和开放域编辑

## 预期目录结构

```text
projects/chartground_edit/
├── README.md               # 子项目入口、用法与状态
├── configs/                # 数据、推理、微调和评测配置
├── data/                   # schema、轻量 manifest 与数据卡；不提交大型数据
├── scripts/                # 数据检查、可视化、推理、评测和 Demo 启动脚本
├── tests/                  # schema、几何一致性、编辑器和 smoke tests
└── chartground_edit/       # Python 包
    ├── datasets/           # 后续：schema、reader、同步变换
    ├── models/             # 后续：Sa2VA 薄适配层
    ├── editing/            # 后续：四类确定性编辑操作
    ├── evaluation/         # 后续：分割与编辑指标
    └── visualization/      # 后续：overlay 与 before/mask/after
```

目录中的模块名表示规划，不代表功能已完成；本轮只创建顶层空目录。

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
