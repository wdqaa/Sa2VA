# synthetic_v0 数据划分审计

审计日期：2026-09-16  
数据版本：`synthetic-v0`  
样本总数：32

本审计只读取现有 `annotations.jsonl` 和生成器实现，没有修改 manifest、已有 split、
图像或 mask。Phase 2B-2 的 test baseline 因而保持可复现。

## 1. 每个 split 的样本数

| split | count |
|---|---:|
| train | 24 |
| val | 4 |
| test | 4 |

## 2. split × chart_type

| split | line | bar | scatter | confidence_band | total |
|---|---:|---:|---:|---:|---:|
| train | 6 | 6 | 6 | 6 | 24 |
| val | 1 | 1 | 1 | 1 | 4 |
| test | 1 | 1 | 1 | 1 | 4 |

仅从 chart type 看三个 split 都覆盖四类图表，但这会掩盖 referring type 和 edit
action 的完全偏置。

## 3. split × referring_type

| split | category | appearance | legend | trend | total |
|---|---:|---:|---:|---:|---:|
| train | 0 | 8 | 8 | 8 | 24 |
| val | 4 | 0 | 0 | 0 | 4 |
| test | 4 | 0 | 0 | 0 | 4 |

结论：test 确实只覆盖 category；val 同样只覆盖 category。train 完全没有 category，
而 appearance、legend、trend 完全没有进入 val/test。

## 4. split × edit_action

| split | highlight | recolor | extract | remove | total |
|---|---:|---:|---:|---:|---:|
| train | 4 | 4 | 8 | 8 | 24 |
| val | 4 | 0 | 0 | 0 | 4 |
| test | 0 | 4 | 0 | 0 | 4 |

val 的四条 instruction 全是 highlight，test 的四条全是 recolor。因此当前数据无法
在同一 split 内区分“指代类型影响”和“编辑动作文本影响”。

## 5. split × chart_type × referring_type

每个 chart/referring 组合在全数据中有 2 条，但分配如下：

| chart_type | referring_type | train | val | test | total |
|---|---|---:|---:|---:|---:|
| line | category | 0 | 1 | 1 | 2 |
| line | appearance | 2 | 0 | 0 | 2 |
| line | legend | 2 | 0 | 0 | 2 |
| line | trend | 2 | 0 | 0 | 2 |
| bar | category | 0 | 1 | 1 | 2 |
| bar | appearance | 2 | 0 | 0 | 2 |
| bar | legend | 2 | 0 | 0 | 2 |
| bar | trend | 2 | 0 | 0 | 2 |
| scatter | category | 0 | 1 | 1 | 2 |
| scatter | appearance | 2 | 0 | 0 | 2 |
| scatter | legend | 2 | 0 | 0 | 2 |
| scatter | trend | 2 | 0 | 0 | 2 |
| confidence_band | category | 0 | 1 | 1 | 2 |
| confidence_band | appearance | 2 | 0 | 0 | 2 |
| confidence_band | legend | 2 | 0 | 0 | 2 |
| confidence_band | trend | 2 | 0 | 0 | 2 |

缺失组合数：train 缺 4 个 category 组合；val 缺 12 个非 category 组合；test 也缺
12 个非 category 组合。

## 6. 生成顺序偏置的根因

`chartground_edit/datasets/synthetic.py::generate_dataset()` 的循环顺序固定为：

```text
chart_type → referring_type → repetition(0, 1)
```

同文件 `_split(index)` 再使用 `index % 8`：余数 0 进入 val，余数 1 进入 test，
其余进入 train。每个 chart type 恰好占连续 8 条，因此每个 chart block 都形成同样
的确定性模式：

| block position | referring / repetition | split | edit action |
|---:|---|---|---|
| 0 | category / 0 | val | highlight |
| 1 | category / 1 | test | recolor |
| 2–3 | appearance / 0–1 | train | extract / remove |
| 4–5 | legend / 0–1 | train | highlight / recolor |
| 6–7 | trend / 0–1 | train | extract / remove |

所以偏置并非随机波动，而是生成循环、action 轮换和 split 公式共同造成的结构性偏置。

## 7. 当前 split 的适用边界

- **Prompt 选择：**val 可用于本阶段预注册的 category/highlight 工程诊断，但不适合
  选择面向四类 referring type 的通用 Prompt。4 条结果只能产生候选，不是最终选择。
- **零样本评测：**test 已用于固定 baseline，但只评估 category/recolor 的四个合成
  样本；不能代表 appearance、legend、trend 或真实图表。
- **微调：**24 条 train 可继续作为数据和训练接口的小规模 sanity/过拟合材料，但缺少
  category 且规模极小，不适合产生可泛化模型。
- **泛化结论：**不适合。split 与 referring/action 高度混杂，样本量也不足。

## 8. 对 synthetic_v1 的直接要求

synthetic_v1 必须在每个 `chart_type × referring_type` 组合内部独立划分，而不是按全局
生成位置取模。推荐每组合 20 条，train/val/test 为 12/4/4；16 个组合合计
192/64/64，共 320 条。还需按数据序列或模板族分组后再划分，使用 split 独立 seed，
并检测近重复，避免只换颜色或编辑动作的同源图跨 split。

