# Phase 2C：val-only Prompt 诊断

状态：**已完成固定的 4 样本 × 3 Prompt val 诊断**  
运行日期：2026-09-16

## 1. 目的与实验边界

本实验检查完整编辑指令中的动作/参数文本是否干扰目标定位。只使用 synthetic_v0 的
4 条 val 样本；Phase 2B-2 test split 没有重新运行，也没有用于本轮 Prompt 选择。
三个 Prompt 在运行前固定，每个 sample/Prompt 组合只推理一次，同一个模型实例完成
全部 12 次调用。

val 仅包含 category referring，且 edit action 全是 highlight。本实验不能回答其他
referring type 或 recolor/extract/remove 的动作文本是否产生干扰，也不能作为最终泛化
评测。样本数只有 4，因此不进行显著性检验。

## 2. instruction 拆分

synthetic_v0 的生成模板为：

```text
请将{referring_expression}{action_suffix}
```

实现位于 `inference/instruction_processing.py`。提取器仅使用 `chart_type`、
`referring_type` 和 `edit_action` 选择必须在原 instruction 中逐字出现的模板片段：

1. 验证精确的 `请将` 前缀和 action suffix；
2. 将中间连续子串作为 `referring_expression`；
3. 验证相应 chart noun 和 referring frame；
4. 任一结构不匹配即明确失败，不猜测。

`target_attributes` 完全不参与提取。edit parameters 也只从明确的动作后缀规范化：
recolor 的“红色”映射为 `color=red`，extract 的“透明背景”映射为
`background=transparent`，remove 的“背景色”映射为 `fill=background_color`；本轮
val 全是 highlight，因此 edit parameters 均为空。

val 的实际拆分为：

| sample_id | full instruction | referring expression | action |
|---|---|---|---|
| `cge_line_category_00` | 请将类别 series_1 对应的曲线高亮 | 类别 series_1 对应的曲线 | highlight |
| `cge_bar_category_00` | 请将类别 D 对应的柱子高亮 | 类别 D 对应的柱子 | highlight |
| `cge_scatter_category_00` | 请将类别 group_1 对应的散点序列高亮 | 类别 group_1 对应的散点序列 | highlight |
| `cge_confidence_band_category_00` | 请将类别 band_1 对应的置信区间高亮 | 类别 band_1 对应的置信区间 | highlight |

## 3. 三个预注册 Prompt

P0 `full_instruction`：

```text
<image>Please segment the chart element targeted by this instruction: {instruction}
Please respond with a segmentation mask.
```

P1 `target_only_en`：

```text
<image>Please segment the chart element described by this referring expression: {referring_expression}
Please respond with a segmentation mask.
```

P2 `target_only_zh`：

```text
<image>请分割图中由以下指代表达式指定的图表元素：{referring_expression}
请使用 [SEG] 标记返回分割掩码。
```

中文 referring expression 没有翻译或补充。所有 variant 都固定使用第一个模型 mask，
没有根据 GT 选择 mask 或调整阈值。

## 4. 每样本配对结果

表中括号内为 predicted foreground pixels。

| sample | P0 IoU / Dice (pixels) | P1 IoU / Dice (pixels) | P2 IoU / Dice (pixels) |
|---|---:|---:|---:|
| line category | 0 / 0 (83) | 0 / 0 (17) | 0 / 0 (30) |
| bar category | 0.943775 / 0.971074 (8,704) | 0.935220 / 0.966526 (8,786) | 0.532129 / 0.694627 (15,624) |
| scatter category | 0.008073 / 0.016016 (29) | 0.001029 / 0.002055 (3) | 0.070024 / 0.130882 (390) |
| confidence-band category | 0.617095 / 0.763214 (9,245) | 0.425432 / 0.596917 (6,372) | 0.802535 / 0.890452 (12,049) |

三个 Prompt 在 line 上都得到非空但完全不相交的预测。P1 没有使任何样本从 disjoint
转为 overlap，也没有在任一样本上提高 IoU。P2 提高了 scatter 和 confidence band，
但 bar 出现明显过分割并显著下降。所有 variant 都保持 3/4 overlap、1/4 nonempty
disjoint；没有 empty 或 no-SEG 转换。

12 条文本输出均为 `Sure, [SEG].<|im_end|>`。

## 5. 聚合指标

| metric | P0 full_instruction | P1 target_only_en | P2 target_only_zh |
|---|---:|---:|---:|
| samples | 4 | 4 | 4 |
| inference success rate | 100% | 100% | 100% |
| SEG output rate | 100% | 100% | 100% |
| empty prediction rate | 0% | 0% | 0% |
| nonempty-disjoint rate | 25% | 25% | 25% |
| overlap rate | 75% | 75% | 75% |
| macro mean IoU | 0.392236 | 0.340420 | 0.351172 |
| median IoU | 0.312584 | 0.213231 | 0.301076 |
| macro mean Dice | 0.437576 | 0.391375 | 0.428990 |
| median Dice | 0.389615 | 0.299486 | 0.412755 |
| micro IoU | 0.666654 | 0.558417 | 0.609943 |
| micro Dice | 0.799991 | 0.716647 | 0.757720 |
| mean inference time (ms) | 757.661 | 569.087 | 546.161 |

模型加载一次，耗时 9,460.079 ms；整个运行的 PyTorch peak allocated memory 为
5,004.661 MiB。P0 首条推理包含明显的首次生成 warm-up，因此不能把三列平均耗时
差异直接解释为 Prompt 语言带来的稳定速度优势。

## 6. 可视化与完整产物

![Phase 2C val-only Prompt diagnostic](../assets/phase2c_prompt_diagnostic.png)

完整临时结果位于 `<WORK_DIR>`，包含 12 条
`result.json`/mask/Prompt/文本，以及 `results.jsonl`、`summary.json`、
`summary.csv`、`paired_results.json`、`paired_results.csv` 和 `gallery.png`。

## 7. 关于编辑动作干扰的结论

当前结果**不支持**“删除编辑动作会稳定改善定位”：P1 的 macro/micro IoU 和 Dice
都低于 P0，且四个样本都没有提高 IoU。P2 的变化是混合的，说明 wrapper 语言和目标
短语裁剪会改变 mask 几何，但不是一致改善。中文 wrapper 没有损害协议输出：4/4
均产生 `[SEG]` 和非空 mask；这只表示协议成功，不表示 segmentation accuracy 为
100%。

按这 4 条 val 的聚合结果，P0 `full_instruction` 只能被保留为 synthetic_v1 balanced
val 的**候选方案**，不能称为已确定的最佳 Prompt。P1/P2 的 target-only 设计仍可作为
预注册对照，但不应根据本轮单样本现象继续改模板。

## 8. 不能得出的结论

- 不能声称 P0 对真实图表或四种 referring type 普遍更优。
- 不能声称 highlight 动作一定有益或其他编辑动作不会干扰。
- 不能把 100% inference/SEG success 表述为 100% 分割准确率。
- 不能根据本轮 val 结果修改已经冻结的 test Prompt 或重新运行 test。
- 不能用 4 条结果进行统计显著性检验或决定微调配置。

## 9. synthetic_v1 推荐规范

- 4 chart types × 4 referring types × 每组合 20 条，共 320 条；
- 每组合独立划分 train/val/test = 12/4/4，总计 192/64/64；
- schema 显式保存 `full_instruction`、`referring_expression`、`edit_action`、
  `edit_parameters`、`target_type`、`target_attributes`、`generator_seed`、
  `style_family`、`difficulty`、`distractor_count`；
- 每个 split 覆盖全部 16 个组合，category/legend/appearance/trend 平衡；
- split 使用不同 seed，并按底层数据序列、模板族或 style family 分组后划分，防止
  近重复跨 split；
- 编辑颜色独立采样，不能总由目标当前颜色决定；
- 扰动 series 数量、颜色、marker、线宽、网格、图例位置和 distractor 数量；
- 将细线、小 marker、相近颜色、交叉/遮挡和 confidence band 重叠纳入 difficulty；
- test 永不用于 Prompt 调优，Prompt 只在 balanced val 上选择。

本轮只制定此规范，没有生成 synthetic_v1。
