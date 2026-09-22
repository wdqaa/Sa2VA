# Phase 2B-2：Sa2VA-InternVL3-2B 零样本诊断基线

状态：**固定协议的 synthetic_v0 test split 已运行**  
运行日期：2026-09-15

## 实验目的与边界

本实验只验证固定 Sa2VA 推理协议在 `synthetic_v0` test split 上的真实表现与批量
工程链路。test split 仅有 4 条合成样本，因此这是诊断性基线，不能视为最终统计
结果，也不能据此声称模型具备图表指代分割泛化能力。

本轮未修改或翻译 annotation instruction，未增加 `target_attributes`，未按图表类型
改变 Prompt，未根据 GT 改阈值或选择 mask，未重试样本，也未进行模型微调。test
结果没有用于 Prompt 选择；后续 Prompt 实验只能在 val split 上进行，最终 Prompt
确定后不能再根据 test 结果修改。

## 固定配置

- 模型：`ByteDance/Sa2VA-InternVL3-2B`
- revision：`15837dcaecc304714a1f0f069e74f47e47521c7f`
- 本地 checkpoint：通过 CLI 的 `--checkpoint` 传入
- 精度：BF16
- 加载：单卡、`use_flash_attn=True`、`local_files_only=True`，无 `device_map`、量化或 CPU offload
- 执行：物理 GPU 2 映射为进程内 `cuda:0`；一个 backend 实例，模型加载一次，按 manifest 顺序串行执行
- 环境：Python 3.11.16、torch 2.6.0+cu124、transformers 4.57.1、flash-attn 2.7.3

固定 Prompt：

```text
<image>Please segment the chart element targeted by this instruction: {instruction}
Please respond with a segmentation mask.
```

## test split 与逐样本结果

| sample_id | chart / referring | instruction | text output | masks / raw shape | pred / GT pixels | intersection | IoU | Dice | 状态 | 编辑 |
|---|---|---|---|---|---:|---:|---:|---:|---|---|
| `cge_line_category_01` | line / category | 请将类别 series_2 对应的曲线改成红色 | `Sure, [SEG].<\|im_end\|>` | 1 / `(1,320,480)` bool | 1,234 / 1,436 | 40 | 0.015209 | 0.029963 | 非空、有少量交集 | 成功 |
| `cge_bar_category_01` | bar / category | 请将类别 B 对应的柱子改成红色 | `Sure, [SEG].<\|im_end\|>` | 1 / `(1,320,480)` bool | 11,738 / 6,960 | 0 | 0 | 0 | nonempty_disjoint | 成功 |
| `cge_scatter_category_01` | scatter / category | 请将类别 group_2 对应的散点序列改成红色 | `Sure, [SEG].<\|im_end\|>` | 1 / `(1,320,480)` bool | 578 / 970 | 175 | 0.127458 | 0.226098 | 非空、有交集 | 成功 |
| `cge_confidence_band_category_01` | confidence_band / category | 请将类别 band_2 对应的置信区间改成红色 | `Sure, it is [SEG].<\|im_end\|>` | 1 / `(1,320,480)` bool | 983 / 11,229 | 0 | 0 | 0 | nonempty_disjoint | 成功 |

四个 raw mask 均已在原图坐标系，后处理只显式移除长度为 1 的前导维度，未发生
resize。主结果固定使用第一个 mask。四条 prediction 都非空，因此现有编辑器都使用
真实 predicted mask 完成 recolor；GT 只用于指标与对比可视化。

## 聚合指标

| 指标 | 真实结果 |
|---|---:|
| sample count | 4 |
| inference success | 4 / 4（100%） |
| empty prediction | 0 / 4（0%） |
| nonempty disjoint | 2 / 4（50%） |
| overlap | 2 / 4（50%） |
| macro mean IoU / median IoU | 0.035667 / 0.007605 |
| macro mean Dice / median Dice | 0.064015 / 0.014981 |
| micro IoU / Dice | 0.006158 / 0.012241 |
| multiple-mask samples | 0 |
| model load time | 9,586.266 ms |
| mean / total `predict_forward` time | 782.740 / 3,130.961 ms |
| PyTorch peak allocated memory | 5,004.330 MiB |

Macro 指标对全部 4 条样本逐条平均，包括 IoU/Dice 为 0 的结果。Micro 指标由全部
样本累计的 215 个交集像素、14,533 个预测前景像素和 20,595 个 GT 前景像素计算。
峰值显存是加载前 reset 后的 `torch.cuda.max_memory_allocated()`，覆盖模型加载和全部
推理，不等同于 `nvidia-smi` 的进程占用。

## 产物与运行命令

完整临时结果位于 `<WORK_DIR>`，包含每样本的
`result.json`、文本、mask、overlay、predicted-mask 编辑图和 comparison，以及根级
`results.jsonl`、`summary.json`、`summary.csv`、`gallery.png`。版本化总览：

![Sa2VA-InternVL3-2B synthetic_v0 test zero-shot results](../assets/sa2va_2b_zeroshot_test.png)

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
projects/sa2va/.venv/bin/python \
  projects/chartground_edit/scripts/run_sa2va_split.py \
  --checkpoint <MODEL_ROOT>/Sa2VA-InternVL3-2B \
  --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl \
  --split test \
  --device cuda:0 \
  --dtype bfloat16 \
  --expected-count 4 \
  --continue-on-sample-error \
  --output-dir <WORK_DIR>
```

第一次从受限执行沙箱启动时，GPU 设备不可见，backend 在 CUDA 门槛处产生
`model_load_failed`，没有加载权重或调用任何样本的 `predict_forward`。随后在允许
GPU 访问的上下文中用相同固定参数完成上述唯一正式运行；这不是对模型输出的挑选或
重试。

## 解释、风险与下一步

4/4 inference success、mask/编辑/文件闭环完整，说明批量工程链路工作正常；低 IoU
说明零样本语义定位质量差，并不等于工程链路失败。当前有明显的“能生成 `[SEG]`
和非空 mask，但选错同图其他序列/元素”的模式：bar 与 confidence band 完全不相交，
line 也只在曲线交叉处产生少量交集。scatter 是四条中重叠最高的一条，但 IoU 仍只有
0.127458。

下一步应先在 val split 做预先声明的 Prompt 诊断，而不是直接根据这 4 条 test 结果
改 Prompt。考虑到样本过少且四条 test 全是 category referring，Prompt 诊断之后还应
扩充覆盖真实图表和更丰富 referring type 的数据；在此之前不宜直接进入微调，也不应
把当前 test 作为 Prompt 搜索集。
