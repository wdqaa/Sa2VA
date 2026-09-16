# Phase 3B balanced synthetic_v1 val Prompt benchmark

运行日期：2026-09-16  
数据范围：仅 `synthetic_v1` val，64 条  
正式调用：64 × 3 = 192 次，全部完成  
正式选择：P2 `target_only_zh`，**selected on validation**

## 实验身份与完整性

- checkpoint：`ByteDance/Sa2VA-InternVL3-2B`
- revision：`15837dcaecc304714a1f0f069e74f47e47521c7f`
- dtype：BF16；单卡；FlashAttention；无量化、offload 或 device map
- 物理 GPU 0 → 进程内 `cuda:0`，NVIDIA GeForce RTX 3090
- manifest SHA-256：`ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`
- protocol SHA-256：`5de44dcf8e21845aaf3d8c5355faedd62b80899cc54b698e7fe2731aa839ef1d`
- Prompt registry SHA-256：`dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0`
- protocol 前后 hash 相同；模型加载 1 次；192/192 backend 调用完成；没有重试
- 落盘 192 个二值原尺寸 mask；逐个独立重算像素计数、IoU 和 Dice，0 个不一致
- synthetic_v1 train/test 没有执行模型推理

正式命令：

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_prompt_benchmark_v1.py --checkpoint /home/dqwang/Model/Sa2VA-InternVL3-2B --manifest projects/chartground_edit/data/synthetic_v1/annotations.jsonl --split val --device cuda:0 --dtype bfloat16 --expected-samples 64 --output-dir /tmp/chartground_edit_phase3b_balanced_val
```

## Prompt 聚合指标

Phase 3B 的结果 schema 将执行、mask 协议和预测内容分开记录：

- `execution_success`：`predict_forward` 正常完成且没有运行时异常。
- `mask_contract_valid`：输出可按冻结规则归一化为原图尺寸二值 mask；全零 mask 仍然合法。
- `segmentation_token_present`：生成文本包含 `[SEG]`。
- `nonempty_prediction` / `empty_prediction`：归一化 mask 是否含前景，二者严格互补。
- `overlapping_prediction`：预测 mask 与 GT 存在至少一个交集像素。
- `inference_success`：保留原始值以兼容已有消费者，但已标记 deprecated；在本次结果中其
  历史含义等价于 nonempty prediction，不能解释成程序执行成功。

| metric | P0 full instruction | P1 target-only EN | P2 target-only ZH |
|---|---:|---:|---:|
| attempts | 64 | 64 | 64 |
| execution success | 64/64 | 64/64 | 64/64 |
| mask contract valid | 64/64 | 64/64 | 64/64 |
| `[SEG]` output rate | 100% | 100% | 100% |
| nonempty prediction | 35/64 | 36/64 | 36/64 |
| empty prediction | 29/64 | 28/64 | 28/64 |
| nonempty-disjoint rate | 12.5000% | 15.6250% | 15.6250% |
| overlapping prediction | 27/64 | 26/64 | 26/64 |
| 16-group Macro IoU | 0.164503 | 0.179093 | **0.182199** |
| 16-group Macro Dice | 0.192781 | 0.210226 | **0.216221** |
| sample Macro IoU | 0.164503 | 0.179093 | 0.182199 |
| sample Macro Dice | 0.192781 | 0.210226 | 0.216221 |
| Micro IoU | 0.150734 | **0.202185** | 0.192302 |
| Micro Dice | 0.261979 | **0.336362** | 0.322572 |
| median IoU / Dice | 0 / 0 | 0 / 0 | 0 / 0 |
| worst-sample IoU | 0 | 0 | 0 |
| mean latency (ms) | 578.854 | 575.031 | 568.848 |
| median latency (ms) | 564.549 | 563.761 | 563.906 |

所有 192 条文本都含 `[SEG]`，且上游每次都返回一个 `(1,320,480)` bool mask；192 个
mask 均完成合法归一化。其中 85 个 mask 是合法的空预测（P0/P1/P2 为 29/28/28），
不是程序执行失败或 mask contract 失败。legacy `inference_success` 仍保留原始 false 值，
但仅为兼容字段。100% `[SEG]` 只表示文本协议成功，不是 100% segmentation accuracy。
三种文本共有：184 条 `Sure, [SEG].<|im_end|>`、5 条
`Sure, it is [SEG].<|im_end|>`、3 条较长的等价回答。

## 16 个 chart/referring group

下表每组 `n=4`。单元格为 `mean IoU / mean Dice / empty rate / nonempty-disjoint rate`。

| group | n | P0 I/D/E/X | P1 I/D/E/X | P2 I/D/E/X |
|---|---:|---:|---:|---:|
| bar/appearance | 4 | 0.0000/0.0000/1.00/0.00 | 0.0120/0.0228/0.50/0.25 | 0.0003/0.0006/0.50/0.25 |
| bar/category | 4 | 0.9410/0.9696/0.00/0.00 | 0.9520/0.9753/0.00/0.00 | 0.9468/0.9725/0.00/0.00 |
| bar/legend | 4 | 0.6593/0.6993/0.00/0.25 | 0.7140/0.7315/0.00/0.25 | 0.7189/0.7341/0.00/0.25 |
| bar/trend | 4 | 0.0000/0.0000/0.50/0.50 | 0.0003/0.0006/0.25/0.50 | 0.0000/0.0000/0.50/0.50 |
| confidence_band/appearance | 4 | 0.0000/0.0000/1.00/0.00 | 0.0000/0.0000/0.75/0.25 | 0.0000/0.0000/1.00/0.00 |
| confidence_band/category | 4 | 0.0168/0.0315/0.75/0.00 | 0.0610/0.0981/0.75/0.00 | 0.0874/0.1295/0.75/0.00 |
| confidence_band/legend | 4 | 0.2220/0.2930/0.25/0.25 | 0.2816/0.3545/0.25/0.25 | 0.2720/0.3477/0.25/0.25 |
| confidence_band/trend | 4 | 0.1939/0.2208/0.00/0.50 | 0.1943/0.2186/0.50/0.25 | 0.1629/0.1972/0.25/0.50 |
| line/appearance | 4 | 0.0000/0.0000/1.00/0.00 | 0.0000/0.0000/0.75/0.25 | 0.0000/0.0000/1.00/0.00 |
| line/category | 4 | 0.0000/0.0000/0.75/0.25 | 0.0000/0.0000/1.00/0.00 | 0.0000/0.0000/1.00/0.00 |
| line/legend | 4 | 0.2484/0.3185/0.00/0.00 | 0.2500/0.3223/0.00/0.00 | 0.2527/0.3266/0.00/0.00 |
| line/trend | 4 | 0.0425/0.0769/0.00/0.00 | 0.1070/0.1842/0.00/0.00 | 0.1831/0.2862/0.00/0.25 |
| scatter/appearance | 4 | 0.0000/0.0000/1.00/0.00 | 0.0000/0.0000/1.00/0.00 | 0.0000/0.0000/1.00/0.00 |
| scatter/category | 4 | 0.1237/0.1846/0.50/0.00 | 0.1261/0.1902/0.50/0.00 | 0.1306/0.2011/0.50/0.00 |
| scatter/legend | 4 | 0.0966/0.1585/0.00/0.25 | 0.0885/0.1457/0.00/0.50 | 0.0949/0.1557/0.00/0.50 |
| scatter/trend | 4 | 0.0879/0.1318/0.50/0.00 | 0.0788/0.1198/0.75/0.00 | 0.0657/0.1083/0.25/0.00 |

P2 的最弱组按确定性排序为 `bar/trend`，mean IoU=0；同样为 0 的还有
confidence-band/appearance、line/appearance、line/category 和 scatter/appearance。
主要系统性失败是 appearance 指代与细线/趋势目标产生空 mask 或错误目标。bar/category
和 bar/legend 明显较强，但不能用这些大目标的 micro 指标掩盖其他组失败。

## Action 分组

| action | Prompt | n | mean IoU | mean Dice | empty | disjoint |
|---|---|---:|---:|---:|---:|---:|
| extract | P0 | 16 | 0.081922 | 0.103463 | 0.562 | 0.062 |
| extract | P1 | 16 | 0.081887 | 0.103291 | 0.562 | 0.062 |
| extract | P2 | 16 | 0.085494 | 0.111008 | 0.562 | 0.062 |
| highlight | P0 | 16 | 0.223254 | 0.272665 | 0.250 | 0.188 |
| highlight | P1 | 16 | 0.242948 | 0.292564 | 0.188 | 0.250 |
| highlight | P2 | 16 | 0.236534 | 0.288306 | 0.312 | 0.188 |
| recolor | P0 | 16 | 0.145412 | 0.168869 | 0.500 | 0.125 |
| recolor | P1 | 16 | 0.167000 | 0.194674 | 0.438 | 0.188 |
| recolor | P2 | 16 | 0.178351 | 0.209167 | 0.375 | 0.188 |
| remove | P0 | 16 | 0.207422 | 0.226128 | 0.500 | 0.125 |
| remove | P1 | 16 | 0.224536 | 0.250374 | 0.562 | 0.125 |
| remove | P2 | 16 | 0.228416 | 0.256403 | 0.500 | 0.188 |

相对 P1，P0 在 highlight/recolor/remove 的 mean IoU 分别低 0.01969/0.02159/0.01711，
extract 基本相同；相对 P2，P0 四个 action 都更低。完整编辑指令存在干扰的迹象，但
P1/P2 的 empty/disjoint 变化并不一致，因此不能解释为确定的动作因果效应。

## Difficulty 与 distractor

| difficulty / distractors | Prompt | n | mean IoU | mean Dice | empty | disjoint |
|---|---|---:|---:|---:|---:|---:|
| easy / 2 | P0 | 21 | 0.237849 | 0.287290 | 0.286 | 0.143 |
| easy / 2 | P1 | 21 | 0.255055 | 0.303547 | 0.238 | 0.238 |
| easy / 2 | P2 | 21 | 0.248305 | 0.299005 | 0.333 | 0.190 |
| medium / 3 | P0 | 22 | 0.149344 | 0.167365 | 0.500 | 0.136 |
| medium / 3 | P1 | 22 | 0.175390 | 0.202719 | 0.500 | 0.136 |
| medium / 3 | P2 | 22 | 0.188191 | 0.218854 | 0.409 | 0.182 |
| hard / 4 | P0 | 21 | 0.107037 | 0.124900 | 0.571 | 0.095 |
| hard / 4 | P1 | 21 | 0.107009 | 0.124769 | 0.571 | 0.095 |
| hard / 4 | P2 | 21 | 0.109815 | 0.130678 | 0.571 | 0.095 |

synthetic_v1 中 difficulty 与 distractor count 一一对应，二者结果相同，不能从本实验
分离“视觉难度”和“实体数量”的独立影响。三个 Prompt 都随难度上升而明显下降。

## Paired 结果与 bootstrap

- unique IoU wins：P0=10、P1=7、P2=14；另有 33 个 IoU tie。
- 三种预测 mask 完全相同：22/64 样本。
- P2 与第二名 P1 的 16-group Macro IoU 差值：`+0.003106`。
- P2 − P0：`+0.017696`，paired bootstrap 95% CI
  `[-0.000062, 0.039690]`。
- P2 − P1：`+0.003106`，paired bootstrap 95% CI
  `[-0.006147, 0.015220]`。

两组 CI 都包含 0。P2 因 primary metric 最高而按预注册规则被选择，但没有明确优势，
只能称为 **selected on validation / operational selection**，不能称为显著优于 P0/P1。
按 group 分别挑 Prompt 的 oracle 分布为 P0/P1/P2=6/5/5，这只说明组间异质性，不能
用于正式的按类型 Prompt 路由。

P2 相对 P0：line mean IoU `+0.036208`、confidence-band `+0.022397`，scatter
`-0.004251`。这不改变全局选择规则。

## Worst samples 与失败模式

P2 的 worst 10 均为 IoU=0，包括：

- `cgev1_bar_appearance_89956671cc`（empty）
- `cgev1_bar_appearance_9a05d14b88`（empty）
- `cgev1_bar_appearance_ab3a42ae49`（nonempty disjoint）
- `cgev1_bar_legend_bbdf5fdb7c`（nonempty disjoint）
- `cgev1_bar_trend_58fe84fcad`（nonempty disjoint）
- `cgev1_bar_trend_5948ec9852`（empty）
- `cgev1_bar_trend_7512027fe0`（empty）
- `cgev1_bar_trend_f5c9ef2c57`（nonempty disjoint）
- `cgev1_confidence_band_appearance_3f7d1698f8`（empty）
- `cgev1_confidence_band_appearance_4f32c82a47`（empty）

总体上，模型会生成合法 `[SEG]` 文本和 shape 正确的 mask，但大量 mask 全空，或非空却
选择错误图元。工程协议成功和 segmentation quality 必须分开解释。

## 可视化与产物

gallery 每组固定选择 sample ID 字典序最小者，与模型效果无关；已实际打开检查，16 个
组合完整，GT 与 P0/P1/P2 真实预测均显示，空预测和 IoU=0 未隐藏。

![Phase 3B balanced val Prompt benchmark](../assets/phase3b_balanced_val_prompt_benchmark.png)

- 完整临时输出：`/tmp/chartground_edit_phase3b_balanced_val`
- 标量记录：`results/phase3b_balanced_val_metrics.jsonl`
- 聚合结果：`results/phase3b_balanced_val_summary.json`
- gallery：`assets/phase3b_balanced_val_prompt_benchmark.png`

## 最终验证

- Phase 3B 专项测试：19 passed。
- ChartGround-Edit 全量测试：128 passed，188 warnings；warning 均为既有 Pillow
  `mode` 参数弃用提示。
- `compileall`：通过。
- 独立结果复核：192 条记录、192 个保存 mask，0 个尺寸、二值性或重算指标错误。
- `git diff --check`：通过。

metric-semantics 收尾只读取既有标量结果和保存 mask，没有加载模型或运行推理；冻结
protocol、Prompt、IoU/Dice、bootstrap 和 P2 选择均未改变。

## 结论与下一阶段边界

P2 `target_only_zh` 是按冻结规则得到的唯一全局 Prompt，应原样用于后续独立 Phase 3C
frozen synthetic_v1 test baseline。该结论只适用于 synthetic_v1 validation，不是 test
结论或真实图表泛化结论。Phase 3C 仍需单独审核；本阶段没有运行 test、训练或微调。
