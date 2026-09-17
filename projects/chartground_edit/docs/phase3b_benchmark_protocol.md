# Phase 3B balanced synthetic_v1 val Prompt benchmark protocol

状态：预注册并冻结  
协议版本：`phase3b-balanced-val-v1`  
预注册日期：2026-09-16  
Bootstrap seed：`20260916`

## 实验边界

本协议只允许读取 `synthetic_v1` 的 `split=val`。预期样本数为 64，覆盖全部 16 个
`chart_type × referring_type` 组合，每组合 4 条且四种 edit action 各 1 条。每条样本
分别运行 P0、P1、P2 一次，预期总尝试数为 192。禁止运行或查看模型在 train/test 上
的输出，禁止根据中间结果修改 Prompt、阈值、mask 选择或本协议。

## 固定模型与数据身份

- 模型：`ByteDance/Sa2VA-InternVL3-2B`
- revision：`15837dcaecc304714a1f0f069e74f47e47521c7f`
- checkpoint：必须由 CLI 的 `--checkpoint` 参数传入；本机验证路径为
  `<MODEL_ROOT>/Sa2VA-InternVL3-2B`，代码中不得作为默认值硬编码。
- dtype：BF16
- 单 GPU；不使用量化、CPU offload 或 `device_map`
- `local_files_only=True`、`use_flash_attn=True`
- manifest：`projects/chartground_edit/data/synthetic_v1/annotations.jsonl`
- manifest SHA-256：`ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`
- 预注册代码基线：`b022e4d10c8f6fc922fd3d18790ef999b85f36a2`

## 固定 Prompt

顺序固定为 `full_instruction,target_only_en,target_only_zh`。模板直接复用
`chartground_edit.inference.prompt_variants`，不复制到运行脚本。

P0 `full_instruction`：

```text
<image>Please segment the chart element targeted by this instruction: {instruction}
Please respond with a segmentation mask.
```

SHA-256：`175bf7d46db1ee62eb73a2111850c05d1a53659f142a3409dd711f3c2e14031a`

P1 `target_only_en`：

```text
<image>Please segment the chart element described by this referring expression: {referring_expression}
Please respond with a segmentation mask.
```

SHA-256：`5260f3469f0da8708e6df4c6461ae2fbb851dc668a9210946d863da7059008fc`

P2 `target_only_zh`：

```text
<image>请分割图中由以下指代表达式指定的图表元素：{referring_expression}
请使用 [SEG] 标记返回分割掩码。
```

SHA-256：`37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806`

Prompt registry 使用 variant 名到模板文本的 sorted compact JSON 计算 SHA-256：
`dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0`。

## 调度

模型和 tokenizer 在同一进程中只加载一次。样本按 manifest 中 val 顺序确定性运行。
第 `i` 个样本的 Prompt 顺序将 canonical order 左旋 `i mod 3`：P0/P1/P2、P1/P2/P0、
P2/P0/P1，之后循环。每次记录 `prompt_order_position`。latency 只作描述，不参与选择。

## Mask 与失败处理

- 只使用 `prediction_masks` 的第一个 mask，不读取 GT 选择 mask。
- 只移除明确的 leading singleton 维度；非 singleton 多余维度明确失败。
- bool、0/1、0/255 可接受；非法 dtype、NaN/Inf 或其他值域明确失败。
- 尺寸不同时以 nearest-neighbor 恢复原图尺寸。
- 无 mask、空 mask、无 `[SEG]` 和 protocol/inference failure 都原样记录，不替换、不重试。
- GT 全部非空；空预测或失败在聚合中以 IoU=0、Dice=0 计入，且不从分母删除。
- 如果进程因机器、驱动或断电中断，整次 192-attempt run 作废；只允许从头重跑完整协议，
  不允许选择性补跑。

## 指标

每个 Prompt 报告 attempt、inference success、`[SEG]`、empty、nonempty-disjoint、
overlap、sample macro/median IoU 和 Dice、micro IoU/Dice、worst-sample IoU、平均/中位
latency。分组维度为 16 个 chart/referring 组合、chart type、referring type、edit action、
difficulty 和 distractor count；每组至少报告 count、mean IoU、mean Dice、empty rate 和
nonempty-disjoint rate。

Primary metric 是 16-group Macro IoU：先在每个 chart/referring 组合内对 4 条样本取
mean IoU，再对 16 个组等权平均。Secondary metrics 是 16-group Macro Dice、
nonempty-disjoint rate、worst-group mean IoU 和 empty prediction rate。Micro 指标不能
替代 primary metric。

## 预注册全局 Prompt 选择规则

1. 选择 16-group Macro IoU 最高的 Prompt。
2. 如果数值精确相同到 `1e-6`，选择 16-group Macro Dice 更高者。
3. 如果仍精确相同到 `1e-6`，选择 nonempty-disjoint rate 更低者。
4. 如果仍相同，保留 P0 `full_instruction`；若 P0 不在剩余并列项中，按 P0/P1/P2
   canonical order 取最先者。
5. latency 不参与 Prompt 选择。
6. 不按 chart/referring/action/difficulty 选择不同 Prompt。
7. per-type oracle 只作分析，不是正式选择。

## Paired bootstrap

对 16 个 chart/referring group 做 10,000 次 paired bootstrap。每次从 16 组中有放回
抽取 16 组，计算正式 selected Prompt 相对每个其他 Prompt 的 group-macro IoU 差值。
使用 NumPy `default_rng(20260916)`，报告 2.5% 和 97.5% 分位数。bootstrap 只描述
不确定性，不改变选择规则；CI 包含 0 时只能称为 “selected on validation”，不得称为
显著优于。

## 协议完整性

运行脚本在加载模型前计算本文件 SHA-256，并在全部尝试结束后再次计算。两次必须完全
一致；若变化，整次 benchmark 作废且不按结果修改本文件。synthetic_v1 test 在代码
边界显式拒绝，并保持冻结，只有后续独立 Phase 3C 才能在选定全局 Prompt 下运行一次。
