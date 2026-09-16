# Phase 3C frozen synthetic_v1 test protocol

状态：预注册并冻结  
协议版本：`phase3c-frozen-test-v1`  
预注册日期：2026-09-16  
Bootstrap seed：`20260916`  
Bootstrap iterations：`10000`

## 实验边界

本协议只允许读取 `synthetic_v1` 的 `split=test`。预期样本数和模型调用数均为 64；
每个样本按 manifest 顺序且只运行一次。禁止访问 train 推理、重新运行 val、比较 Prompt、
根据 test 结果修改 Prompt 或选择性重跑任何空预测、低 IoU、非法输出或异常样本。
若整个进程因硬件、驱动或断电中断，本次完整运行作废；记录原因后只能从头重跑完整
64 条，不能补跑子集。

## 固定模型与数据身份

- 模型：`ByteDance/Sa2VA-InternVL3-2B`
- checkpoint：`/home/dqwang/Model/Sa2VA-InternVL3-2B`（CLI 必填，代码不设本机默认值）
- revision：`15837dcaecc304714a1f0f069e74f47e47521c7f`
- dtype：BF16
- 单物理 GPU 映射为进程内 `cuda:0`；不使用 `device_map`、量化或 CPU offload
- `torch.inference_mode()`、`local_files_only=True`、`use_flash_attn=True`
- manifest：`projects/chartground_edit/data/synthetic_v1/annotations.jsonl`
- manifest SHA-256：`ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`
- test sample-ID list SHA-256：`3061387f0012b13c2fd998d81819df0e7648e8e47ae9ad86ff06490ecfe2e5c4`

sample-ID list hash 的输入是 test 样本按 manifest 顺序排列的 64 个 `sample_id`，每个 ID
后接一个 LF，包括最后一个 ID 后的终止 LF。

## 固定 Prompt

唯一允许的 Prompt 是 Phase 3B 已选择的 P2 `target_only_zh`。runner 必须从
`chartground_edit.inference.prompt_variants` registry 调用，不能复制、重写、翻译、
补充或组合 Prompt。精确模板为：

```text
<image>请分割图中由以下指代表达式指定的图表元素：{referring_expression}
请使用 [SEG] 标记返回分割掩码。
```

- P2 template SHA-256：`37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806`
- Prompt registry SHA-256：`dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0`

模型输入只能是图像和 P2 使用该条 `referring_expression` 生成的精确 Prompt。禁止把
`full_instruction`、`edit_action`、`edit_parameters`、`target_attributes`、GT mask、目标
几何或其他隐藏 metadata 传给模型。动作与参数只允许在模型推理完成后进入编辑后端。

## Test 组成与调用规则

- test 恰好 64 条，覆盖全部 16 个 `chart_type × referring_type` 组合。
- 每组合恰好 4 条，`highlight`、`recolor`、`extract`、`remove` 各 1 条。
- 一个 backend 实例，模型与 tokenizer 只加载一次。
- 每条样本一次调用，总调用数严格为 64；不重试、不替换、不修复预测。
- `prediction_masks` 永远只取第一个；不使用 GT 选择 mask。
- mask 后处理沿用 Phase 3B：只移除 leading singleton 维；接受 bool、0/1、0/255；
  NaN/Inf、其他值域和非 singleton 多余维度非法；尺寸不同时 nearest-neighbor 恢复原图。

## Metric semantics 与异常处理

每条记录明确包含且分别解释：

- `execution_success`：`predict_forward` 正常完成且没有运行时异常。
- `mask_contract_valid`：输出按冻结规则归一化为原图尺寸二值 mask；全零 mask 仍合法。
- `segmentation_token_present`：生成文本包含 `[SEG]`。
- `nonempty_prediction` / `empty_prediction`：合法归一化 mask 是否含前景，二者互补。
- `overlapping_prediction`：预测与 GT 至少有一个交集像素。
- `nonempty_disjoint`：预测非空但与非空 GT 无交集。

记录还包含 predicted/GT foreground pixels、intersection、union、IoU、Dice、`resized`、
`generated_text`、raw mask metadata、latency 以及 error type/message。不得使用 deprecated
`inference_success` 表示程序执行成功，Phase 3C 新记录不写该字段。

合法空 mask 为 `execution_success=True`、`mask_contract_valid=True`、
`nonempty_prediction=False`、IoU=Dice=0。程序异常或非法 mask 保存原尺寸全零 failure
placeholder，使 IoU=Dice=0 并纳入所有正式聚合分母；它不是模型预测，也不得用于编辑。

## 正式指标与 bootstrap

整体报告 execution success、mask contract valid、`[SEG]`、nonempty/empty、overlap、
nonempty-disjoint 的 count/rate；16-group Macro IoU/Dice、sample Macro IoU/Dice、Micro
IoU/Dice、median IoU/Dice、worst-group IoU 和 mean/median latency。分组维度为 16 个
chart/referring 组合、chart type、referring type、edit action、difficulty 和
distractor count。

16-group Macro 先在每个 chart/referring 组合内对 4 条样本取均值，再对 16 组等权平均。
使用 NumPy `default_rng(20260916)`，以 16 个 group 为抽样单位、有放回抽取 16 组，执行
10,000 次 bootstrap，分别报告 test 16-group Macro IoU 和 Dice 的 2.5%/97.5% 分位数。
该区间只是 test 估计区间，不用于修改 Prompt 或返回 validation 重新选择。

## 编辑闭环

每个合法非空预测把该样本真实的 `edit_action` 和 `edit_parameters` 交给现有确定性编辑
后端；编辑 mask 必须与保存的 predicted mask 是同一二值数组，绝不使用 GT mask 替代。
空预测或无合法预测记录 `edit_skipped_empty=True`；非空预测记录
`edit_execution_success` 及错误信息。编辑成功只说明链路运行成功，不代表分割正确。
完整编辑输出保存在临时目录，不全部加入 Git。

## Gallery 与产物

Gallery 按固定 chart type 顺序 `line,bar,scatter,confidence_band` 和 referring type 顺序
`category,appearance,legend,trend` 覆盖全部 16 组；每组选择 test 中 `sample_id` 字典序
最小者，与 IoU、Dice、empty 或成功状态无关。每行展示 original、GT overlay、P2
prediction overlay 和 edited result；空预测明确显示 `EMPTY / EDIT SKIPPED`，不隐藏
IoU=0 或失败案例。

完整临时输出固定为 `/tmp/chartground_edit_phase3c_frozen_test`。仓库只保存 protocol、
结果报告、64 条标量 JSONL、summary JSON 和 16 组 gallery。

## 协议完整性

runner 在加载模型前计算本文件 SHA-256，并在全部 64 次尝试结束后再次计算；两次必须
完全一致。若 hash 改变，整次运行作废。本协议不包含 Prompt 搜索、选择、路由或 test
结果反馈逻辑；P2 无论 test 结果如何都保持冻结。
