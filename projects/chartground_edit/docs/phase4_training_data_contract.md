# Phase 4 synthetic_v1 train-only 数据契约

版本：`phase4-training-data-contract-v1`  
源 manifest SHA-256：`ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`

## 模型边界

未来训练 adapter 每条只允许向模型侧提供：RGB image、`referring_expression` 生成的冻结
P2 Prompt、binary GT mask、image width/height 和 sample ID。`full_instruction`、
`edit_action`、`edit_parameters`、`target_attributes`、`difficulty`、scene/content ID、目标
几何与 generation metadata 都是审计字段，绝不拼入模型文本。

P2 必须由 `chartground_edit.inference.prompt_variants.build_prompt_variant` 调用
`target_only_zh`，模板原文为：

```text
<image>请分割图中由以下指代表达式指定的图表元素：{referring_expression}
请使用 [SEG] 标记返回分割掩码。
```

其冻结模板 SHA-256 为
`37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806`，registry 语义
SHA-256 为 `dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0`。

## 对话、label 与 mask

为去除官方 loader 对 `ANSWER_LIST` 的随机选择，本项目冻结官方列表中的一个合法答案：

```json
[
  {"from": "human", "value": "<exact P2 prompt>"},
  {"from": "gpt", "value": "Sure, [SEG]."}
]
```

assistant target 恰好含一个 `[SEG]`。按官方 `template_map_fn` 和
`tokenize_conversation`，BOS 与整个用户 input 的 labels 为 -100；assistant target 和
EOS 的 labels 为真实 token ID。因此语言 loss 只监督 answer/EOS。

原始 GT 必须是非空二值 PNG，与 RGB 的 `(width,height)` 完全一致。本项目纯数据 adapter
返回 `np.uint8 [1,H,W]`、值域 `{0,1}`；接入官方 collate 时转为
`torch.uint8 [1,H,W]`，batch 侧包装为 `list[tensor]`。同一条静态图像的
`frames_per_batch=1`；SAM 输入为 `uint8 [3,1024,1024]`。预测产生后，官方
`Sa2VAModel.forward` 用 nearest 把 GT resize 到 low-resolution mask logits 的空间尺寸，
mask loss 在该分辨率计算。schema 已禁止空 GT；本项目不定义空 GT 训练语义。

## 对齐冲突与 Phase 4B-0 解决方案

“assistant target 恰好一个 `[SEG]`”并不等于完整 token 序列只有一个：冻结 P2 本身也
含 `[SEG]`，所以完整对话有两个。官方 `Sa2VAModel.forward` 目前用
`input_ids == seg_token_idx`，没有利用 -100 labels 排除 Prompt token；随后
`check_obj_number(..., fix_number=5)` 会截断/重复 token 与 mask。因此当前实现下，
P2+official target 会选择用户 Prompt 的 `[SEG]` hidden state，丢弃 assistant 的
`[SEG]`，再把唯一 pair 重复五份。

Phase 4B-0 已实现并用真实 tokenizer/collator 验证：segmentation mask 只选择
`(input_ids == seg_token_idx) & (labels != -100)` 的唯一 assistant 位置。项目 collator 和
训练模型入口都执行 strict one-to-one 检查；数量不符会携带 sample ID、两侧数量、token
位置和 policy 报错。strict 分支在进入 legacy `check_obj_number(fix_number=5)` 前阻止，
并且根本不调用该函数。P2 和 assistant target 均未删改。完整证据见
`phase4b_alignment_protocol.md`。

## 纯数据实现

`chartground_edit.training.Phase4TrainDataset` 只读取冻结 manifest 与 ID 清单，不导入
Sa2VA/Transformers。它强制 manifest hash、selection hash、唯一 ID 和 `split=train`，
再读取 RGB 和二值 mask。它保持现有 v0/v1 reader 不变；训练契约只消费 v1。

审计字段不会出现在 `Phase4DataSample` 中。模型侧字段固定为 `sample_id,image,mask,
image_width,image_height,prompt,assistant_target`。`ChartGroundPhase4Dataset` 已作为正式
bridge 复用官方 tokenizer、动态图块、SAM resize、conversation encoding 和 collator。
它不启用随机 crop/flip；GT 保持原图尺寸，SAM 图像和 loss 前的 GT resize 都是完整画幅
的对应缩放。

## 冻结子集

- smoke1：`configs/phase4_smoke1_ids.json`，在 train 的 bar/category 候选中，计算 mask
  foreground ratio，选离候选中位数最近者，sample ID 打破平局。结果是
  `cgev1_bar_category_6d51bac154`，ratio 0.05291015625，候选中位数
  0.05346028645833333。
- overfit32：`configs/phase4_overfit32_ids.json`。按生成器固定 chart/referring 顺序遍历
  16 组，每组从 12 条 train 中选两条；冻结目标依次最小化全局 action 不平衡/方差、
  同 action、difficulty 不平衡/方差、同 difficulty，随后偏好更宽 distractor 间距，最后
  sample ID 打破平局。结果 action 各 8；difficulty easy/medium/hard=11/10/11；
  distractor 2/3/4=11/10/11。规则不读取任何模型输出或 Phase 3C 指标。

两份清单只保存 sample ID、规则、manifest hash 和聚合审计信息，不保存 scene ID、
content ID、底层内容、图片或 mask 副本。
