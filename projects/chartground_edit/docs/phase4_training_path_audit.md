# Phase 4A Sa2VA 训练路径审计

审计日期：2026-09-16  
范围：源码、配置、纯数据与静态参数估算；未加载 checkpoint，未构建模型，未执行
forward/backward/optimizer，也未访问 val/test 推理。

## 1. 环境与入口

当前仓库使用 `projects/sa2va/.venv/bin/python`：Python 3.11.16、PyTorch
2.6.0+cu124、Transformers 4.57.1、PEFT 0.17.1、MMEngine 0.10.7、XTuner
0.1.23、DeepSpeed 0.18.0。`projects/sa2va/pyproject.toml` 的 `latest` extra 正好固定
Transformers 4.57.1 / PEFT 0.17.1；InternVL3 配置应使用这一组，而不是 `legacy`
组。`setup_env.sh` 默认也是 `latest`。

官方启动入口为 `tools/train.py`：它替换 `xtuner.tools.train.parse_args` 后进入
`xtuner.tools.train.main`，Python config 由 MMEngine 读取，随后构建 dataset、model、
optimizer wrapper 和 `TrainLoop`。`tools/dist.sh` 用 `torchrun`（或旧
`torch.distributed.launch`）传入 `--launcher pytorch` 和 DeepSpeed 配置。官方 README
给出的训练命令是 8 GPU，并建议至少 8 张 A100；仓库没有宣称 RTX 3090 单卡受支持。
`tools/train.py --launcher none` 是代码存在的单进程入口，但“可启动”不等于官方对
24 GB 单卡做过容量验证。

## 2. 完整训练调用链

1. **配置与构建**：`projects/sa2va/configs/sa2va_finetune.py` 是图像指代分割微调示例；
   `projects/sa2va/configs/sa2va_in30_2b.py` 是完全匹配 InternVL3-2B 的官方训练配置。
   两者都由 `tools/train.py` 交给 XTuner/MMEngine registry 构建。
2. **checkpoint 加载**：`InternVLMLLM.__init__` 继承 XTuner
   `InternVL_V1_5`，后者用 `AutoConfig.from_pretrained`、`AutoModel.from_pretrained`
   和 `AutoTokenizer.from_pretrained` 加载 `model_path` 的 InternVL 基座；
   `Sa2VAModel.__init__` 再用 `guess_load_checkpoint(pretrained_pth)` 非严格加载 Sa2VA
   训练态权重。官方 fine-tune config 明确要求一个 `.pth`。
3. **HF→训练态转换**：`tools/convert_to_pth.py:main` 会真实构建 HF model、反向映射
   `vision_model/language_model/mlp*` key 并保存 `.pth`。本轮禁止加载模型，因此没有执行。
   现有冻结 checkpoint 是 HF 推理目录；Phase 4B 前必须在获准的独立预处理步骤验证
   转换结果和 missing/unexpected keys。
4. **特殊 token**：`Sa2VAModel._add_special_tokens` 调用
   `InternVLMLLM.add_special_tokens`；后者执行 `tokenizer.add_tokens(...,
   special_tokens=True)` 并在确有新 token 时 resize language embedding。`seg_token_idx`
   来自 `tokenizer("[SEG]", add_special_tokens=False).input_ids[0]`。
5. **对话和 label**：官方 `Sa2VAFinetuneDataset._parse_annotations` 对每个 phrase 随机选
   `SEG_QUESTIONS` 和 `ANSWER_LIST`，首问前置 `<image>\n`；
   `Sa2VADatasetMixin._process_conversations_for_encoding` 将 `<image>` 替换成视觉 token；
   `template_map_fn` 套 `PROMPT_TEMPLATE.qwen_chat`；`tokenize_conversation` 给 BOS 和用户
   input 的 label 填 `IGNORE_INDEX=-100`，assistant output 与 EOS 保留真实 label。
6. **InternVL 图像路径**：`Sa2VADatasetMixin._process_single_image` 在默认
   `single_image_mode=False` 下调用 `dynamic_preprocess`，按宽高比切 1–12 个 448×448
   tile，必要时加 thumbnail，再做 RGB、bicubic、ToTensor 和 ImageNet normalize。
7. **SAM2 图像路径**：同一原图先经配置的 `DirectResize(target_length=1024)` 直接变为
   1024×1024，输出 CHW uint8 `g_pixel_values`；`SAM2TrainRunner.preprocess_image`
   再除以 255 并做 ImageNet normalize。该路径和 MLLM tile 路径不同，但都来自同一
   未裁剪原图。
8. **GT mask 路径**：官方 finetune loader 将 polygon rasterize 为原图尺寸 uint8
   `[N,H,W]`。`sa2va_collect_fn` 保留为 batch 内 tensor list，并产生
   `frames_per_batch=[1,...]`。`Sa2VAModel.forward` 在预测产生后用 nearest
   `F.interpolate` 把 GT 调到 low-resolution `pred_masks` 的空间尺寸再计算 loss；没有
   对 GT 做 InternVL dynamic tiling。
9. **`[SEG]` 到 mask**：`Sa2VAModel.forward` 用
   `seg_token_mask = input_ids == self.seg_token_idx`，从最后层 hidden states 中取全部
   `[SEG]` 位置，经过 `text_hidden_fcs`。`check_obj_number` 先把 token/mask 数量截到
   两者最小值，然后默认将双方随机截断或重复到 `fix_number=5`。这不是严格失败式的一
   token/一 mask 校验。
10. **投影和 SAM2**：`text_hidden_fcs` 是
    `Linear(llm_hidden,llm_hidden) → ReLU → Linear(llm_hidden,256)`；
    `SAM2TrainRunner.get_sam2_embeddings` 调冻结 image encoder，
    `inject_language_embd` 将投影结果作为 `language_embd`；扩展
    `projects.sa2va.models.extension.SAM2Base._forward_sam_heads` 将其拼到 sparse prompt
    embedding，再调用 `sam_mask_decoder`。
11. **loss**：`InternVLMLLM._compute_loss` 是 shifted causal cross entropy；
    `Sa2VAModel.forward` 另返回 `loss_mask` 和 `loss_dice`。2B/finetune 配置的 mask
    loss 是 sigmoid `CrossEntropyLoss`、weight 2.0；Dice 是 sigmoid activated、naive
    Dice、eps 1.0、weight 0.5。`loss_sample_points=True` 时每 mask 采 12,544 个不确定
    点；返回 dict 的三项由 MMEngine 作为 loss 项求和，因此总权重为
    `1.0*llm_loss + 2.0*mask_CE + 0.5*Dice`（具体 loss 类内部应用后两项权重）。
12. **冻结和 optimizer**：`Sa2VAModel` 先冻结整个 grounding encoder，配置
    `frozen_sam2_decoder=False` 时只重新开启 `sam_mask_decoder`；`text_hidden_fcs`
    默认训练。InternVL wrapper 冻结 language model 与 `vision_model` 后，**没有冻结**
    InternVL 的视觉-语言 projector `mllm.model.mlp1`；PEFT 再开启 LoRA 和
    `modules_to_save`。配置没有 paramwise optimizer 过滤；MMEngine 默认 constructor
    将模型参数交给 AdamW，真正有梯度/状态的是 `requires_grad=True` 的集合。
13. **精度与调度**：官方是 AdamW，lr 4e-5，betas 0.9/0.999，weight decay 0.05，
    max-norm 1；`AmpOptimWrapper(dtype='bfloat16', loss_scale='dynamic')`；2B 主配置
    batch 2、accumulation 8（注释为 16 GPU），fine-tune 配置 accumulation 16（注释为
    8 GPU）。Linear warmup 占 epoch 5%，随后 cosine 到 0。XTuner
    `InternVL_V1_5.__init__` 无条件启用 language-model gradient checkpointing。
14. **保存/恢复**：`CheckpointHook` 每 2000 iter 保存、最多 2 个、
    `save_optimizer=False`。`Sa2VAModel.state_dict` 只保留 MLLM wrapper 挑出的权重、
    SAM2 mask decoder 和 `text_hidden_fcs`；`InternVL_V1_5.state_dict` 在 LoRA 模式下
    保存 PEFT adapter 与 `modules_to_save`，并总是保存 MLLM `mlp1`。`--resume`/config
    `load_from` 走 MMEngine checkpoint；由于官方配置不保存 optimizer，不能把其称作
    完整 optimizer-state 恢复。
15. **训练态→推理态**：`tools/convert_to_hf.py` 构建训练模型、载入 `.pth` 的
    `state_dict`、`_merge_lora()`、调用 `all_state_dict()` 并映射 key，最后
    `save_pretrained` 成现有 HF backend 可读取的目录。现有
    `Sa2VAInternVL3Backend` 不直接接收独立 PEFT adapter；验证性重载应走转换后的新 HF
    目录。转换与重载本轮均未执行。

## 3. 官方 InternVL3-2B / LoRA 配置

`sa2va_in30_2b.py` 是完全匹配的 2B 训练配置；`sa2va_finetune.py` 也以
`OpenGVLab/InternVL3-2B` 为默认 base，并要求 Sa2VA `.pth`。结论如下：

| 项目 | 当前源码事实 |
|---|---|
| 依赖 | InternVL3 使用 `latest`：Transformers 4.57.1、PEFT 0.17.1 |
| LLM / vision | `freeze_llm=True`、`freeze_visual_encoder=True`，但随后给 LLM 注入 LoRA；视觉-语言 projector `mllm.model.mlp1` 不属于这两个被冻结子模块，保持可训练 |
| LoRA target | config 未写 `target_modules`；XTuner `find_all_linear_names` 动态找全部 `nn.Linear` leaf name，并移除 `lm_head/output_layer`。对本地 HF config 所示 Qwen2 层，准确 leaf names 为 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`；仍须在 Phase 4B 实例化后打印集合核验 |
| LoRA 超参 | r=128、alpha=256、dropout=0.05、bias=none、CAUSAL_LM |
| embedding/lm_head | `modules_to_save=["embed_tokens","lm_head"]`，PEFT 会令其可训练并保存；因此官方“LoRA”绝非只有 LoRA A/B |
| `[SEG]` embedding | 若 `[SEG]` 是新增 token，resize 后 embedding 位于可训练的 `embed_tokens`；整个 embedding matrix 而非只一行进入 `modules_to_save` |
| `text_hidden_fcs` | 默认可训练并由 `Sa2VAModel.state_dict` 保存 |
| SAM2 | 整体先冻结；`frozen_sam2_decoder=False` 只开启 `sam_mask_decoder`。prompt encoder、image encoder、memory 模块与 `obj_ptr_proj` 保持冻结 |
| optimizer | 无 paramwise 排除；LoRA、embedding/lm_head、`mllm.model.mlp1`、`text_hidden_fcs`、SAM2 mask decoder 都在实际可训练集合，冻结参数无梯度 |
| checkpointing | language model gradient checkpointing 默认开启 |
| 框架 | MMEngine + XTuner；官方脚本可选 DeepSpeed 并以 torchrun 分布式启动 |
| 单 3090 | 有单进程代码路径，没有官方 3090 支持/容量承诺；README 推荐 8×A100 |

## 4. A/B/C 静态比较

参数量来自 2B HF config 的 hidden=1536、28 个 Qwen2 layer、intermediate=8960、
KV width=256、vocab=151,679，以及当前源码公式；未实例化模型，所以是配置级估算。

| 策略 | 可训练集合与估算 | 能解决 / 不能解决 | 24 GB 风险与保存/推理 |
|---|---|---|---|
| A projection only | `text_hidden_fcs`，2,754,304 params；BF16 参数/梯度各约 5.3 MiB，FP32 Adam 两状态约 21 MiB | 可直接证明 mask loss 到投影层的梯度与参数更新，可重新映射现有语言 hidden 到 SAM prompt；不能改变语言表示、`[SEG]` 生成或 SAM decoder | 三者最低，仍需完整冻结 backbone 的 forward activations；`.pth` 保存 projection（以及 state_dict 固定包含的冻结 MLLM projector/decoder key，需验证实际 checkpoint 内容）。HF 转换后现有 backend 才能加载。最适合 smoke1 和首轮 overfit32；是官方模型/loop的受控消融，不是官方默认 fine-tune 配方 |
| B LLM LoRA + projection | LoRA A/B 约 147.7M；保留官方 `modules_to_save` 时另含 embed+lm_head 约 466.0M，总约 616.4M。为符合 B 的受控定义必须显式冻结 `mllm.model.mlp1` | 可改变语言 grounding 和输出分布；冻结 vision/SAM 限制视觉及 decoder 适应，且全 embedding/head 显著放大训练集合 | BF16 trainable+grad 约 2.30 GiB、FP32 Adam states 约 4.59 GiB（不含 base/activation）；24 GB OOM 风险高。保存 PEFT/modules-to-save+projection，需 HF merge/export。适合 smoke，只有 A 失败且容量确认后才用于 overfit32；只符合官方 LoRA机制，不等同官方完整 trainable 集合 |
| C B + SAM2 mask decoder | B 加 source-derived约 4.22M decoder，总约 620.6M；仍按三策略定义显式冻结 `mllm.model.mlp1` | 再允许 mask head适应细线/图表形状；仍不改 image encoder；更容易过拟合，也更难判断改进来自何处 | 比 B 额外参数小，但 SAM decoder backward activation 增加显存。保存 adapter/modules-to-save+projection+decoder，再转换 HF。可做 smoke，不是单卡首选 overfit；最接近官方配置，但官方还会训练约 8.66M 的 `mlp1` |

按本地 HF config 静态计算，`mllm.model.mlp1` 约 8,662,016 参数；所以不做额外冻结的
官方 C 实际总可训练量约 629.3M，而不是 620.6M。该数仍需 Phase 4B 实例化后的
`named_parameters()` 作为最终事实。

推荐 Phase 4B 从 **A** 开始，因为它最直接验证 mask-loss 梯度闭环，且不会把
embedding/lm-head 的约 4.66 亿参数混入“LoRA”叙述。只有 A 的单步保存/重载/合法 mask
全部通过后，才考虑 B，再考虑 C。

## 5. 必须先解决的硬门禁

冻结 P2 模板自身含有 `[SEG]`，assistant target `Sure, [SEG].` 又含一个。官方
`Sa2VAModel.forward` 不看 label，只取输入序列中全部 `[SEG]`；`check_obj_number` 面对
2 token/1 mask 会先保留第一个 token（用户 Prompt 中的 token）和一个 mask，再重复到
5，assistant 的监督 token 被丢弃。因此当前源码不能同时满足“P2 原样进入模型”、
“official assistant target 含一个 `[SEG]`”和“assistant `[SEG]` 唯一对应 GT”。

Phase 4B 前必须预先批准并测试一种不改 P2 文本的实现方式，首选是让 segmentation
位置选择同时满足 `input_ids == seg_token_idx` 与 `labels != IGNORE_INDEX`，并把数量不符
改为明确报错或为静态图像配置 `fix_number=1`。这需要一个最小、可审计的训练侧模型
适配点；当前 monolithic `forward` 没有 hook，不能在不改/不复制核心 forward 的情况下
完成。本轮没有实现该修改，也没有宣称训练链路可用。

## 6. 资源与 OOM 回退顺序

冻结 HF checkpoint 的 BF16 总权重仅按“约 2B base + SAM2”估算为约 4–6 GiB；CUDA
kernel/workspace、InternVL 最多 12 个 448 tile、1024 SAM2 分支和反向保存的 activation
才是单卡不确定项。A 的 trainable state 约 32 MiB（BF16 param+grad、FP32 Adam 两状态），
B/C 则仅 trainable state 已约 6.9–7.0 GiB，若 optimizer 另保 FP32 master copy还会更高。
因此不能从静态参数量宣称 3090 可运行。

未来 OOM 回退顺序被冻结为：保留官方 language gradient checkpointing；micro-batch 降到
1；用 gradient accumulation 恢复有效 batch；降低 dynamic tile 上限或 image/SAM 输入
分辨率且对 mask 做同一几何映射；降低 LoRA rank；只在仓库明确支持时考虑其他方案。
不建议仓库当前训练配置未证明支持的量化训练、CPU offload 或依赖替换。
