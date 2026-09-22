# Phase 4B-0 `[SEG]`—mask 对齐协议

状态：**Phase 4B-0 静态门禁与 Phase 4B-1 真实单步 smoke 均已通过。**

## 原始错误与真实复现

冻结样本为 `cgev1_bar_category_6d51bac154`。使用本地
Sa2VA-InternVL3-2B tokenizer、冻结 P2、official assistant target、
`Sa2VABaseDataset` 的图像/tokenization 方法和 `sa2va_collect_fn` 得到长度 1844 的
真实 batch。只列出与门禁有关的位置：

| `[SEG]` 位置 | role | label | ignore |
|---:|---|---:|---|
| 1823 | user | -100 | 是 |
| 1840 | assistant | 151674 | 否 |

`seg_token_idx=151674`。旧条件 `input_ids == seg_token_idx` 选择 `[1823,1840]`；新条件只
选择 `[1840]`。GT 是一个非空 `torch.uint8 [1,320,480]` 二值 mask，前景像素 8127。
旧 `check_obj_number(fix_number=5)` 对 2 token/1 mask 先按最小数量截为 1/1，再重复为
5/5；因此不仅静默改数，还恰好丢掉第二个、受监督的 assistant token。

复现命令只加载 tokenizer 和轻量预处理，不加载或构建模型：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
projects/sa2va/.venv/bin/python \
  projects/chartground_edit/scripts/diagnose_phase4b_alignment.py \
  --checkpoint <LOCAL_SA2VA_HF_DIR> \
  --manifest projects/chartground_edit/data/synthetic_v1/annotations.jsonl \
  --selection projects/chartground_edit/configs/phase4_smoke1_ids.json
```

## 选择公式与 strict 规则

训练态目标语义固定为：

```text
supervised_seg_mask =
    (input_ids == seg_token_idx) AND (labels != ignore_index)
```

它不依赖 `[SEG]` 的先后顺序，也不选择“第二个”或“最后一个”。`labels` 必须存在，且与
`input_ids` 的 shape、device 一致，且二者都为 `torch.long`；否则立即报错。causal LM 的 `labels`
原样传给 MLLM，不被改写。

ChartGround 配置使用 `object_count_policy=strict_one_to_one`：每个样本必须恰好一个
supervised `[SEG]`、恰好一个 GT mask，且两者相等。dataset collator 先检查一次；
`Sa2VAModel.forward` 在调用 MLLM 之前再次检查。错误包含 sample ID、两侧数量、token
位置和 policy。strict 分支不会调用 `check_obj_number`，所以不会截断、复制、补齐或跨
样本匹配。

## 实现边界与 legacy 兼容

项目侧新增正式 `ChartGroundPhase4Dataset` 和 `chartground_sa2va_collect_fn`，复用官方
`Sa2VABaseDataset._process_single_image`、conversation encoding 和
`sa2va_collect_fn`。模型只接收 image、P2、official target、mask 和 sample ID；
`full_instruction`、action、edit parameters 和 target attributes 不进入模型。

上游没有 token-selection hook，因此对 `projects/sa2va/models/sa2va.py` 做了小范围、
通用且默认不变的修改：增加 `select_seg_token_mask`、严格对齐检查和四个显式配置字段，
并在 strict 分支绕过 legacy `check_obj_number`。未复制 `forward`，未 monkey patch，HF
`predict_forward` 和各推理目录未修改。默认仍为：

```text
seg_token_selection=all
object_count_policy=legacy_fix_number
expected_masks_per_sample=None
ignore_index=-100
```

潜在 upstream PR 只需要上述通用模型 hook、配置校验和单元测试；ChartGround dataset、
策略 A 冻结/保存与项目配置不应进入该 PR。

## P2 与真实数据契约

P2 registry 文本和 SHA-256
`37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806` 均未改变；用户
Prompt 和 assistant target 各保留一个 `[SEG]`。smoke1 只来自 train。原图为
480×320，GT 保持原图尺寸、二值、非空；MLLM 支路得到 7×3×448×448 tile，SAM 支路
得到 3×1024×1024。Phase 4B 配置不启用 crop/flip 等随机几何增强；SAM 图像和之后由
nearest resize 的 GT 都使用同一个完整画幅缩放，不存在只变换一侧的增强。

## 单步配置契约

`configs/phase4b_smoke1.py` 基于官方 `sa2va_in30_2b.py`/`sa2va_finetune.py`：只读
smoke1，micro batch=1，accumulation=1，BF16，`max_iters=1`，seed 20260916，无
val/test、无 resume，输出 `<WORK_DIR>`。LLM、vision、InternVL
`mlp1` 和整个 SAM2 冻结，只开启 `text_hidden_fcs`。模型构建后会核验所有 trainable name
前缀和预注册参数量 2,754,304。非有限 total loss 在 backward 前失败，非有限 gradient
由 `error_if_nonfinite=True` 的 grad clipping 失败；logger interval=1 会记录三个 loss、
total loss、grad norm，hook 补充 peak CUDA memory。checkpoint metadata 保存完整对齐策略。

Phase 4B-0 时配置只做了解析；Phase 4B-1 已由专用单步脚本真实构建并执行一次。路径通过 MMEngine 环境变量替换，也可用 CLI
`--cfg-options` 覆盖：`CHARTGROUND_BASE_MODEL_PATH`、`CHARTGROUND_SA2VA_PTH`、
`CHARTGROUND_SA2VA_HF_REVISION`。版本化配置不含用户名绝对路径。

## 唯一 checkpoint 输入/输出方案

1. 源是固定 revision `15837dcaecc304714a1f0f069e74f47e47521c7f` 的完整
   Sa2VA-InternVL3-2B HF 目录；本地目录的 index 声明 2,316,157,234 parameters、
   8,656,605,384 bytes，实际目录约 8.1 GiB。
2. 使用 `tools/convert_to_pth.py <HF_DIR> --arch-type internvl --save-path
   <EXTERNAL_DIR>/sa2va_full_bf16.pth` 生成唯一训练输入。
   该工具加载整个 HF 模型，把 `vision_model/language_model/mlp*` key 映射回
   `mllm.model.*`，把 `.g_weight` 还原为 `.gamma`，然后保存 raw state dict。它以
   `torch_dtype=torch.bfloat16` 加载，所以不只是字符串 key mapping：浮点权重会按
   Transformers 加载规则进入 BF16。实际输出为 4,632,880,109 bytes，SHA-256
   `5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6`；1,589 个 tensor
   均为 BF16，共 2,316,157,490 parameters。
3. 训练还需要匹配的本地 `OpenGVLab/InternVL3-2B` base 目录；`InternVLMLLM` 先从该
   base 构建，再由上述 full PTH 覆盖 Sa2VA 权重。实际固定 repo/revision 为
   `OpenGVLab/InternVL3-2B` / `899155015275a9b7338c7f4677e19c784e0e5a21`。revision 由版本化配置、实验记录和
   训练 checkpoint 的 `chartground_phase4b` metadata 保存；`convert_to_pth.py` 自身不写
   revision metadata。
4. 单步输出固定为 `<WORK_DIR>/iter_1.pth`。策略 A subclass 的
   `state_dict()` 只保存四个 `text_hidden_fcs.*` tensor；MMEngine wrapper 另含 config/meta，
   不含 optimizer。实际文件为 11,020,648 bytes、4 个 FP32 tensor、2,754,304 elements，
   SHA-256 `c99044844a51b6c109908d957718aaf91a75177f5c9e5d755fb2c75eb8087936`。
5. 正式 `load_projection_checkpoint` 只接受与当前四个 trainable names 完全相同的 key
   集合，并核验参数量 metadata；真实运行已将内存 projection 清零后逐 tensor 精确恢复。
   按本轮约束没有构建第二个 2B 模型，也没有导出完整 HF。现有 HF inference backend
   若要直接使用微调结果，后续仍需显式完整 HF 导出；这不是进入 Phase 4C 训练的门禁。

## Phase 4B-1 真实结果

只读取 smoke1/train，执行恰好一次 forward、backward 和 optimizer.step。language、mask
CE、Dice、total loss 分别为 0.2747028172、0.0920886174、0.0545147285、0.4213061631，
全部 finite。clip 前 gradient norm 为 27.5787943892；4/4 projection tensors 有非零有限
gradient，冻结参数 gradient 数为 0。step 后 changed elements=2,754,304，最大/平均绝对
变化为 `4.0072947741e-05` / `1.5423816660e-05`。构建、forward、backward、step 分别为
49.9327/0.9984/0.0792/0.0616 秒；峰值 allocated/reserved 为 7,211.7173/7,570.0 MiB。
没有 OOM 或重试；进程退出后 GPU 显存已释放。P2 registry/template hashes 均未改变，
strict 记录仍为 assistant position 1840 对一个 GT mask，未调用 `fix_number=5`。
