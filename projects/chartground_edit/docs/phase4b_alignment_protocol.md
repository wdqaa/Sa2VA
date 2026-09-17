# Phase 4B-0 `[SEG]`—mask 对齐协议

状态：**静态门禁实现完成；训练、模型构建和权重转换均未运行。**

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
val/test、无 resume，输出 `/tmp/chartground_edit_phase4b_smoke1`。LLM、vision、InternVL
`mlp1` 和整个 SAM2 冻结，只开启 `text_hidden_fcs`。模型构建后会核验所有 trainable name
前缀和预注册参数量 2,754,304。非有限 total loss 在 backward 前失败，非有限 gradient
由 `error_if_nonfinite=True` 的 grad clipping 失败；logger interval=1 会记录三个 loss、
total loss、grad norm，hook 补充 peak CUDA memory。checkpoint metadata 保存完整对齐策略。

配置只解析过，没有构建模型。路径通过 MMEngine 环境变量替换，也可用 CLI
`--cfg-options` 覆盖：`CHARTGROUND_BASE_MODEL_PATH`、`CHARTGROUND_SA2VA_PTH`、
`CHARTGROUND_SA2VA_HF_REVISION`。版本化配置不含用户名绝对路径。

## 唯一 checkpoint 输入/输出方案

1. 源是固定 revision `15837dcaecc304714a1f0f069e74f47e47521c7f` 的完整
   Sa2VA-InternVL3-2B HF 目录；本地目录的 index 声明 2,316,157,234 parameters、
   8,656,605,384 bytes，实际目录约 8.1 GiB。
2. 使用 `tools/convert_to_pth.py <HF_DIR> --arch-type internvl --save-path
   /tmp/chartground_edit_phase4b_input/Sa2VA-InternVL3-2B.full.bf16.pth` 生成唯一训练输入。
   该工具加载整个 HF 模型，把 `vision_model/language_model/mlp*` key 映射回
   `mllm.model.*`，把 `.g_weight` 还原为 `.gamma`，然后保存 raw state dict。它以
   `torch_dtype=torch.bfloat16` 加载，所以不只是字符串 key mapping：浮点权重会按
   Transformers 加载规则进入 BF16；非浮点 buffer 不应被描述为 BF16。预计输出仍约
   8.1 GiB。转换本轮未运行。
3. 训练还需要匹配的本地 `OpenGVLab/InternVL3-2B` base 目录；`InternVLMLLM` 先从该
   base 构建，再由上述 full PTH 覆盖 Sa2VA 权重。revision 由版本化配置、实验记录和
   训练 checkpoint 的 `chartground_phase4b` metadata 保存；`convert_to_pth.py` 自身不写
   revision metadata。
4. 单步输出固定为 `/tmp/chartground_edit_phase4b_smoke1/iter_1.pth`。策略 A subclass 的
   `state_dict()` 只保存四个 `text_hidden_fcs.*` tensor；MMEngine wrapper 另含 config/meta，
   不含 optimizer。按 FP32 projection tensor 估计约 11 MiB 原始 tensor 数据，最终文件
   大小待真实运行验证。
5. 重载时仍以相同 base + full PTH 构建模型，再加载 `iter_1.pth` 的 projection subset。
   导出调用 `tools/convert_to_hf.py phase4b_smoke1.py iter_1.pth --save-path
   /tmp/chartground_edit_phase4b_smoke1_hf`；该工具会构建完整模型、加载 subset、调用
   `all_state_dict()` 并导出完整 HF 目录，预计再次约 8.1 GiB。现有 HF inference backend
   只能加载这个完整导出目录，不能直接加载 projection-only `.pth`。

仓库有 `convert_to_hf.py`，不需要另找 `pth_to_hf`；但本轮禁止转换/模型加载，因此
HF→PTH key/dtype 完整性、projection subset 重载和 PTH→HF 后的逐 key/hash 复核仍未实测。
此外本机尚未发现匹配的独立 InternVL3-2B base 目录。两者合并为 Phase 4B-1 前唯一的
checkpoint materialization/round-trip 门禁，不能把静态代码路径当成已无损验证。

## Phase 4B-1 验收标准

在不访问 val/test 的前提下，先完成并审计上述 checkpoint 输入；随后只允许一个 smoke1
optimizer step。必须逐项证明 loss finite、projection gradient finite/nonzero、其余模块无
gradient、grad norm finite、projection 参数发生变化、`iter_1.pth` 只含预期 keys、训练态
新进程重载成功、完整 HF 导出重载成功。任何一步失败都停止，不增加 step，也不进入
overfit32。
