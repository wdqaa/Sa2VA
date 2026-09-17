# Phase 4B/4C 预注册 overfit 协议

状态：**Phase 4B-0 静态门禁已实现；训练仍未执行**。本协议不授权加载模型或启动训练。

## 共同冻结项与代码边界

- checkpoint：`<MODEL_ROOT>/Sa2VA-InternVL3-2B`，revision
  `15837dcaecc304714a1f0f069e74f47e47521c7f`；训练前需按官方
  `tools/convert_to_pth.py` 产生并验证训练态 `.pth`，转换不属于 Phase 4A。
- manifest 与 P2 registry hash 沿用已冻结值；只允许 `split=train` 和两份版本化 ID
  清单。dataset 初始化若看到 val/test、未知/重复 ID 或 hash 变化立即失败。
- 推荐策略 A：只训练 `text_hidden_fcs`；LLM、vision、MLLM `mlp1`、整个 SAM2 全冻结。
  LoRA **禁用**（r/alpha/dropout 均 N/A）。这是先证明 mask-loss 梯度闭环的最小路径。
- BF16；seed `20260916`；AdamW lr 4e-5、betas (0.9,0.999)、weight decay 0.05、
  grad clip max-norm 1，沿用官方数值。language gradient checkpointing 保持官方开启。
- 输出只能写 `<WORK_DIR>` 或
  `<WORK_DIR>`，不得写 repo root 或 dataset tree。
- P2 双 `[SEG]` / `fix_number=5` 代码门禁已由 labels-aware + strict one-to-one 解决并通过
  真实 tokenizer/collator 静态验证。执行前仍必须完成 full PTH materialization 和
  HF→PTH→训练 subset→HF round-trip 审计；不得改 P2。
- 显存软上限 22,000 MiB；超过立即完整停止该实验，不把 OOM 前的部分状态当结果。

## Phase 4B：1-sample / 1 optimizer step

输入固定为 `configs/phase4_smoke1_ids.json` 的唯一 ID。micro batch=1、gradient
accumulation=1，只允许 1 个 optimizer step，不设 epoch 重复过拟合。

严格按顺序验收：

1. adapter 读到唯一 train 样本，P2 与 assistant target hash/文本吻合；
2. token 检查证明 Prompt labels 全 -100、assistant labels 有效，assistant `[SEG]` 监督位置
   恰好 1；MLLM tile tensor、SAM 1024 tensor、GT `[1,H,W]` shape/dtype 正确；
3. forward 的 `llm_loss/loss_mask/loss_dice/total loss` 全 finite；
4. backward 成功；`text_hidden_fcs` 至少一个参数有非零有限 gradient；
5. LLM、vision、MLLM `mlp1` 和 SAM2 所有参数无 gradient；
6. grad norm finite 且不超过 clip 后阈值；
7. 单次 `optimizer.step` 后至少一个 `text_hidden_fcs` 参数发生数值变化；
8. `.pth`/adapter 保存到临时目录；记录 trainable/frozen 参数清单和 hash；
9. 新进程按官方训练态路径重新加载，missing/unexpected keys 满足预注册 allowlist；
10. 再经 `tools/convert_to_hf.py` 导出临时 HF 目录，新进程用现有 inference backend 对同一
    train 样本输出 contract-valid mask。编辑、val/test 均不运行。

任何一项失败即 Phase 4B 失败，不能增加 step 诊断。配置已经创建并通过静态解析；以下
命令只能在 checkpoint materialization 门禁另行通过后运行：

```bash
CUDA_VISIBLE_DEVICES=<FREE_GPU_INDEX> HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
projects/sa2va/.venv/bin/python tools/train.py \
  projects/chartground_edit/configs/phase4b_smoke1.py \
  --work-dir <WORK_DIR> \
  --launcher none --seed 20260916
```

运行前必须设置 `CHARTGROUND_BASE_MODEL_PATH`（独立 InternVL3-2B base）和
`CHARTGROUND_SA2VA_PTH`（由固定 Sa2VA HF revision 转出的 full BF16 PTH）。单步训练
checkpoint 是 `text_hidden_fcs` state_dict 子集，不是 LoRA adapter 或完整 PTH；完整重载
方案见 `phase4b_alignment_protocol.md`。

## Phase 4C：32-sample memorization

仅使用 `configs/phase4_overfit32_ids.json`；同一 32 条同时作为诊断评测集。这不是泛化
评测，不访问 val/test，也不和 Phase 3C 失败组联动采样或调权。

- strategy A，micro batch=1，gradient accumulation=8，有效 batch=8；BF16；
- AdamW / lr 4e-5 / betas / decay / grad clip 同上；linear warmup 5% optimizer steps，
  cosine decay 到 0；
- 最多 400 optimizer steps；每 step 记录 total/三项 loss、lr、grad norm、显存；每 20
  step 在相同 32 条上确定性诊断；每 100 step 保存临时 checkpoint，最多保留 2 个；
- 训练前和训练后都用相同 P2、相同 mask contract 评测 32 条，并报告 sample macro、
  16-group macro、micro IoU/Dice、empty/disjoint、每样本结果；
- early success：连续 3 次诊断同时达到 16-group Macro IoU ≥0.95 且最差 sample IoU
  ≥0.85；在下一 checkpoint 边界停止。不得事后放宽；
- hard stop：任一 loss/grad norm 非 finite、显存超过 22,000 MiB、split/hash/token/mask
  contract 失败；
- 结束后保存训练态 `.pth`，新进程重载，转 HF，再由现有 backend 重载并复核同一 32 条。

只有 Phase 4B A 完整通过后才能开始 4C。若 A 在 400 step 内不能 memorization，先报告
失败；不得自动切 B/C。B 的下一候选为官方 r128/alpha256/dropout0.05 LoRA +
`text_hidden_fcs`，但官方 `modules_to_save` 会训练全 embedding/lm_head，必须另立协议与
显存门禁。C 再增加官方 SAM2 mask decoder，不能与 B 同时首次引入。
