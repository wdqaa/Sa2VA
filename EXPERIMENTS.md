# ChartGround-Edit 实验记录

## 2026-09-17 Phase 4C overfit32 projection-only

- 日期：2026-09-17
- 实验 ID：`phase4c-overfit32-text-hidden-fcs-only`
- Git commit：基线 `69ecd57`；Phase 4B-1 已提交，开始时工作区 clean
- 数据：仅 synthetic_v1 train 的冻结 overfit32；16 个 chart/referring 组合各 2 条；每条恰好出现 10 次；未访问 val/test；P2、manifest 和 ID 清单未修改
- 模型/权重：base `/home/dqwang/Model/InternVL3-2B`，revision `899155015275a9b7338c7f4677e19c784e0e5a21`；Sa2VA HF revision `15837dcaecc304714a1f0f069e74f47e47521c7f`；full PTH `/home/dqwang/Model/Sa2VA-InternVL3-2B-train/sa2va_full_bf16.pth`，SHA-256 `5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6`
- 实际配置：seed `20260916`；10 epoch/320 optimizer steps；batch=1、accumulation=1、BF16；AdamW lr `4e-5`、betas `(0.9,0.999)`、weight decay `0.05`；16-step linear warmup + cosine；沿用预注册 protocol 的 optimizer，不做超参数搜索
- 冻结边界：只有 `text_hidden_fcs.{0.weight,0.bias,2.weight,2.bias}` 可训练，共 2,754,304 parameters；LLM、vision、InternVL `mlp1`、SAM2 全冻结；labels-aware + strict one-to-one，未调用 legacy `fix_number=5`
- 训练命令：`CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/chartground_mpl projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_phase4c_train.py --config projects/chartground_edit/configs/phase4c_overfit32.py --base-model /home/dqwang/Model/InternVL3-2B --full-pth /home/dqwang/Model/Sa2VA-InternVL3-2B-train/sa2va_full_bf16.pth --full-pth-sha256 5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6 --output-dir /tmp/chartground_edit_phase4c_overfit32 --base-repo-id OpenGVLab/InternVL3-2B --base-revision 899155015275a9b7338c7f4677e19c784e0e5a21 --sa2va-hf-revision 15837dcaecc304714a1f0f069e74f47e47521c7f`
- loss：step1→step320 为 language `0.252669→0.306876`、mask CE `2.234216→0.083364`、Dice `0.468146→0.137567`、total `2.955030→0.527808`；最低值依次为 `0.243167/0.017682/0.008934/0.286114`；第一→最后 epoch 均值的 mask CE `0.950996→0.294163`、Dice `0.359105→0.191986`
- gradient/runtime：320/320 steps finite；每步 4/4 trainable tensor 非零 gradient，冻结参数 gradient 0；模型构建 39.912 s，训练循环 112.999 s；peak allocated/reserved `7,336.494/9,714.0 MiB`
- checkpoint：`step_32.pth` 11,020,912 bytes、`step_128.pth` 和 `step_320.pth` 各 11,020,920 bytes；每份严格只含四个 FP32 projection tensor 和身份/对齐/step metadata
- 推理命令：`CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_phase4c_inference.py --checkpoint /home/dqwang/Model/Sa2VA-InternVL3-2B --manifest projects/chartground_edit/data/synthetic_v1/annotations.jsonl --selection projects/chartground_edit/configs/phase4_overfit32_ids.json --output-dir /tmp/chartground_edit_phase4c_eval --projection step32=/tmp/chartground_edit_phase4c_overfit32/step_32.pth --projection step128=/tmp/chartground_edit_phase4c_overfit32/step_128.pth --projection step320=/tmp/chartground_edit_phase4c_overfit32/step_320.pth`（其余四个 identity 参数与训练相同）
- 推理结果：baseline/step32/step128/step320 的 16-group Macro IoU 为 `0.164664/0.359869/0.509776/0.569468`，Macro Dice 为 `0.189729/0.447115/0.605529/0.674806`，Micro IoU 为 `0.126421/0.387907/0.579544/0.657489`，empty rate 为 `43.75%/3.125%/0%/0%`，nonempty-disjoint rate 为 `21.875%/3.125%/3.125%/0%`；训练后 baseline 与训练前逐 mask hash 完全一致
- 分组：step320 相对 baseline 的 16/16 组合 mean IoU 均提升，范围 `+0.003507` 到 `+0.936499`；best 为 step320，Macro IoU 增益 `+0.404804`
- OOM/重试：训练无 OOM、无重试。首次训练后推理启动在受限进程中于模型加载前因 CUDA 不可见退出，未创建输出、未运行样本；随后在 GPU 可见环境完成唯一正式 128-call 顺序评测，不涉及额外训练
- 验证：Phase 4A/B/C 专项 44/44、ChartGround-Edit 限定全量 185/185 通过；compileall、配置解析、`git diff --check` 通过；三份 projection checkpoint 经正式 loader 转为 BF16 后逐 tensor 精确一致；320 条训练记录与 128 条 train-only 推理记录独立计数通过
- GPU 释放：训练与推理进程退出后物理 GPU 0 为 2 MiB used、24,252 MiB free、0% utilization
- 产物：完整日志/checkpoint/mask 在 `/tmp/chartground_edit_phase4c_overfit32`、`/tmp/chartground_edit_phase4c_eval`；仓库保留 `results/phase4c_overfit32_{metrics.jsonl,summary.json}`、`assets/phase4c_overfit32_gallery.png` 和结果文档
- 结论：strategy A 在固定 overfit32 上达到预注册“明显成功”标准，证明 projection-only 具有小样本学习/记忆能力；不是泛化结论，未自动转 LoRA/SAM2，未开始完整 train

## 2026-09-17 Phase 4B-1 单样本单步训练 smoke

- 日期：2026-09-17
- 实验 ID：`phase4b1-sa2va-internvl3-2b-one-step`
- Git commit：基线 `68eea6c`（Phase 4B-0 已提交）；开始时工作区 clean
- 数据：仅 synthetic_v1 train smoke1 `cgev1_bar_category_6d51bac154`；未访问 val/test，P2、manifest 和 smoke1 ID 未修改
- base：`OpenGVLab/InternVL3-2B` revision `899155015275a9b7338c7f4677e19c784e0e5a21`，本地 `/home/dqwang/Model/InternVL3-2B`
- Sa2VA 输入：HF revision `15837dcaecc304714a1f0f069e74f47e47521c7f`；官方 `tools/convert_to_pth.py` 产出 `/home/dqwang/Model/Sa2VA-InternVL3-2B-train/sa2va_full_bf16.pth`，4,632,880,109 bytes，SHA-256 `5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6`；1,589 个 tensor、2,316,157,490 parameters，全部 BF16，前缀 `mllm/text_hidden_fcs/grounding_encoder=685/4/900`
- 运行命令：`CUDA_VISIBLE_DEVICES=5 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/chartground_mpl projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_phase4b_smoke1.py --config projects/chartground_edit/configs/phase4b_smoke1.py --base-model /home/dqwang/Model/InternVL3-2B --full-pth /home/dqwang/Model/Sa2VA-InternVL3-2B-train/sa2va_full_bf16.pth --output /tmp/chartground_edit_phase4b_smoke1/iter_1.pth --base-repo-id OpenGVLab/InternVL3-2B --base-revision 899155015275a9b7338c7f4677e19c784e0e5a21 --sa2va-hf-revision 15837dcaecc304714a1f0f069e74f47e47521c7f --full-pth-sha256 5aa030f3203487281abcb57d7dbed72bba618b9f08859e1c020085454b2822e6`
- 模型/冻结门禁：full PTH 对构建模型为 0 missing/0 unexpected；仅 `text_hidden_fcs.{0.weight,0.bias,2.weight,2.bias}` 可训练，共 2,754,304 parameters；optimizer 参数集合完全相同；LLM、vision、InternVL `mlp1`、SAM2 均冻结
- 对齐：supervised assistant `[SEG]` position 1840 与一个 GT mask 严格 1:1；`legacy_fix_number_called=false`
- loss：language `0.2747028172`，mask CE `0.0920886174`，Dice `0.0545147285`，total `0.4213061631`，全部 finite
- gradient/update：clip 前 gradient norm `27.5787943892`；4/4 trainable tensors 有非零有限梯度；冻结参数梯度 0；step 后 changed elements `2,754,304`，最大绝对变化 `4.0072947741e-05`，全可训练元素平均绝对变化 `1.5423816660e-05`
- 时间/显存：模型构建并移至 CUDA `49.9327 s`；forward `0.9984 s`；backward `0.0792 s`；optimizer.step `0.0616 s`；PyTorch peak allocated/reserved `7,211.7173/7,570.0 MiB`
- checkpoint：`/tmp/chartground_edit_phase4b_smoke1/iter_1.pth`，11,020,648 bytes，SHA-256 `c99044844a51b6c109908d957718aaf91a75177f5c9e5d755fb2c75eb8087936`；仅 4 个 FP32 `text_hidden_fcs.*` tensor，共 2,754,304 elements；包含 alignment/base/full-PTH/P2/smoke/step metadata；与 step 后内存值一致，清零后经正式 projection loader 逐 tensor 精确重载一致
- OOM/重试：未 OOM，未重试；严格只有一次 forward、一次 backward、一次 optimizer.step
- 验证：alignment/runtime 专项 21/21 通过；ChartGround-Edit 限定全量 175/175 通过（193 条既有 Pillow、NVML/CPU 或第三方 warning）；projection checkpoint 独立审计、compileall、配置解析和 `git diff --check` 通过
- GPU 释放：进程退出后物理 GPU 5 为 2 MiB used、24,252 MiB free、0% utilization
- 结论：Phase 4B-1 单步训练门禁通过；该结果只证明梯度和保存/重载闭环，不是精度评测。可以另行进入预注册的 Phase 4C 32-sample overfit，但本轮未运行

## 2026-09-17 Phase 4B-0 alignment hard gate and parse-only smoke config

- 日期：2026-09-17
- 实验 ID：`phase4b0-labels-aware-strict-alignment`
- Git commit：基线 `5f20686`（Phase 4A 已提交）；本轮工作区只含待审 Phase 4B-0 改动
- 工作区状态：开始时 clean；对 `projects/sa2va/models/sa2va.py` 做最小、默认兼容的训练侧修改，其余实现位于 ChartGround-Edit
- 数据：只读取 synthetic_v1 train smoke1 `cgev1_bar_category_6d51bac154`；未读取 val/test
- 环境：Python 3.11.16；torch 2.6.0+cu124；transformers 4.57.1；tokenizers 0.22.1；mmengine 0.10.7；xtuner 0.1.23
- 模型/权重：未构建模型，未加载权重，未执行 forward/backward/optimizer，未运行训练或推理
- 静态目的：用真实 tokenizer/collator 复现 P2+official target 的双 `[SEG]`，实现 labels-aware selection、strict 1:1 和可解析的一步配置
- tokenizer/collator 结果：sequence length 1844；`seg_token_idx=151674`；user `[SEG]` position 1823、label=-100；assistant `[SEG]` position 1840、label=151674；旧选择 `[1823,1840]`，新选择 `[1840]`
- mask 结果：GT `torch.uint8 [1,320,480]`，值域 `{0,1}`，前景 8127；MLLM tiles `[7,3,448,448]`，SAM image `[3,1024,1024]`；collator strict 1:1 通过
- legacy 复现：`check_obj_number(fix_number=5)` 前为 2 token/1 mask，先截断为 1/1 后重复，最终 5/5；strict 路径不调用该函数
- 配置：`phase4b_smoke1.py` 已静态解析；train-only smoke1、P2、strategy A、batch=1、accumulation=1、BF16、max_iters=1、seed=20260916、无 val/test/resume、work_dir 在 `/tmp`
- 可训练参数：未实例化验证；配置预注册并在未来 runtime 强制核验 `text_hidden_fcs` only、2,754,304 params
- loss/gradient/显存：未运行
- checkpoint 审计：输入应为固定 revision 的完整 Sa2VA HF 经 `convert_to_pth.py` 产生的约 8.1 GiB full BF16 raw state dict；输出为 `/tmp/chartground_edit_phase4b_smoke1/iter_1.pth` 的 `text_hidden_fcs` subset，预计约 11 MiB raw tensors。转换、保存和重载均未运行
- 结果：alignment 专项 20/20、Phase 4A+4B-0 专项 33/33、ChartGround-Edit 限定全量 174/174 通过（193 warnings，均为既有 Pillow、NVML/CPU 环境或第三方 SWIG warning）；smoke1 真实诊断、compileall、配置解析和 `git diff --check` 均通过
- 结论：双 `[SEG]` 和 `fix_number=5` 代码门禁已解除；Phase 4B-1 仍被唯一 checkpoint materialization/round-trip 门禁阻止（缺独立 InternVL3-2B base，full PTH 未转换，projection subset→HF 未实测）
- 详细协议：`projects/chartground_edit/docs/phase4b_alignment_protocol.md`

## 2026-09-16 Phase 4A training-path audit and train-only subset freeze

- 日期：2026-09-16
- 实验 ID：`phase4a-training-path-audit-overfit-subsets`
- Git commit：基线 `4bed491`（Phase 3C 已提交）；本轮仅新增/修改 ChartGround-Edit 代码、配置、测试、文档和根计划/实验记录
- 工作区状态：开始时 clean；未修改 `projects/sa2va/` 上游核心代码
- 数据版本：冻结 `synthetic_v1` manifest SHA-256 `ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`
- 数据划分与样本数：只读取 train；smoke1=1，overfit32=32；没有对 val/test 运行推理
- 模型与配置：只读审计 `sa2va_in30_2b.py`、`sa2va_finetune.py` 及训练调用链；没有加载/构建模型或 checkpoint
- 可训练参数：未运行。静态推荐 A 为 `text_hidden_fcs` 2,754,304 参数；受控 B/C 估算约 616.4M/620.6M，未额外冻结 InternVL `mlp1` 的官方 C 约 629.3M，须实例化后核验
- 冻结参数：未运行；设计态 A 冻结 MLLM language/vision/mlp1 和整个 SAM2
- 环境：Python 3.11.16；torch 2.6.0+cu124；transformers 4.57.1；peft 0.17.1；mmengine 0.10.7；xtuner 0.1.23；deepspeed 0.18.0
- 随机种子：子集选择不使用随机数；未来训练 seed 预注册为 `20260916`
- 运行命令：schema validate、synthetic_v1 audit、`prepare_phase4_overfit_subsets.py`、独立 subset audit、Phase 4A 专项与 ChartGround-Edit 测试、compileall、diff check；未运行训练命令
- 实验目的：在任何模型训练前确认真实调用链、冻结最小 train-only 诊断集和资源边界
- 预设验收条件：Phase 3C 已提交/初始 clean；只用 train；1/32 数量和 16×2 组合；P2 不含隐藏字段；mask 合法；无模型加载或训练
- 结果：smoke1 为 `cgev1_bar_category_6d51bac154`；overfit32 action 各 8、difficulty easy/medium/hard=11/10/11、distractor 2/3/4=11/10/11。独立 subset audit 通过；Phase 4A 专项 13/13、ChartGround-Edit 全量 154/154 通过（188 条既有 Pillow warning），compileall/diff-check 通过。仓库根 `pytest -q` 仍会收集 `sa2va_eval/projects/ST` 的非本项目脚本，并因缺少 `mmdet`/`projects.ST` 在 collection 阶段失败。官方训练链是 MMEngine/XTuner + InternVL base + Sa2VA `.pth`；官方 LoRA 同时训练/保存 embedding、lm_head，且 `mllm.model.mlp1`、`text_hidden_fcs` 和 SAM2 mask decoder 也可训练
- 问题：冻结 P2 Prompt 自身含 `[SEG]`，official answer 也含 `[SEG]`；当前 `Sa2VAModel.forward` 不按 labels 过滤，并由 `check_obj_number(fix_number=5)` 截断/重复，未经适配会丢掉 assistant token。HF→PTH 转换也尚未实际验证
- 结论：Phase 4A 的审计、子集和纯数据边界完成；没有训练指标、梯度、显存或时间结果。推荐 Phase 4B 策略 A，但当前被 token/mask 对齐硬门禁阻止
- 下一步：审核最小 labels-aware segmentation-position 适配方案；获准后另开 Phase 4B，只执行 1 optimizer step

本文件只记录真实运行过的实验。未运行的字段填写“未运行”，未知字段填写“待确认”，不得用预期值代替结果。每次实验复制以下模板，并按时间倒序追加。

## 2026-09-16 Phase 3C frozen synthetic_v1 test baseline

- 日期：2026-09-16
- 实验 ID：`phase3c-sa2va-internvl3-2b-frozen-test-p2`
- Git commit：基线 `6e7db536c306a9224e70342c8a7f713d8dc526e9`；叠加本轮未提交的冻结协议、runner/聚合/测试、真实标量结果、gallery 和文档
- 工作区状态（clean / dirty，附相关 diff 说明）：开始门禁时 clean 且 Phase 3B 已提交；正式运行时 dirty，仅含本阶段已通过测试的 ChartGround-Edit 实现与冻结协议，无 Sa2VA 上游源码修改
- 数据版本：`synthetic-v1.0.0`，schema `chartground-edit-v1`；manifest SHA-256 `ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`；test sample-ID list SHA-256 `3061387f0012b13c2fd998d81819df0e7648e8e47ae9ad86ff06490ecfe2e5c4`
- 数据划分与样本数：仅 test 64 条；16 个 chart/referring 组合各 4 条且四种 action 各 1 条；每条只运行 P2 一次；train/val 未推理
- 模型与配置：`ByteDance/Sa2VA-InternVL3-2B`；P2 `target_only_zh`；BF16；单卡；`torch.inference_mode()`；`use_flash_attn=True`；`local_files_only=True`；无量化、offload、device map 或重试
- 权重来源与版本：用户预先下载的本地 checkpoint `/home/dqwang/Model/Sa2VA-InternVL3-2B`；revision `15837dcaecc304714a1f0f069e74f47e47521c7f`；本轮未下载
- 可训练参数：未运行训练
- 冻结参数：未运行训练
- 硬件与软件环境：物理 GPU 1 映射到逻辑 `cuda:0`，RTX 3090；正式运行前 nvidia-smi 空闲 24,252 MiB、0% utilization；模型加载前 PyTorch 空闲 23,991.8125 MiB
- 随机种子：模型生成沿用上游确定性配置；16-group bootstrap seed `20260916`，10,000 次
- 运行命令：`CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_frozen_test_v1.py --checkpoint /home/dqwang/Model/Sa2VA-InternVL3-2B --manifest projects/chartground_edit/data/synthetic_v1/annotations.jsonl --split test --prompt-variant target_only_zh --device cuda:0 --dtype bfloat16 --expected-samples 64 --output-dir /tmp/chartground_edit_phase3c_frozen_test`
- 实验目的：在 Prompt 选择完成后，以冻结 P2 对 balanced synthetic_v1 test 做唯一一次正式 zero-shot baseline，并验证 predicted-mask 编辑闭环
- 预设验收条件：运行前协议冻结；64 条/64 调用；模型加载一次；只用图像与 `referring_expression` 生成的 registry P2；首个预测 mask；失败/空 mask 以 0 纳入；固定 group bootstrap；禁止 Prompt 搜索、选择性重跑和 GT mask 编辑
- 结果：64/64 backend 调用和记录完成，execution success、mask contract valid、`[SEG]` 均为 64/64；nonempty 37/64、empty 27/64、overlap 30/64、nonempty-disjoint 7/64。16-group Macro IoU/Dice 0.198033/0.242366，sample Macro 相同，Micro IoU/Dice 0.225301/0.367748，median IoU/Dice 均为 0。最弱组按确定性排序为 `bar/appearance`，mean IoU=0；同为 0 的还有 `line/appearance` 和 `scatter/appearance`
- Bootstrap：16 个 chart/referring group、10,000 次；Macro IoU 95% CI [0.101680, 0.316589]，Macro Dice 95% CI [0.139124, 0.360458]；仅描述 test 估计区间，不用于修改 Prompt
- Validation/test：P2 test 相对 val 的 Macro IoU/Dice 为 +0.015834/+0.026145，empty/nonempty-disjoint rate 为 -0.015625/-0.046875；但 `bar/category` IoU -0.227059、`line/trend` -0.082575，`bar/trend` +0.236659，组间变化不一致，不据此返回 val 或修改 Prompt
- 编辑闭环：37 个非空预测全部以保存的 predicted mask 和样本真实 action/parameters 调用现有编辑器且成功；27 个空预测标记 `edit_skipped_empty`；0 次 GT mask 替代。编辑成功只代表链路执行成功
- 耗时与显存：模型加载 7,931.789 ms；mean/median predict latency 609.330/580.492 ms；PyTorch peak allocated 5,005.364 MiB；正式进程约 55 秒
- 协议完整性：Prompt registry SHA-256 `dc822a33b84b1cdfb72f84bd5288f0ebb37626980496107c5e30d4c4c26217c0`；P2 template SHA-256 `37a785d086a80fef21fd69014670b3892acad5c379722cb79658fb837a923806`；protocol 前后 SHA-256 均为 `4d97adbf7fb3c9e958a58028f7ab3986cda71071fa4a48c9a0c2d66a9bd88817`
- 验证命令与结果：正式 runner 内置及独立二次复核均为 64 条记录、64 个唯一 test ID、64 个二值原尺寸 mask、64 次 IoU/Dice 重算、37 个 predicted-mask 编辑输入，0 error；Phase 3C 专项测试最终 13/13 通过；ChartGround-Edit 全量最终 141/141 通过（188 条既有 Pillow warning）；`compileall` 和 `git diff --check` 通过；进程退出后物理 GPU 1 为 2 MiB used、24,252 MiB free、0% utilization，显存已释放
- 产物路径：完整临时输出 `/tmp/chartground_edit_phase3c_frozen_test`；仓库标量 `projects/chartground_edit/results/phase3c_frozen_test_metrics.jsonl`、summary `phase3c_frozen_test_summary.json`、真实 gallery `projects/chartground_edit/assets/phase3c_frozen_test_gallery.png`、协议/报告 `projects/chartground_edit/docs/phase3c_frozen_test_*`
- 问题：appearance 指代 16 条中 14 条为空，line/scatter appearance 均 4/4 空；难度与 distractor count 在 v1 中一一对应，不能分离二者影响；大目标 bar 组会抬高 micro 指标
- 结论：Phase 3C frozen test baseline 和 predicted-mask 编辑闭环完成；test 结果没有改变 P2，也没有重跑 val、访问 train 或训练/微调
- 下一步：具备进入 Phase 4 的工程条件，但必须先预注册并仅执行 1 样本 smoke 与 32 样本过拟合门槛；未经用户确认不运行完整训练

## 2026-09-16 Phase 3B balanced synthetic_v1 val Prompt benchmark

- 日期：2026-09-16
- 实验 ID：`phase3b-sa2va-internvl3-2b-balanced-val-prompt-benchmark`
- Git commit：基线 `b022e4d10c8f6fc922fd3d18790ef999b85f36a2`；叠加本轮未提交的冻结协议、benchmark 模块/CLI/测试、真实标量结果、gallery 和文档
- 工作区状态（clean / dirty，附相关 diff 说明）：开始时 clean 且 Phase 3A 已提交；正式运行时 dirty，仅含本阶段已通过单元测试的实现和冻结协议，无 Sa2VA 上游源码修改
- 数据版本：`synthetic-v1.0.0`，schema `chartground-edit-v1`；manifest SHA-256 `ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`
- 数据划分与样本数：仅 val 64 条；16 个 chart/referring 组合各 4 条且四种 action 各 1 条；每条 P0/P1/P2 各一次，共 192 次；train/test 未推理
- 模型与配置：`ByteDance/Sa2VA-InternVL3-2B`；BF16；单卡；`use_flash_attn=True`；`local_files_only=True`；无量化、offload、device map 或重试
- 权重来源与版本：用户预先下载的本地 checkpoint；revision `15837dcaecc304714a1f0f069e74f47e47521c7f`；本轮未下载
- 可训练参数：未运行训练
- 冻结参数：未运行训练
- 硬件与软件环境：物理 GPU 0 映射到逻辑 `cuda:0`，RTX 3090；加载前 PyTorch 空闲 23,991.8125 MiB；正式进程前 nvidia-smi 空闲约 24,252 MiB、0% utilization
- 随机种子：模型生成沿用上游确定性配置；paired bootstrap seed `20260916`，10,000 次
- 运行命令：`CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_prompt_benchmark_v1.py --checkpoint /home/dqwang/Model/Sa2VA-InternVL3-2B --manifest projects/chartground_edit/data/synthetic_v1/annotations.jsonl --split val --device cuda:0 --dtype bfloat16 --expected-samples 64 --output-dir /tmp/chartground_edit_phase3b_balanced_val`
- 实验目的：在不访问 test 的条件下，以预注册 primary metric 从 P0/P1/P2 选择一个全局 Prompt，供后续 frozen-test baseline 使用
- 预设验收条件：protocol 在首次推理前冻结；64×3 单次顺序调用；模型加载一次；失败/空 mask 以 0 纳入；16-group Macro IoU 选择；固定 bootstrap；不按类型路由或根据结果改 Prompt
- 结果：192/192 backend 调用完成，模型加载 1 次，无全局错误。P0/P1/P2 的 execution success 均为 64/64，mask contract valid 均为 64/64，`[SEG]` rate 均为 100%；nonempty prediction 为 35/64、36/64、36/64，empty prediction 为 29/64、28/64、28/64。历史 `inference_success` 字段保留原值并标记 deprecated，不能解释为执行成功。16-group Macro IoU 为 0.164503/0.179093/0.182199，Macro Dice 为 0.192781/0.210226/0.216221，Micro IoU 为 0.150734/0.202185/0.192302。按预注册规则选择 P2 `target_only_zh`。nonempty-disjoint rate 为 12.5%/15.625%/15.625%。P2−P0 group Macro IoU +0.017696，95% CI [-0.000062, 0.039690]；P2−P1 +0.003106，95% CI [-0.006147, 0.015220]；均包含 0，只能称 selected on validation。落盘 192 个 mask 的尺寸/二值性和标量指标独立重算全部通过；protocol 前后 SHA-256 均为 `5de44dcf8e21845aaf3d8c5355faedd62b80899cc54b698e7fe2731aa839ef1d`。
- 耗时与显存：模型加载 7,963.318 ms；P0/P1/P2 mean predict latency 578.854/575.031/568.848 ms；PyTorch peak allocated 5,006.784 MiB；正式进程约 130 秒（含加载、192 次调用、保存与汇总）
- metric-semantics 收尾：未加载模型、未运行推理，仅由现有 192 条标量和保存 mask 离线重聚合；命令为 `projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/reaggregate_phase3b_metrics.py --metrics projects/chartground_edit/results/phase3b_balanced_val_metrics.jsonl --summary projects/chartground_edit/results/phase3b_balanced_val_summary.json --protocol projects/chartground_edit/docs/phase3b_benchmark_protocol.md`。protocol 未修改，IoU/Dice/bootstrap/P2 选择均未改变
- 验证命令与结果：`projects/sa2va/.venv/bin/python -m pytest projects/chartground_edit/tests/test_prompt_benchmark_v1.py -q` 为 19/19 通过；`projects/sa2va/.venv/bin/python -m pytest projects/chartground_edit/tests -q` 为 128/128 通过，188 条既有 Pillow `mode` 弃用 warning；`compileall`、独立落盘 mask/指标复核和 `git diff --check` 均通过
- 产物路径：完整临时输出 `/tmp/chartground_edit_phase3b_balanced_val`；仓库标量 `projects/chartground_edit/results/phase3b_balanced_val_metrics.jsonl` 和 `phase3b_balanced_val_summary.json`；真实 gallery `projects/chartground_edit/assets/phase3b_balanced_val_prompt_benchmark.png`；协议/报告见 `projects/chartground_edit/docs/phase3b_*`
- 问题：虽然 192/192 都生成 `[SEG]` 且返回一个 `(1,320,480)` bool mask，85 个 mask 为空；appearance、line category、bar trend 等组接近或等于 0。difficulty 与 distractor count 在 v1 中一一对应，无法分离影响。P2 的 bootstrap CI 包含 0，没有明确统计优势。
- 结论：P2 是 synthetic_v1 val 上按冻结规则得到的 operational selection，不是 test 或真实图表结论；execution、mask contract、`[SEG]`、nonempty 和 overlap 是不同语义，均不等同于 segmentation accuracy
- 下一步：等待审核；若进入独立 Phase 3C，只能使用冻结的 P2 对 synthetic_v1 test 运行一次，不再根据 test 改 Prompt；本轮不训练或微调

## 2026-09-16 Phase 3A balanced synthetic_v1 生成与审计

- 日期：2026-09-16
- 实验 ID：`phase3a-synthetic-v1-generation-audit`
- Git commit：基线 `f3783cf`；叠加本轮尚未提交的 v1 schema/Reader/生成器/审计器、测试、gallery 和文档
- 工作区状态（clean / dirty，附相关 diff 说明）：dirty；仅 ChartGround-Edit Phase 3A 与根 PLAN/EXPERIMENTS，无 Sa2VA 上游源码修改
- 数据版本：`synthetic-v1.0.0`，schema `chartground-edit-v1`，mask semantics `chartground-edit-mask-v0`
- 数据划分与样本数：train 192 / val 64 / test 64，共 320；16 个 chart/referring 组合各为 12/4/4；每种 action 共 80
- 模型与配置：未运行
- 权重来源与版本：未运行
- 可训练参数：未运行
- 冻结参数：未运行
- 硬件与软件环境：Python 3.11.16；NumPy 2.3.4；Pillow 11.3.0；pytest 9.1.1；CPU 本地生成，不需要 GPU
- 随机种子：`20260916`；逐样本 seed 由固定输入字符串的 SHA-256 确定，manifest 内无重复
- 运行命令：`projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/generate_synthetic_v1.py --output-dir projects/chartground_edit/data/synthetic_v1 --seed 20260916 --clean --gallery-output projects/chartground_edit/assets/synthetic_v1_gallery.png`；独立 `validate_synthetic_v1.py --expected-count 320`；独立 `audit_synthetic_v1.py --expected-count 320 --near-duplicate-threshold 0.01`；临时目录同 seed 再生成并逐文件哈希比较
- 实验目的：构建 split、组合、action、难度明确且可复现的 320 条合成数据，严格检查 mask 语义、重复与跨 split 泄漏
- 预设验收条件：320/192/64/64；每组合 12/4/4；action 3/1/1；v0 兼容；所有 mask 二值非空并符合图表语义；身份/文件/内容无精确重复；同 seed 字节一致；独立审计硬失败为 0
- 结果：生成与独立 schema 校验 320/320 通过；独立审计硬失败 0。easy/medium/hard 为 106/107/107；空 mask、全一 mask、语义错误、ID/path/seed/scene/content 重复、跨 split family 泄漏、image/mask 精确重复均为 0。前景像素 min/median/mean/max 为 592/2419/5667.5625/18744。近重复指纹阈值 0.01 保守报告 97 对跨 split 人工复核候选，不自动删除。Phase 3A 专项测试首次 18/18 通过、16 条 warning；最终全量 ChartGround-Edit 回归 109/109 通过、188 条既有 Pillow `mode` 弃用 warning；`compileall` 通过。同 seed 双目录 manifest SHA-256 均为 `ebad55fd98356204e572ffe6607a16a34c9dde8a916a9977a7f08bc4aed2ba82`，640 个 image/mask 文件集合及逐文件哈希完全一致，0 个 mismatch。
- 产物路径：本地忽略数据 `projects/chartground_edit/data/synthetic_v1/`；版本化 gallery `projects/chartground_edit/assets/synthetic_v1_gallery.png`；schema、data card 和审计见 `projects/chartground_edit/docs/`
- 问题：共享 axes/layout 导致轻量指纹产生较多候选；合成字体、布局和语言模板仍不代表真实出版图表；这些限制不属于模型指标
- 结论：Phase 3A 数据硬约束通过，具备进入独立 Phase 3B balanced-val Prompt benchmark 的数据条件；没有运行任何模型、val/test 推理或训练
- 下一步：等待审核；若进入 Phase 3B，只在 64 条 balanced val 上执行预注册 Prompt benchmark，继续冻结 test

## 2026-09-16 Phase 2C val-only Prompt 诊断

- 日期：2026-09-16
- 实验 ID：`phase2c-sa2va-internvl3-2b-val-prompt-diagnostic`
- Git commit：基线 `8463d0a`；叠加本轮尚未提交的 instruction/Prompt/诊断模块、CLI、测试、真实 gallery 和指定文档
- 工作区状态（clean / dirty，附相关 diff 说明）：dirty；仅 ChartGround-Edit inference 边界、Phase 2C 脚本/测试/资产/文档及 PLAN/README/本记录，无 Sa2VA 上游源码修改
- 数据版本：`synthetic-v0`，schema `chartground-edit-v0`；manifest 未修改
- 数据划分与样本数：仅 val 4 条，均为 category/highlight；每条运行 `full_instruction`、`target_only_en`、`target_only_zh`，共 12 次；未运行 train/test
- 模型与配置：`ByteDance/Sa2VA-InternVL3-2B`；BF16；单卡；`use_flash_attn=True`；`local_files_only=True`；无量化、offload、device map 或输入分辨率覆盖
- 权重来源与版本：用户预先下载的本地 checkpoint；revision `15837dcaecc304714a1f0f069e74f47e47521c7f`；本轮未下载
- 可训练参数：未运行训练
- 冻结参数：未运行训练
- 硬件与软件环境：物理 GPU 1，进程内逻辑 `cuda:0`，RTX 3090；运行前 `nvidia-smi` 约 24,252 MiB 空闲、0% utilization；Python 3.11.16；torch 2.6.0+cu124；transformers 4.57.1；flash-attn 2.7.3
- 随机种子：模型生成使用上游 `do_sample=False`；未另外设置随机种子
- 运行命令：`CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_prompt_diagnostic.py --checkpoint /home/dqwang/Model/Sa2VA-InternVL3-2B --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl --split val --prompt-variants full_instruction,target_only_en,target_only_zh --device cuda:0 --dtype bfloat16 --expected-count 4 --output-dir /tmp/chartground_edit_phase2c_prompt_diagnostic`
- 实验目的：仅在 val 上比较预注册 P0/P1/P2，诊断完整编辑动作文本是否干扰目标定位，并保持 test 冻结
- 预设验收条件：指代表达只从原 instruction 连续子串提取；同一模型加载一次；12 个组合各推理一次；首个 mask 为主结果；不翻译、不用隐藏属性、不用 GT 选择 Prompt/mask、不重试
- 结果：12/12 inference success，12/12 输出 `[SEG]`，无空 mask；所有文本均为 `Sure, [SEG].<|im_end|>`。P0/P1/P2 macro IoU 为 0.392236/0.340420/0.351172，macro Dice 为 0.437576/0.391375/0.428990，micro IoU 为 0.666654/0.558417/0.609943，micro Dice 为 0.799991/0.716647/0.757720；三个 variant 均为 25% nonempty-disjoint、75% overlap。P1 没有提高任何样本 IoU；P2 改善 scatter/confidence-band，但明显恶化 bar。
- 耗时与显存：模型加载一次（计数 1），9,460.079 ms；P0/P1/P2 平均推理时间为 757.661/569.087/546.161 ms；PyTorch peak allocated 5,004.661 MiB。P0 第一条含首次生成 warm-up，不将均值差异解释为稳定速度优势。
- 产物路径：完整临时输出 `/tmp/chartground_edit_phase2c_prompt_diagnostic`；版本化 gallery `projects/chartground_edit/assets/phase2c_prompt_diagnostic.png`；结果文档 `projects/chartground_edit/docs/phase2_prompt_diagnostic.md`；split 审计 `projects/chartground_edit/docs/synthetic_v0_split_audit.md`
- 问题：val 只有 4 条、全部 category/highlight；train 缺 category，val/test 缺 appearance/legend/trend，edit action 与 split 完全混杂；无法检验其他动作参数或进行显著性分析
- 结论：删除编辑动作没有显示一致改善；中文 wrapper 不影响 `[SEG]` 协议成功，但会改变 mask 几何。按当前有限 val 聚合，P0 仅作为 balanced synthetic_v1 val 的候选，不是最终 Prompt；100% inference success 不等于 100% segmentation accuracy
- 下一步：先实现并审计 balanced synthetic_v1（建议 320 条、192/64/64），随后只在其 val 上重新比较预注册候选；不重新运行 test，不开始微调

## 2026-09-15 Phase 2B-2 synthetic_v0 test 零样本基线

- 日期：2026-09-15
- 实验 ID：`phase2b2-sa2va-internvl3-2b-zeroshot-test`
- Git commit：基线 `272637b`；叠加本轮尚未提交的 ChartGround-Edit 批量脚本、测试、真实 gallery 和指定文档
- 工作区状态（clean / dirty，附相关 diff 说明）：dirty；仅 `projects/chartground_edit/` 推理批处理、测试、资产/文档及根 PLAN/EXPERIMENTS，无 Sa2VA 上游源码修改
- 数据版本：`synthetic-v0`，schema `chartground-edit-v0`
- 数据划分与样本数：test，4 条：`cge_line_category_01`、`cge_bar_category_01`、`cge_scatter_category_01`、`cge_confidence_band_category_01`；未运行 train/val
- 模型与配置：`ByteDance/Sa2VA-InternVL3-2B`；BF16；单卡；`use_flash_attn=True`；`local_files_only=True`；固定 Phase 2B-1 Prompt；无量化、offload、device map 或输入分辨率覆盖
- 权重来源与版本：用户预先下载的本地 checkpoint；revision `15837dcaecc304714a1f0f069e74f47e47521c7f`；本轮未下载
- 可训练参数：未运行训练
- 冻结参数：未运行训练
- 硬件与软件环境：物理 GPU 2，进程内逻辑 `cuda:0`，RTX 3090；模型加载前 PyTorch 空闲 23,991.8125 MiB；Python 3.11.16；torch 2.6.0+cu124；transformers 4.57.1；flash-attn 2.7.3
- 随机种子：模型生成使用上游 `do_sample=False`；未另外设置随机种子
- 运行命令：`CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_sa2va_split.py --checkpoint /home/dqwang/Model/Sa2VA-InternVL3-2B --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl --split test --device cuda:0 --dtype bfloat16 --expected-count 4 --continue-on-sample-error --output-dir /tmp/chartground_edit_phase2b2_test`
- 实验目的：在固定协议下验证模型只加载一次的 test split 批量推理、失败记录、全样本聚合指标、predicted-mask 编辑和可视化链路
- 预设验收条件：严格选择 4 条 test；一个 backend、顺序单次推理；第一个 mask 为主结果；失败也计入汇总；不使用 GT 选 mask、驱动编辑或调 Prompt
- 结果：4/4 inference success；每条均为 1 个 `(1,320,480)` bool raw mask，非空且编辑成功。line IoU/Dice 0.015209/0.029963；bar 0/0、nonempty disjoint；scatter 0.127458/0.226098；confidence_band 0/0、nonempty disjoint。macro mean IoU/Dice 0.035667/0.064015，median 0.007605/0.014981；micro IoU/Dice 0.006158/0.012241；empty rate 0%，nonempty-disjoint rate 50%，overlap rate 50%。文本前三条为 `Sure, [SEG].<|im_end|>`，confidence band 为 `Sure, it is [SEG].<|im_end|>`。
- 耗时与显存：模型只加载一次（计数 1），9,586.266 ms；4 条 `predict_forward` 合计 3,130.961 ms、平均 782.740 ms；PyTorch peak allocated 5,004.330 MiB，覆盖加载和全部推理
- 产物路径：完整临时输出 `/tmp/chartground_edit_phase2b2_test`；版本化真实总览 `projects/chartground_edit/assets/sa2va_2b_zeroshot_test.png`；详细记录 `projects/chartground_edit/docs/phase2_zeroshot_results.md`
- 问题：首次受限沙箱启动时 GPU 设备不可见，模型加载前即失败且没有调用 `predict_forward`；随后在 GPU 可访问上下文以相同固定配置完成唯一正式运行。零样本结果显示模型常返回合法非空 mask，但会选择错误序列或元素。
- 结论：批量工程调用链真实跑通，但只有 4 条合成样本且整体重叠很低，不能视为最终统计结果；低性能不等于工程链路失败
- 下一步：若获审核通过，应只在 val split 做预先声明的 Prompt 诊断，再扩充真实图表与 referring-type 数据；最终 Prompt 不得根据本轮 test 结果修改，不直接开始微调

## 2026-09-15 Phase 2B-1 单样本零样本 smoke

- 日期：2026-09-15
- 实验 ID：`phase2b1-sa2va-internvl3-2b-single-smoke`
- Git commit：基线 `e827280`；叠加尚未提交的 Phase 2B-0 文档和本轮 ChartGround-Edit 实现
- 工作区状态（clean / dirty，附相关 diff 说明）：dirty；仅 ChartGround-Edit inference、CLI、测试、真实 gallery 资产及指定文档，无 Sa2VA 上游源码修改
- 数据版本：`synthetic-v0`，schema `chartground-edit-v0`
- 数据划分与样本数：test；仅 `cge_bar_category_01`，1 条；没有运行第二个样本或完整 test split
- 模型与配置：`ByteDance/Sa2VA-InternVL3-2B`；BF16；单卡；`use_flash_attn=True`；`local_files_only=True`；无量化、offload、device map 或输入降采样覆盖
- 权重来源与版本：用户预先下载的本地 checkpoint；revision `15837dcaecc304714a1f0f069e74f47e47521c7f`；本轮未下载
- 可训练参数：未运行训练
- 冻结参数：未运行训练
- 硬件与软件环境：物理 GPU 2，进程内逻辑 `cuda:0`，RTX 3090；加载前 PyTorch 报告空闲 23,991.8125 MiB；torch 2.6.0+cu124；transformers 4.57.1；flash-attn 2.7.3
- 随机种子：模型生成使用上游 `do_sample=False`；未另外设置随机种子
- 运行命令：`CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/run_sa2va_baseline.py --checkpoint /home/dqwang/Model/Sa2VA-InternVL3-2B --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl --sample-id cge_bar_category_01 --device cuda:0 --dtype bfloat16 --edit-action recolor --edit-color '#E63946' --output-dir /tmp/chartground_edit_phase2b1_smoke`
- 实验目的：验证固定 Prompt → 官方 `predict_forward` → 原图尺寸二值 mask → 指标 → predicted-mask 编辑 → 可视化的单样本调用链
- 预设验收条件：只跑固定柱状图样本一次；模型和 tokenizer 同一本地 checkpoint；第一个预测 mask 为主结果；不使用 GT 选 mask 或驱动编辑；失败也保存结构化结果
- 结果：成功调用模型；文本输出 `Sure, [SEG].<|im_end|>`；`prediction_masks` 为 list，含 1 个 NumPy bool mask，原始 shape `(1, 320, 480)`、值域 false/true；后处理 shape `(320, 480)`，无需 resize，前景 11,738 像素；GT 前景 6,960 像素；empty prediction=false；IoU=0.0；Dice=0.0；recolor 编辑成功。预测非空但落在错误柱体，不能解释为分割成功。最终 ChartGround-Edit 回归 65/65 通过，`compileall` 通过。
- 耗时与显存：模型及 tokenizer 加载 31,683.579 ms；`predict_forward` 1,223.360 ms；`torch.cuda.max_memory_allocated` 为 4,917.639 MiB。峰值从加载前 reset 后统计，覆盖模型加载和推理；未同步采集 nvidia-smi 峰值，两者不可比较。
- 产物路径：完整临时输出 `/tmp/chartground_edit_phase2b1_smoke`；版本化真实对比图 `projects/chartground_edit/assets/sa2va_2b_smoke.png`
- 问题：单样本零样本定位错误，IoU/Dice 为 0；这只验证调用链，不足以判断 2B 在 test split 上的整体质量。上游还打印 `torch_dtype` 弃用和 timm import FutureWarning，本轮未修改上游接口。
- 结论：Phase 2B-1 单样本端到端调用链已真实跑通；完整 Phase 2 baseline、test split 汇总和任何训练仍未完成
- 下一步：先审核该失败案例及 adapter 产物；若继续，应在独立阶段运行全部 4 条 test 样本并保留所有失败，而不是根据 GT 调 Prompt

## 2026-09-15 Phase 2B-0 环境与 checkpoint 校验

- 日期：2026-09-15
- 实验 ID：`phase2b0-environment-checkpoint-audit`
- Git commit：基线 `e827280`；本轮开始时工作区 clean
- 工作区状态（clean / dirty，附相关 diff 说明）：dirty；仅 Phase 2B-0 环境文档、README、PLAN 和本记录；`projects/sa2va/.venv` 被 gitignore 排除
- 数据版本：未使用数据
- 数据划分与样本数：未运行
- 模型与配置：目标为 `ByteDance/Sa2VA-InternVL3-2B`、BF16；未创建或加载模型
- 权重来源与版本：用户预先下载的本地 checkpoint；revision `15837dcaecc304714a1f0f069e74f47e47521c7f`，本轮未下载
- 可训练参数：未运行
- 冻结参数：未运行
- 硬件与软件环境：Python 3.11.16；uv 0.12.14；torch 2.6.0+cu124；torchvision 0.21.0+cu124；transformers 4.57.1；peft 0.17.1；flash-attn 2.7.3；7 张 RTX 3090 的轻量 CUDA 通信通过
- 随机种子：未使用
- 运行命令：`curl -LsSf https://astral.sh/uv/install.sh | sh`；`bash setup_env.sh sa2va latest`；`/home/dqwang/.local/bin/uv pip install --python projects/sa2va/.venv/bin/python pytest==9.1.1`；轻量 import/CUDA 检查；checkpoint JSON/index/safetensors header 只读检查；`projects/sa2va/.venv/bin/python -m pytest projects/chartground_edit/tests -q`；`projects/sa2va/.venv/bin/python -m compileall -q projects/chartground_edit/chartground_edit projects/chartground_edit/scripts projects/chartground_edit/tests`
- 实验目的：建立独立、可复现的 InternVL3 推理环境，并在不加载权重的前提下验证本地 checkpoint 结构
- 预设验收条件：锁定依赖可导入；PyTorch CUDA 通信成功；checkpoint 配置、tokenizer、index、分片和 revision metadata 一致；ChartGround-Edit 回归通过；不运行模型
- 结果：关键模块全部导入；CUDA 可用且识别 7 张 GPU；40/40 测试通过；compileall 通过；26 个顶层 checkpoint artifact 与两个 safetensors 文件头/1,589 个 index 映射一致；24 个本地 metadata revision 均匹配预期
- 产物路径：`projects/chartground_edit/docs/phase2_environment.md`
- 问题：`uv pip check` 报告 mmengine/OpenCV 包名和 xtuner/Python 约束两项上游 metadata 不一致；checkpoint cache 留有 24 个零字节 lock；当前 GPU 显存占用过高，不适合启动模型
- 结论：环境、FlashAttention、CUDA 通信和 checkpoint 结构已验证；模型加载、推理、显存峰值、IoU、Dice 和推理时间均未运行
- 下一步：等待 GPU 释放和用户审核后，另行执行 Phase 2B-1 单样本 smoke test

## 2026-09-15 Phase 1B GT mask 编辑后端

- 日期：2026-09-15
- 实验 ID：`phase1b-editing-v0`
- Git commit：基线 `99fd941`；本轮开始时工作区 clean
- 工作区状态（clean / dirty，附相关 diff 说明）：dirty；仅 ChartGround-Edit Phase 1B 编辑模块、CLI、测试、gallery 及指定文档
- 数据版本：`synthetic-v0`，编辑协议 `editing-v0`
- 数据划分与样本数：使用 Phase 1A 的 32 条本地合成样本；gallery 选 4 条代表样本
- 模型与配置：未运行
- 权重来源与版本：未运行
- 可训练参数：未运行
- 冻结参数：未运行
- 硬件与软件环境：Python 3.11.16；NumPy 2.4.6；Pillow 12.3.0；未使用 GPU
- 随机种子：编辑器无随机过程；输入数据 seed `20260915`
- 运行命令：`python projects/chartground_edit/scripts/edit_with_mask.py --image projects/chartground_edit/data/synthetic_v0/images/cge_bar_category_01.png --mask projects/chartground_edit/data/synthetic_v0/masks/cge_bar_category_01.png --action recolor --output /tmp/chartground_edit_recolor.png --color '#E63946'`；`python projects/chartground_edit/scripts/generate_editing_gallery.py --manifest projects/chartground_edit/data/synthetic_v0/annotations.jsonl --output projects/chartground_edit/assets/editing_v0_gallery.png`；`python -m pytest projects/chartground_edit/tests/test_editing.py -q`；`python -m pytest projects/chartground_edit/tests -q`
- 实验目的：验证四种 mask 驱动编辑语义与像素边界，并建立模型无关的后端
- 预设验收条件：尺寸/值域/空 mask 错误明确；recolor/remove 的 mask 外字节级不变；extract alpha 正确；highlight 强度有效；四动作覆盖四图表；CLI 成功及缺失文件报错清晰；结果确定性
- 结果：Phase 1B 测试首次运行 31/31 通过；Phase 1A+1B 全量回归首次运行 40/40 通过；gallery 4 行生成成功并已人工打开检查
- 产物路径：`projects/chartground_edit/assets/editing_v0_gallery.png`
- 问题：remove 仅填色，会同时抹除 mask 区域内重叠的曲线/网格且不恢复；邻域中位数不适合复杂纹理背景；仅验证 GT mask
- 结论：Phase 1B 的独立编辑后端满足本轮验收条件
- 下一步：停止并等待用户审核；不提前接入 Sa2VA 或运行模型

## 2026-09-15 Phase 1A 数据与可视化闭环

- 日期：2026-09-15
- 实验 ID：`phase1a-synthetic-v0`
- Git commit：基线 `e2d56a8`；本轮存在 Phase 1A 新增/修改文件
- 工作区状态（clean / dirty，附相关 diff 说明）：dirty；仅 ChartGround-Edit 子项目代码/文档、根 `PLAN.md` 与本记录
- 数据版本：`synthetic-v0`，schema `chartground-edit-v0`
- 数据划分与样本数：train 24 / val 4 / test 4，共 32；16 个 chart/referring 组合各 2 条
- 模型与配置：未运行
- 权重来源与版本：未运行
- 可训练参数：未运行
- 冻结参数：未运行
- 硬件与软件环境：Python 3.11.16；NumPy/Pillow/matplotlib/pytest 均可导入；未使用 GPU
- 随机种子：`20260915`
- 运行命令：`python -m pytest projects/chartground_edit/tests -q`；`PYTHONPATH=projects/chartground_edit python projects/chartground_edit/scripts/generate_synthetic_v0.py --output-dir projects/chartground_edit/data/synthetic_v0 --seed 20260915 --clean`
- 实验目的：验证 v0 JSONL、二值 PNG mask、Reader、确定性生成、同步几何变换和 32 样本可视化闭环
- 预设验收条件：32/32 schema/路径/尺寸/值域/非空通过；16 个组合平衡；同 seed annotation/mask 一致；Reader 完整读取；变换后对齐且 mask 保持二值
- 结果：首次 pytest 运行 9/9 通过；正式生成命令报告 `generated_and_validated=32`；gallery 已人工打开抽查，未见可见错位
- 产物路径：`projects/chartground_edit/data/synthetic_v0/annotations.jsonl`、`projects/chartground_edit/data/synthetic_v0/gallery.png`、同目录 `images/`、`masks/`、`visualizations/`
- 问题：合成图刻意简单，不代表真实科学图表分布；模型推理与训练未运行
- 结论：Phase 1A 数据协议和可视化闭环满足本轮验收条件
- 下一步：停止并等待用户审核；不提前进入 Sa2VA 接入或训练

## 实验模板

- 日期：
- 实验 ID：
- Git commit：
- 工作区状态（clean / dirty，附相关 diff 说明）：
- 数据版本：
- 数据划分与样本数：
- 模型与配置：
- 权重来源与版本：
- 可训练参数：
- 冻结参数：
- 硬件与软件环境：
- 随机种子：
- 运行命令：
- 实验目的：
- 预设验收条件：
- 结果：
- 产物路径：
- 问题：
- 结论：
- 下一步：
