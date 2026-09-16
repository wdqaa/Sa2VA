# ChartGround-Edit 实验记录

本文件只记录真实运行过的实验。未运行的字段填写“未运行”，未知字段填写“待确认”，不得用预期值代替结果。每次实验复制以下模板，并按时间倒序追加。

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
