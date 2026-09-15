# ChartGround-Edit 实验记录

本文件只记录真实运行过的实验。未运行的字段填写“未运行”，未知字段填写“待确认”，不得用预期值代替结果。每次实验复制以下模板，并按时间倒序追加。

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
