# ChartGround-Edit 项目说明

## 项目背景

科学图表把类别、数值、趋势和不确定性编码为曲线、柱体、散点、颜色、图例及置信区间。通用指代分割模型能处理自然图像中的对象，但对“accuracy 对应的曲线”“下降最快的曲线”等图表语义未必可靠。ChartGround-Edit 以 Sa2VA 为底座，将自然语言指代解析、像素级图表元素定位和确定性的局部编辑串成一个可评测、可演示的系统。

## 项目目标

输入一张科学图表和一条自然语言指令，系统返回被指代图表元素的二值 mask，并基于该 mask 执行受限、可复现的编辑操作。第一版强调明确范围、可靠评测和工程完整度，不追求开放域图像编辑。

## MVP 范围

支持的图表：

- 折线图
- 柱状图
- 散点图
- 带置信区间的科学曲线图

支持的指代：

- 类别指代，如“蓝色曲线”
- 外观指代，如“最高的柱子”
- 图例指代，如“accuracy 对应的曲线”
- 趋势指代，如“下降最快的曲线”

支持的编辑：`highlight`、`recolor`、`extract`、`remove`。

## 输入输出形式

建议的统一推理输入：

```json
{
  "image": "relative/path/to/chart.png",
  "instruction": "把 accuracy 对应的曲线改成红色",
  "operation": "recolor",
  "parameters": {"color": "#ff0000"}
}
```

建议的统一输出：

```json
{
  "prediction_text": "... [SEG] ...",
  "mask": "relative/path/to/binary_mask.png",
  "edited_image": "relative/path/to/edited_chart.png",
  "metadata": {
    "operation": "recolor",
    "original_size": [1024, 768]
  }
}
```

训练样本的最小标注应包含稳定的 `sample_id`、图像路径、自然语言表达、目标 mask（PNG、RLE 或 polygon 三选一并在数据版本内统一）、图表类型、指代类型、目标元素属性和数据划分。编辑参数属于任务输入，不应混入分割真值。

## 技术架构

```text
图表 + 指代表达
  ├─ MLLM 视觉预处理 ──> 视觉 token + 文本 token ──> MLLM 生成 [SEG]
  │                                                     │
  └─ SAM2 grounding 预处理 ──> SAM2 图像特征             └─ [SEG] 隐状态
                                                               │
                                                     text_hidden_fcs
                                                               │
                                                     SAM2 prompt space
                                                               │
                                                     segmentation decoder
                                                               │
                                                           二值 mask
                                                               │
                                         确定性编辑器 + 操作参数
                                                               │
                                                          编辑后图表
```

建议把新增逻辑隔离在 `projects/chartground_edit/`：数据 schema 与校验、图表数据集、同步几何变换、Sa2VA 适配器、mask 后处理、四种编辑算子、评测和 Demo。只有在适配器无法复用公开接口时，才提出对上游核心代码的最小改动，并单独审核。

### Sa2VA 源码映射

以下记录基于当前提交 `b6fed1a` 的真实源码。

| 功能 | 文件、类与函数 | 当前行为 |
| --- | --- | --- |
| 数据集加载 | `projects/sa2va/datasets/sa2va_data_finetune.py`：`Sa2VAFinetuneDataset.load_data_list`、`_parse_annotations`、`prepare_data`；参考主训练集 `projects/sa2va/datasets/sa2va_data_01_refseg.py`：`Sa2VA01RefSeg._parse_annotations`、`prepare_data` | 微调格式从 JSON 读取 `image`、`mask`、`text`；polygon 经 `pycocotools.mask` 解码为二值 mask，并构造 human/GPT 对话。|
| 公共预处理 | `projects/sa2va/datasets/base.py`：`Sa2VADatasetMixin._process_single_image`、`_process_multiple_images`、`_create_token_string`、`_process_conversations_for_encoding`、`get_inputid_labels`；`projects/sa2va/models/preprocess/image_resize.py`：`DirectResize.apply_image` | MLLM 图像支路按架构生成视觉张量；grounding 支路生成 `g_pixel_values`。InternVL 使用动态图块，Qwen 使用其 processor，SAM2 支路默认直接缩放到 1024×1024。|
| 文本编码与 batch | `projects/sa2va/datasets/data_utils.py`：`template_map_fn`、`tokenize_conversation`、`sa2va_collect_fn` | 用户输入 label 置为 ignore，助手输出参与语言损失；collate 产出 `input_ids`、`labels`、`attention_mask`、`position_ids`、`pixel_values`、可选 `image_grid_thw`、`g_pixel_values`、`masks`、`frames_per_batch`。|
| 图像与文本组织 | `projects/sa2va/datasets/base.py`：`_init_architecture_config`、`_create_token_string`；`projects/sa2va/models/mllm/internvl.py`：`InternVLMLLM.forward`、`_embed_visual_features`；`projects/sa2va/models/mllm/qwenvl.py`：`Qwen2_5_VL.forward`；`projects/sa2va/models/mllm/qwen3vl.py`：`Qwen3VL.forward` | InternVL 用 `<img><IMG_CONTEXT>…</img>`，Qwen 用 `<|vision_start|><|image_pad|>…<|vision_end|>`，LLaVA 用 `<image>`；视觉特征填入相应占位 token，文本与视觉 token 一起进入语言模型。|
| `[SEG]` token | `projects/sa2va/datasets/common.py`：`ANSWER_LIST`；`projects/sa2va/models/sa2va.py`：`Sa2VAModel._add_special_tokens`、`forward`；`projects/sa2va/models/utils.py`：`find_seg_indices` | 数据回答模板包含 `[SEG]`；初始化时加入 tokenizer 并保存 token id；训练前向通过 `input_ids == seg_token_idx` 找出对应隐藏状态。|
| 文本隐藏状态投影 | `projects/sa2va/models/sa2va.py`：`Sa2VAModel.__init__`、`forward` 中的 `text_hidden_fcs` | 两层 MLP：MLLM hidden size → 同维 → SAM2 `hidden_dim`，ReLU 后接无 dropout 的输出层。|
| SAM2 / decoder 训练调用 | `projects/sa2va/models/sa2va.py`：`Sa2VAModel.forward` → `projects/sa2va/models/sam2_train.py`：`SAM2TrainRunner.get_sam2_embeddings`、`inject_language_embd` → `projects/sa2va/models/extension/sam2_base.py`：`SAM2Base._forward_sam_heads` | SAM2 backbone 提取 FPN 特征；投影后的语言 embedding 作为额外 sparse prompt 拼入 prompt encoder 输出，再调用 `sam_mask_decoder` 得到 mask logits。|
| 推理调用 | `projects/sa2va/hf/models_qwen3vl/modeling_sa2va_qwen.py`：`Sa2VAChatModelQwen.predict_forward`、`get_seg_hidden_states` → 同目录 `sam2.py`：`SAM2.get_sam2_embeddings`、`language_embd_inference` | 生成文本后提取每个生成 `[SEG]` 对应的 hidden state，投影并注入 SAM2；预测 mask 双线性缩放回原图尺寸，sigmoid 后以 0.5 二值化。不同 HF backbone 目录有平行实现，不能假定完全共享同一 Python 类。|
| mask loss | `projects/sa2va/models/sa2va.py`：`Sa2VAModel.sample_points`、`forward`；损失配置见 `projects/sa2va/configs/sa2va_finetune.py` | 真值最近邻缩放到预测分辨率；可采样不确定点；mask BCE 由 mmdet `CrossEntropyLoss(use_sigmoid=True, loss_weight=2.0)` 提供，Dice 为 `DiceLoss(..., loss_weight=0.5)`。|
| 语言模型 loss | `projects/sa2va/models/sa2va.py`：`Sa2VAModel.forward` 返回 `llm_loss=output.loss`；InternVL 的显式实现位于 `projects/sa2va/models/mllm/internvl.py`：`InternVLMLLM._compute_loss` | 对 next-token shifted logits/labels 做交叉熵；Qwen/LLaVA wrapper 把 `labels` 交给 Hugging Face 模型并使用其 `output.loss`。|
| 冻结与可训练参数 | `projects/sa2va/models/sa2va.py`：`Sa2VAModel.__init__`；`projects/sa2va/models/mllm/internvl.py`、`qwenvl.py`、`qwen3vl.py`、`llava.py` 的初始化与 LoRA 准备函数；配置见 `projects/sa2va/configs/sa2va_finetune.py` | 示例配置冻结 LLM 主体和视觉编码器，启用 LLM LoRA，并保存/训练指定 embedding 与 lm head；整个 grounding encoder 先冻结，`frozen_sam2_decoder=False` 时仅重新开启 `sam_mask_decoder`；`text_hidden_fcs` 默认可训练。最终集合仍应在实例化后以 `named_parameters()` 实测。|
| 训练入口与配置 | `tools/train.py`：`parse_args`、`train.main`；`tools/dist.sh`；`projects/sa2va/configs/sa2va_finetune.py`；主训练配置在 `projects/sa2va/configs/` 及其 Qwen 子目录 | XTuner/MMEngine 从 Python config 构建模型、数据集、优化器和 loop；分布式脚本用 torchrun 启动。|
| 评测入口 | `projects/sa2va/evaluation/run_all_evals.py`：`main`、`run_command`；`sa2va_eval_refcoco.py`：`main`；`sa2va_eval_ref_vos.py` 主入口；`sa2va_eval_gcg.py`；`projects/sa2va/evaluation/dist_test.sh` | HF 格式模型通过 `predict_forward` 推理；现有入口覆盖 RefCOCO、GCG、RefVOS。通用 MMEngine 测试入口另见 `tools/test.py:main`。|

### 训练态关键调用链

1. `Sa2VAFinetuneDataset.prepare_data` 读取图片与 polygon mask，生成含 `[SEG]` 的对话。
2. `Sa2VADatasetMixin._process_single_image` 分别准备 MLLM 的 `pixel_values` 和 SAM2 的 `g_pixel_values`；随后生成视觉占位 token 并编码对话。
3. `sa2va_collect_fn` padding 文本并组织所有模态和真值。
4. `Sa2VAModel.forward` 先调用 MLLM，取得 language-model loss 与最后层 hidden states。
5. `[SEG]` 位置的 hidden states 经 `text_hidden_fcs` 投影为 SAM2 prompt embedding。
6. `SAM2TrainRunner.get_sam2_embeddings` 提取图像特征，`inject_language_embd` 将语言 embedding 传给扩展的 `_forward_sam_heads`。
7. `SAM2Base._forward_sam_heads` 将语言 embedding 拼到 sparse prompt，调用 `sam_mask_decoder`。
8. 预测 mask 与最近邻缩放后的 GT 计算 sigmoid mask loss 和 Dice loss，和 `output.loss` 一起返回给 MMEngine。

### 当前环境快照

在 2026-09-15 的只读检查中，当前 shell 使用 `/home/dqwang/miniconda3/envs/llava/bin/python`（Python 3.10.21），而 `projects/sa2va/pyproject.toml` 声明 `requires-python = ">=3.11,<3.12"`；仓库内未发现 `projects/sa2va/.venv`，当前 PATH 也没有 `uv`。`nvidia-smi` 无法与 NVIDIA 驱动通信，因此本轮不能确认可用 GPU、CUDA 或显存。上述检查没有安装依赖、创建环境或加载模型。

## 数据集设计

### 数据来源策略

- 合成优先：用固定随机种子和可追溯参数生成四类图表，同时直接获得每个语义元素的无歧义像素 mask。
- 少量真实补充：只使用许可清晰、可再分发或仅发布标注索引的数据；人工复核图例、颜色、趋势和遮挡。
- 严格划分：按底层数值序列/模板族划分 train/validation/test，避免同一图表换配色后跨集合泄漏。
- 难例分层：颜色相近、曲线交叉、图例顺序扰动、细线、置信区间重叠、反锯齿和压缩失真。

### 标注与质量控制

- 一个表达可以对应一个或多个图元，但必须给出确定的 union-mask 语义。
- 明确区分线本体、marker、置信区间和图例样例线是否属于目标；该规则写入数据卡。
- 所有几何增强共享同一变换记录，并可将 mask 反变换回原图坐标。
- 校验图像/mask 尺寸、二值值域、非空目标、越界 polygon、表达与目标 ID 的一一对应。
- 保存生成器版本、字体与渲染后端版本、随机种子和 schema 版本。

## 评测指标

- 分割主指标：mIoU、cIoU；辅以 Dice/F1、Boundary F-score 和空预测率。
- 分层结果：按图表类型、指代类型、目标尺寸、线宽、遮挡/交叉和合成/真实来源报告。
- 指令输出：`[SEG]` 生成召回率、每条表达的 mask 数匹配率。
- 编辑正确性：目标区域修改覆盖率、非目标区域保持率、操作参数一致性；`extract` 额外检查 alpha/mask 一致性。
- 系统指标：单图端到端延迟、峰值显存、失败类型统计。所有指标必须由可复现脚本生成，不手填结论。

## 简历展示价值

- 展示从问题定义、数据 schema、合成数据与质量控制，到多模态模型适配、轻量微调、评测和 Demo 的完整闭环。
- 体现对 MLLM token grounding、SAM2 prompt/decoder、参数高效微调和像素级编辑的工程理解。
- 产出可视化强、容易解释的 before/mask/after 案例，同时用分层指标和失败案例体现研究严谨性。
- 通过隔离式子项目设计、测试、数据卡、配置和实验日志体现开源维护能力。

## 当前不做的功能

- 单个字符级 OCR 分割
- 数学公式理解
- 复杂三维图表
- 任意开放域图像编辑
- 本轮不实现训练、启动训练、下载权重/大型数据，也不修改 Sa2VA 模型核心逻辑

## 已知不确定点

- 需要先确定使用仓库声明的 Python 3.11 环境方案，并恢复或确认 GPU 驱动可见性；当前环境不足以验证 Sa2VA 推理。
- MVP 采用 InternVL3、Qwen2.5-VL、Qwen3-VL 还是 LLaVA 作为首个基线，取决于本机已有权重、显存和许可证要求。
- “曲线”mask 是否包含 marker，“置信区间曲线”是否默认包含 band，以及 `remove` 后采用透明、背景色填充还是局部修复，尚需形成标注/产品约定。
- 当前微调 loader 使用 polygon；ChartGround-Edit 是复用 polygon 格式还是支持无损 PNG/RLE 需要在 Phase 1 用 32 个样本验证。
- 当前训练代码会把原尺寸 GT 直接最近邻缩放到 decoder 输出尺寸。细线图表是否需要更高分辨率监督、边界损失或线宽容忍指标，必须经小样本实验验证。
- HF 推理代码按 backbone 复制在多个目录；首个基线确定前，适配器应避免绑定某一个私有实现细节。
