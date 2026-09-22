# Phase 2A：Sa2VA 零样本基线审计与接入计划

状态：**规划完成，未执行推理**  
审计日期：2026-09-15  
审计仓库提交：`d985631dc5a23eb14e4946b29630d78cbca9eb25`

## 1. 范围与证据边界

本轮只读取当前仓库、Conda 环境、`<MODEL_ROOT>`、Hugging Face 本地缓存和磁盘状态，并查询了 ByteDance 官方 Hugging Face 仓库的小型元数据。未下载模型或数据，未安装/升级依赖，未加载模型，未执行推理或训练，也未修改 Sa2VA 核心代码。

以下“路径完整”仅表示当前源码中能追踪到从图像和文本到原图尺寸二值 mask 的静态调用链；由于当前环境没有 PyTorch、也没有完整 Sa2VA checkpoint，运行时兼容性和峰值显存均为**未确认**。远端 checkpoint 内的 `trust_remote_code` 与本地 `projects/sa2va/hf/` 是否逐字节一致也**未确认**，Phase 2B 应固定 checkpoint revision 并在执行前复查下载代码。

## 2. 硬件与当前运行条件

以用户从普通终端提供的信息为准：

- CUDA：12.4；NVIDIA driver：550.78。
- GPU：7 张 NVIDIA GeForce RTX 3090，每张 24,576 MiB。
- 当前显存占用：GPU 0/1 为 8,690 MiB；GPU 2/3/4 为 24,252 MiB；GPU 5 为 13,278 MiB；GPU 6 为 13,866 MiB。
- 按总量减占用计算，GPU 0/1 各约有 15,886 MiB 可用，GPU 2/3/4 各约 324 MiB，GPU 5 约 11,298 MiB，GPU 6 约 10,710 MiB。占用是瞬时状态，Phase 2B 启动前必须重新检查。
- `/home`：15 TiB 总量，13 TiB 已用，约 2.0 TiB 可用，使用率 87%。
- 当前解释器：`<PYTHON>`，Python 3.11.16。
- 当前解释器不能导入 `torch`，也没有 `transformers`，因此当前**不具备运行条件**。
- `projects/sa2va/.venv` 不存在；命令行中没有找到 `uv`。

## 3. 当前仓库的共同推理链

四条候选 HF 路径都通过 checkpoint 中注册的 Hugging Face remote code 暴露 `predict_forward`，并复用同一套 SAM2 实现。四份 `sam2.py` 的 SHA-256 一致：

- `projects/sa2va/hf/models/sam2.py`
- `projects/sa2va/hf/models_qwen2_5_vl/sam2.py`
- `projects/sa2va/hf/models_qwen3vl/sam2.py`
- `projects/sa2va/hf/models_llava/sam2.py`

共同调用链如下：

1. InternVL/LLaVA 的 `preparing_for_generation()` 注册 tokenizer、`[SEG]` token id 和生成配置；Qwen2.5/Qwen3 在 `predict_forward()` 内注册 processor 和 `[SEG]` token id。四个模型类的构造函数都创建 `DirectResize(1024)`。
2. 各模型的 `predict_forward()` 分别准备 MLLM 图像输入，同时将原始 PIL 图像直接缩放为 1024×1024，交给 `SAM2.preprocess_image()` 和 `SAM2.get_sam2_embeddings()`。
3. 模型调用 `generate(..., output_hidden_states=True, return_dict_in_generate=True)`；模块级 `get_seg_hidden_states()` 按 `[SEG]` token id 从生成 hidden state 中取出对应向量。
4. `self.text_hidden_fcs` 将文本 hidden state 投影为 SAM2 的语言提示 embedding。
5. `SAM2.language_embd_inference()` 调用 `SAM2VideoPredictor.add_language_embd()`，再由 `propagate_in_video()` 产出 mask logits。
6. `predict_forward()` 使用 `torch.nn.functional.interpolate(..., size=(original_h, original_w), mode='bilinear', align_corners=False)` 恢复原图尺寸，再执行 `sigmoid() > 0.5`，最后 `.cpu().numpy()`。

`SAM2VideoPredictor.init_state()` 将内部 `video_height`、`video_width` 设为 `image_size=1024`。从当前代码可静态推导：单图、单个 `[SEG]` 的 SAM2 输出在阈值前通常为 `(1, 1, 1024, 1024)` logits，处于“原图被非等比例拉伸到 1024 正方形”的坐标系；此 shape 尚未经过真实运行确认。公开 `predict_forward()` 返回的 `prediction_masks` 是一个列表，每个元素为 NumPy `bool` 数组，静态可见 shape 为 `(1, H_original, W_original)`，坐标系为原图像素坐标。保存 PNG 时应移除长度为 1 的 batch 维，并映射 `False -> 0`、`True -> 255`。

注意：这里没有 padding 或 crop 可逆参数。SAM2 分支是直接 1024 正方形缩放，再直接回原尺寸；MLLM 分支可能使用动态切块或 processor resize，但不参与 SAM2 mask 的坐标逆变换。

文本中的 `[SEG]` 字符串筛选还可复用 `projects/sa2va/models/utils.py` 的 `find_seg_indices(text)`：存在 `<answer>...</answer>` 时只取第一个 answer 范围内的 `[SEG]`，否则取全部 `[SEG]`。

## 4. 候选模型路径审计

### 4.1 InternVL 系列

| 审计项 | 当前仓库证据与结论 |
|---|---|
| 对应目录 | HF 导出实现：`projects/sa2va/hf/models/`；训练封装：`projects/sa2va/models/mllm/internvl.py`；配置在 `projects/sa2va/configs/` 根目录。 |
| 模型类和关键函数 | `configuration_sa2va_chat.py::Sa2VAChatConfig`；`modeling_sa2va_chat.py::Sa2VAChatModel`、`preparing_for_generation()`、`predict_forward()`、模块函数 `get_seg_hidden_states()`；`sam2.py::SAM2`。训练侧为 `internvl.py::InternVLMLLM`。 |
| checkpoint/配置 | `projects/sa2va/README.md` 列出 InternVL2.5 的 `ByteDance/Sa2VA-1B/4B/8B/26B` 和 InternVL3 的 `ByteDance/Sa2VA-InternVL3-2B/8B/14B`。精确 InternVL3 配置包括 `sa2va_in30_2b.py`、`sa2va_in30_8b.py`、`sa2va_in30_14b.py`。本地未发现完整 Sa2VA checkpoint。 |
| 图像推理入口 | `projects/sa2va/demo/demo.py` 和 `projects/sa2va/evaluation/sa2va_eval_refcoco.py` 通过 Hugging Face `AutoModel`/`AutoModelForCausalLM` 加载，再调用 `model.predict_forward(image=PIL.Image, text=..., tokenizer=...)`。 |
| 自然语言分割入口 | 同一 `predict_forward()`。RefCOCO 提示词来源为 `projects/sa2va/evaluation/dataset/RES.py::RESDataset.get_questions()`，形式为 `<image>\n Please segment {expression} in this image.`。 |
| `[SEG]` 解析位置 | `Sa2VAChatModel.preparing_for_generation()` 得到 token id；`modeling_sa2va_chat.py::get_seg_hidden_states()` 提取 token hidden state；回答文本范围选择可用 `projects/sa2va/models/utils.py::find_seg_indices()`。 |
| hidden state 到 decoder | `predict_forward()` → `get_seg_hidden_states()` → `self.text_hidden_fcs` → `self.sam2_model.language_embd_inference()` → `add_language_embd()` → `propagate_in_video()`。 |
| 最终 mask | 公开结果为 `list[np.ndarray]`，元素为 bool、静态可见 shape `(1,H,W)`，已经恢复到原图坐标；内部 logits 为 1024 正方形坐标。运行时 shape 未确认。 |
| 官方示例命令 | 有。`projects/sa2va/README.md` 给出通用 `demo/demo.py`、Gradio 和评测命令；notebook 使用 Sa2VA-8B。`demo/predict-img.py` 会在缓存缺失时触发下载，本项目不应使用该脚本。 |
| 训练/微调配置 | 有。`projects/sa2va/configs/sa2va_in30_2b.py`、`sa2va_in30_8b.py`、`sa2va_in30_14b.py` 是 InternVL3 训练配置；同目录 `sa2va_finetune.py` 是微调示例。`InternVLMLLM` 支持 LLM/backbone LoRA 配置。没有发现专门为 `Sa2VA-1B` 写好的本项目 32 样本配置。 |
| 实现完整性 | 源码静态调用链完整，能够追踪到原图尺寸二值 mask；本机运行未验证。 |
| 主要依赖 | PyTorch、torchvision、transformers、Pillow、NumPy、peft、timm、einops、qwen-vl-utils/项目公共依赖、mmengine、flash-attn 等；InternVL2.5 按 README 对应 `legacy` 依赖组。具体锁定版本见 `projects/sa2va/pyproject.toml` 和 `uv.lock`。 |

### 4.2 Qwen2.5-VL 系列

| 审计项 | 当前仓库证据与结论 |
|---|---|
| 对应目录 | HF 实现：`projects/sa2va/hf/models_qwen2_5_vl/`；训练封装：`projects/sa2va/models/mllm/qwenvl.py`；配置：`projects/sa2va/configs/sa2va_qwenvl25/`。 |
| 模型类和关键函数 | `configuration_sa2va_chat.py::Sa2VAChatConfigQwen`（继承 `Qwen2_5_VLConfig`）；`modeling_sa2va_chat.py::Sa2VAChatModelQwen`、`preparing_for_generation()`、`predict_forward()`、`get_seg_hidden_states()`；训练侧 `qwenvl.py::Qwen2_5_VL`。 |
| checkpoint/配置 | README 列出 `ByteDance/Sa2VA-Qwen2_5-VL-3B`、`ByteDance/Sa2VA-Qwen2_5-VL-7B`；配置目录有 3B、7B 及 finetune 配置。 |
| 图像推理入口 | `demo/demo.py` 以 `AutoProcessor` 加载 Qwen processor，`predict_forward(..., processor=...)`；评测脚本同样区分 processor 分支。 |
| 自然语言分割入口 | `Sa2VAChatModelQwen.predict_forward()` 使用 Qwen chat template 和 `qwen_vl_utils.process_vision_info()` 处理图文消息。 |
| `[SEG]` 解析位置 | `Sa2VAChatModelQwen.predict_forward()` 从 `processor.tokenizer` 获取 id；模块函数 `get_seg_hidden_states()` 提取；文本范围可用 `models/utils.py::find_seg_indices()`。 |
| hidden state 到 decoder | `predict_forward()` → `get_seg_hidden_states()` → `text_hidden_fcs` → `SAM2.language_embd_inference()`。后续为共同 SAM2 路径。 |
| 最终 mask | 与共同路径一致：公开返回原图坐标的 bool NumPy mask 列表；静态可见元素 shape `(1,H,W)`。运行时未确认。 |
| 官方示例命令 | README 有可接受任意 checkpoint path 的通用 demo/eval 命令；未发现专门只写给 Qwen2.5-VL checkpoint 的独立命令。 |
| 训练/微调配置 | 有，位于 `projects/sa2va/configs/sa2va_qwenvl25/`；另有 `projects/sa2va/configs/sa2va_qwen_finetune.py`。`Qwen2_5_VL` 支持 LoRA。 |
| 实现完整性 | 静态链完整；当前机器未运行验证。 |
| 主要依赖 | PyTorch、torchvision、transformers、AutoProcessor、qwen-vl-utils、Pillow、NumPy、peft、mmengine、flash-attn 等；README 将 Qwen2.5-VL 归入 `latest` 依赖组。 |

### 4.3 Qwen3-VL 系列

| 审计项 | 当前仓库证据与结论 |
|---|---|
| 对应目录 | SAM2 HF 实现：`projects/sa2va/hf/models_qwen3vl/`；SAM3 变体：`projects/sa2va/hf/models_qwen3vl_sam3/`；训练封装：`projects/sa2va/models/mllm/qwen3vl.py`；配置：`projects/sa2va/configs/sa2va_qwenvl3/`。本计划只把 SAM2 变体作为可比候选。 |
| 模型类和关键函数 | `models_qwen3vl/configuration_sa2va_chat.py::Sa2VAChatConfigQwen`（继承 `Qwen3VLConfig`）；`modeling_sa2va_chat.py::Sa2VAChatModelQwen`、`preparing_for_generation()`、`predict_forward()`、`get_seg_hidden_states()`；训练侧 `qwen3vl.py::Qwen3VL`。 |
| checkpoint/配置 | README 列出 `ByteDance/Sa2VA-Qwen3-VL-2B`、`ByteDance/Sa2VA-Qwen3-VL-4B` 和独立 SAM3 checkpoint。当前 SAM2 训练配置只发现 `sa2va_qwen3_4b.py`，未发现精确 2B 配置。 |
| 图像推理入口 | 通用 `demo/demo.py` 的 Qwen processor 分支以及通用评测入口，最终调用 `predict_forward(..., processor=...)`。 |
| 自然语言分割入口 | `Sa2VAChatModelQwen.predict_forward()`，通过 Qwen3-VL chat template 和 `process_vision_info()` 构造模型输入。 |
| `[SEG]` 解析位置 | `Sa2VAChatModelQwen.predict_forward()` 从 processor tokenizer 获取 id，模块函数 `get_seg_hidden_states()` 提取；文本范围可用 `models/utils.py::find_seg_indices()` 筛选。 |
| hidden state 到 decoder | `predict_forward()` → `get_seg_hidden_states()` → `text_hidden_fcs` → SAM2；SAM3 checkpoint 走另一套 `models_qwen3vl_sam3/sam3.py::SAM3` 路径，不应混用。 |
| 最终 mask | SAM2 版本与共同路径一致；SAM3 版本的具体输出契约不作为 Phase 2B 首轮接口依据。 |
| 官方示例命令 | README 有通用 demo/eval 命令和 checkpoint 链接；未发现 Qwen3-VL-2B 专用命令。 |
| 训练/微调配置 | 有 Qwen3-VL 4B SAM2 和 4B/8B SAM3 配置；`Qwen3VL` 支持 LoRA。精确 2B SAM2 训练配置未确认。 |
| 实现完整性 | SAM2 HF 推理链静态完整；2B 精确训练配置缺失，运行未验证。SAM3 是独立路线，不纳入最小 baseline。 |
| 主要依赖 | PyTorch、torchvision、较新的 transformers、AutoProcessor、qwen-vl-utils、Pillow、NumPy、peft、mmengine、flash-attn 等；README 对应 `latest` 依赖组。 |

### 4.4 LLaVA 系列

| 审计项 | 当前仓库证据与结论 |
|---|---|
| 对应目录 | HF 实现：`projects/sa2va/hf/models_llava/`；训练封装：`projects/sa2va/models/mllm/llava.py`；配置：`projects/sa2va/configs/sa2va_llava/`。 |
| 模型类和关键函数 | `configuration_sa2va_chat.py::Sa2VAChatConfigLlava`（继承 `LlavaConfig`）；`modeling_sa2va_chat.py::Sa2VAChatModelLlava`、`preparing_for_generation()`、`predict_forward()`、`get_seg_hidden_states()`；训练侧 `llava.py::LlavaVLM`。 |
| checkpoint/配置 | README 列出 `ByteDance/Sa2VA-LLaVA-1.5-7B`；精确训练配置为 `configs/sa2va_llava/sa2va_llava15_7b.py`。 |
| 图像推理入口 | 通用 `demo/demo.py` 和评测脚本的非 Qwen tokenizer 分支，调用 `predict_forward(image=..., text=..., tokenizer=...)`。MLLM 图像分支缩放到 336×336 并执行 CLIP normalization。 |
| 自然语言分割入口 | `Sa2VAChatModelLlava.predict_forward()`，内部构造 Vicuna/LLaVA 对话模板。SAM2 仍单独接收 1024 正方形图像。 |
| `[SEG]` 解析位置 | `preparing_for_generation()` 和模块函数 `get_seg_hidden_states()`；字符串层面可由 `models/utils.py::find_seg_indices()` 选择。 |
| hidden state 到 decoder | `predict_forward()` → `get_seg_hidden_states()` → `text_hidden_fcs` → `SAM2.language_embd_inference()`。 |
| 最终 mask | 与共同路径一致：原图坐标 bool NumPy mask 列表，静态可见元素 shape `(1,H,W)`；运行时未确认。 |
| 官方示例命令 | README 有通用 demo/eval 命令，但未发现 LLaVA checkpoint 专用命令。 |
| 训练/微调配置 | 有精确 7B 配置；`LlavaVLM` 支持 LoRA。 |
| 实现完整性 | 静态推理和训练路径完整；运行未验证。 |
| 主要依赖 | PyTorch、torchvision、含 `LlavaForConditionalGeneration` 的 transformers、Pillow、NumPy、peft、mmengine、flash-attn 等。当前 `llava` Conda 环境的 transformers 4.37.2 与项目当前依赖配置不对应，不能据此认定可运行。 |

### 4.5 导出代码与 checkpoint 映射

`projects/sa2va/tools/convert_to_hf.py` 根据配置路径/架构选择并复制相应 HF remote-code 目录：Qwen3 选择 `hf/models_qwen3vl`，Qwen2.5 选择 `hf/models_qwen2_5_vl`，LLaVA 选择 `hf/models_llava`，否则选择 InternVL 的 `hf/models`，并写入对应 `auto_map`。这证明仓库内四套 HF 路径与 checkpoint 类型的预期映射，但不能证明远端某一 revision 与当前工作树完全相同。

## 5. 本地已有资源

### 5.1 Conda 环境

只读检查到 `base`、`chartground`、`d2l`、`deepsc`、`formula_ui`、`llava`、`minimind`、`py310`、`py310_ft`、`py310_ft_fsdp`、`qwen3vl` 等环境。没有一个已确认同时满足当前项目的 Python 3.11、PyTorch、transformers 和 Sa2VA 完整依赖组合：

- `chartground`：Python 3.11.16，但无 torch/transformers。
- `qwen3vl`：Python 3.11.16，但无 torch/transformers。
- 其他被抽查环境存在不同版本的 torch/transformers/peft，但均为 Python 3.10 或与 `projects/sa2va/pyproject.toml` 的依赖组不匹配。

因此不复用这些环境，也不把“能导入某几个包”当作 Sa2VA 可运行的证据。

### 5.2 模型目录与 HF 缓存

| 资源 | 路径 | 状态与用途判断 |
|---|---|---|
| Qwen3-VL-4B-Instruct | `<MODEL_ROOT>/Qwen3-VL-4B-Instruct` | 目录约 12 MiB；index 声明约 8.27 GiB 权重，但 2 个 shard 都不存在。它还是基础 `Qwen3VLForConditionalGeneration`，不是包含 SAM2/text projection 的 Sa2VA checkpoint。不可直接使用。 |
| llava-v1.5-7b | `<MODEL_ROOT>/llava-v1.5-7b` | 约 13 GiB，两个原始 LLaVA shard 存在；配置类为 `LlavaLlamaForCausalLM`，不是 Sa2VA HF 路径要求的组合模型，也没有 SAM2/text projection。不可直接作为本阶段 checkpoint。 |
| Qwen3-1.7B | `<MODEL_ROOT>/Qwen3-1.7B` | 约 3.8 GiB，文本基础模型，不是 Sa2VA。 |
| vicuna-7b-v1.5 | `<MODEL_ROOT>/vicuna-7b-v1.5` | 约 13 GiB，基础语言模型，不是 Sa2VA。 |
| Qwen2.5-1.5B/3B-Instruct | `<MODEL_ROOT>/Qwen2.5-1.5B-Instruct`、`<MODEL_ROOT>/Qwen2.5-3B-Instruct` | 分别约 2.9/5.8 GiB，文本基础模型，不是 Sa2VA。 |
| HF cache refs | `~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct`、`~/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-7b` | 只发现 refs 等小文件，未发现可用 snapshot 权重。 |

结论：本地**没有完整、代码版本明确且可直接加载的 Sa2VA checkpoint**。

## 6. 官方 checkpoint 元数据与下载预算

以下数值来自 2026-09-15 对 ByteDance 官方 Hugging Face 仓库 API 的只读元数据查询，不包含任何权重下载。GiB 按 `bytes / 1024^3` 计算：

| checkpoint | 固定 revision | 仓库总字节 | 权重字节 | 权重文件 |
|---|---|---:|---:|---:|
| `ByteDance/Sa2VA-1B` | `82faf06c93f6ce3fdc0ad3d45b57fd52c463daeb` | 4,058,775,277（3.78 GiB） | 4,046,848,176（3.77 GiB） | 1 |
| `ByteDance/Sa2VA-InternVL3-2B` | `15837dcaecc304714a1f0f069e74f47e47521c7f` | 8,673,308,961（8.08 GiB） | 8,656,821,792（8.06 GiB） | 2 |
| `ByteDance/Sa2VA-Qwen2_5-VL-3B` | `200fea03ebb9976e21804c5a7a1915cab274105e` | 17,191,910,442（16.01 GiB） | 17,175,636,184（16.00 GiB） | 4 |
| `ByteDance/Sa2VA-Qwen3-VL-2B` | `173fb6512b3ad6f3c1fe25efa09e723d01d5b6d4` | 10,683,757,998（9.95 GiB） | 10,667,312,184（9.93 GiB） | 3 |
| `ByteDance/Sa2VA-LLaVA-1.5-7B` | `2655e3116cd0199bf6e62332ff6c801136de5e9c` | 29,225,622,126（27.22 GiB） | 29,221,113,480（27.21 GiB） | 6 |

这些是磁盘 artifact 大小，不等同于加载后显存。完整 Python 环境的磁盘增量需由包解析器确定，目前未确认。若下载到指定本地目录且工具另保留缓存，保守地为首选 checkpoint 预留两倍仓库体积（约 7.56 GiB）；这是操作余量，不是模型自身大小。当前约 2.0 TiB 可用空间充足。

## 7. 首个 baseline 选择

### 7.1 唯一首选

**`ByteDance/Sa2VA-1B`，固定 revision `82faf06c93f6ce3fdc0ad3d45b57fd52c463daeb`，InternVL2.5 路径，BF16。**

选择依据：

1. 当前仓库的 InternVL HF 图像分割路径可静态追踪至原图尺寸 bool mask。
2. 它是审计到的官方组合 checkpoint 中最小者，仓库 3.78 GiB、权重 3.77 GiB，最适合频繁调试。
3. 当前 GPU 0/1 各约 15.51 GiB 空闲；与其他候选相比，首选给激活、KV cache、SAM2 和 allocator 留出的静态空间最大。
4. README 将该模型和 `legacy` 依赖组明确关联，checkpoint 名称和代码家族可追踪。
5. `InternVLMLLM` 已有 LoRA 支持，因此后续具备轻量微调路径；但仍需单独制作配置并先通过 32 样本过拟合门槛。

精度选择为 BF16，因为仓库评测入口显式用 `torch.bfloat16`，HF wrapper 也以 BF16 作为推理 dtype；RTX 3090 支持 BF16。首轮不使用 4-bit：仓库没有为该 HF 推理路径提供经过验证的 4-bit 命令或 bitsandbytes 依赖契约。FP16 只作为 BF16 实测不兼容后的受控回退。

**显存结论边界：**3.77 GiB 只是磁盘权重字节，不能据此声称峰值显存。按当前占用，GPU 0/1 各约 15.51 GiB 空闲；扣除磁盘权重数值后约 11.74 GiB 的“静态字节余量”仅用于候选间排序，不能替代实际显存测量。是否能单卡完成首次生成必须在 Phase 2B 用单样本 smoke test 测量。当前源码多处直接使用 CUDA，并未提供经验证的 CPU offload 路径，因此首轮**不计划 CPU offload**；若单卡 OOM，应停止并重新审计，而不是临时声称 offload 可用。

### 7.2 唯一备选

**`ByteDance/Sa2VA-InternVL3-2B`，固定 revision `15837dcaecc304714a1f0f069e74f47e47521c7f`，BF16。**

它同样有完整静态调用链，并且仓库有精确 `sa2va_in30_2b.py` 和 finetune 示例，后续 LoRA 路径更明确。缺点是权重约 8.06 GiB，按 GPU 0/1 当前空闲状态只剩约 7.45 GiB 的静态字节余量，运行空间明显小于首选；因此只在首选发生明确的代码/兼容性问题后尝试，而且仍需单样本预检。

### 7.3 不选其他路径的原因

- Qwen3-VL-2B：官方仓库约 9.95 GiB，比 InternVL3-2B 更大；依赖 Qwen processor；当前仓库只有精确 4B SAM2 训练配置，未发现 2B 配置。它更新但不因此更适合首轮稳定性验证。
- Qwen2.5-VL-3B：权重约 16.00 GiB，已经大于 GPU 0/1 当前约 15.51 GiB 空闲，尚未计入激活、KV cache 和 SAM2；不适合作为当前单卡首测。
- LLaVA-1.5-7B：权重约 27.21 GiB，超过单张 24 GiB GPU 的物理容量；本地 13 GiB LLaVA 是基础模型，不是 Sa2VA。首轮采用它会引入多卡或 offload 变量。
- 更大的 InternVL/Qwen checkpoint：不满足“尽量小、便于反复调试”的优先级，也没有证据表明首轮调用链验证必须依赖更大参数量。
- Qwen3-VL-SAM3：使用独立 SAM3 grounding 路径，会同时改变 backbone 和 mask decoder，不适合作为最小 SAM2 基线。

## 8. 所需环境与依赖

当前仓库 `projects/sa2va/pyproject.toml` 要求 Python `>=3.11,<3.12`、PyTorch `>=2.6.0`，并定义 CUDA 12.4 wheel 源；`legacy` 组锁定 transformers 4.49.0、peft 0.11.1，`latest` 组锁定 transformers 4.57.1、peft 0.17.1。首选 InternVL2.5 应使用 `legacy` 组。

首选所需主要组件包括：

- `torch`、`torchvision`；
- `transformers==4.49.0`、`peft==0.11.1`（项目 legacy 组）；
- `flash-attn==2.7.3`（项目公共依赖；部分模型源码有 fallback，但项目环境仍显式要求）；
- `timm`、`einops`、`sentencepiece`/对应 tokenizer 依赖；
- `numpy`、`Pillow`、`opencv-python-headless`；
- `mmengine`、项目的 xtuner 及其他 `pyproject.toml` 公共依赖；
- Hugging Face Hub 下载工具，仅在用户批准下载 checkpoint 后需要。

不要从上述清单手工拼装环境；下一轮应以 `projects/sa2va/pyproject.toml`、`projects/sa2va/uv.lock` 和仓库根目录 `setup_env.sh` 为唯一依赖来源。当前还缺少 `uv`，本轮不安装。

## 9. ChartGround-Edit 的最小推理边界

本轮只定义接口，不实现模型加载。建议新增一个薄的 `Sa2VAHFBackend`，其外部契约不暴露 InternVL/Qwen/LLaVA 内部类型：

```python
@dataclass
class PredictionResult:
    mask: Image.Image | None          # mode L，原图尺寸，只含 0/255
    raw_mask: np.ndarray | None       # 上游公开返回的 bool mask；不是 logits
    text_output: str | None
    model_name: str
    instruction: str
    inference_time_ms: float
    metadata: dict[str, Any]
    success: bool
    failure_reason: str | None

def predict_mask(
    image: Image.Image,
    instruction: str,
    parameters: Mapping[str, Any] | None = None,
) -> PredictionResult:
    ...
```

最小实现原则：

- backend 构造参数显式指定本地 checkpoint、固定 revision 记录、tokenizer 或 processor 类型、device 和 dtype；不要根据模型名称字符串猜 backbone。
- 只调用 checkpoint remote code 的公开 `predict_forward()`。PyTorch/transformers 延迟导入，使 Phase 1 数据与编辑模块无需模型环境也能工作。
- `raw_mask` 在首版仅指 `predict_forward()` 已阈值化的公开 bool 输出。当前公开接口不暴露 logits/probability；不修改上游代码就不能诚实地把它记录成 logits。
- 当前 upstream 固定采用 `sigmoid > 0.5`。首版 `parameters` 中的 threshold 只能是 0.5，其他值明确报“不受公开接口支持”，不能对 bool mask 再伪造概率阈值。
- 验证公开 mask 的 shape 恰为 `(1,H_original,W_original)` 或去 batch 后 `(H_original,W_original)`。尺寸不符应记为 `mask_size_mismatch`，不再做第二次 resize，以免掩盖 upstream 坐标错误。
- 转 PNG 时将 bool/0-1 映射为 mode `L` 的 0/255。原始恢复已经由 upstream 完成；当前没有 padding/crop 参数可逆变换，只有 1024 正方形 resize 的反向插值。
- 多个 mask 时沿用官方评测的文本规则：有 `<answer>` 时选择第一个 answer 内的 `[SEG]`，否则选择全部 `[SEG]`；单目标 ChartGround-Edit v0 默认对选中的 mask 做逻辑并集，并在 metadata 记录 token 数、mask 数和所选下标。数量不匹配不得静默截断。
- 文本没有 `[SEG]` 时返回 `mask=None`、`success=False`、`failure_reason='no_seg_token'`，保留文本和耗时。存在 `[SEG]` 但无 mask 时记录 `missing_mask`。
- 有效但全空的模型输出保留为全 0 PNG，并标记 `failure_reason='empty_prediction'`。它参与真实指标，不能换成 GT mask。由于现有 editor 对空 mask 有明确拒绝行为，该样本不执行编辑，`edited_path` 记为 null。
- 调用成功、mask 非空且尺寸/值域合法时，直接把 `result.mask` 交给已有 `chartground_edit.editing.edit(image, mask, action, parameters)`。backend 不复制编辑逻辑。

这样 ChartGround-Edit 只依赖“原图尺寸二值 mask + 状态元数据”契约，backbone 特有的 tokenizer、processor、CUDA 和 `[SEG]` hidden-state 处理都留在一个薄 backend 内，不建设通用插件框架。

## 10. Phase 2B 最小验收实验

### 10.1 样本与 Prompt

使用 `projects/chartground_edit/data/synthetic_v0/annotations.jsonl` 的 test split。当前 test split 为 4 条，恰好覆盖 line、bar、scatter、confidence_band 四种 chart type。annotation 中的 `instruction` 必须原样嵌入，不得引用 `target_attributes` 或 GT mask 构造额外提示。

建议首轮统一 Prompt：

```text
<image>
Please segment the chart element targeted by this instruction in the image: {instruction}
Please respond with a segmentation mask.
```

首轮只验证调用链，不做 prompt sweep，不追求高指标。

### 10.2 逐样本产物

每条 test 样本保存：

- 原始图像；
- 模型 `predicted_mask.png`（失败时没有文件，JSONL 明确记录原因；空预测保存全 0 mask）；
- GT mask；
- predicted-mask overlay 与 GT-mask overlay；
- 使用 predicted mask 的 edited result；预测失败或为空时该路径为 null；
- 一行 JSONL，至少包含 sample id、输入/产物路径、完整 prompt、checkpoint/revision、dtype/device、文本输出、`[SEG]` 数、mask 数、状态/失败原因、推理时间、IoU 和 Dice。

再生成一张预测/GT 对比 gallery。gallery 必须显式显示失败、无 `[SEG]` 和空 mask，不能只展示成功样本。

### 10.3 指标定义

- IoU：`|P ∩ G| / |P ∪ G|`。GT 按 Phase 1A 约束非空；无 mask、无 `[SEG]`、无效 mask 和空预测都按全 0 prediction 计 0，同时保留失败类型。
- Dice：`2|P ∩ G| / (|P| + |G|)`；上述失败同样计 0。
- empty prediction rate：无可用前景像素的样本数除以总样本数；`mask=None` 也计入，并另按原因统计。
- inference success rate：得到含前景、原图尺寸、二值合法且与 `[SEG]` 数量规则一致的 mask 的样本数除以总样本数。
- 单样本推理时间：只包围 `predict_forward()`，模型加载时间另记；CUDA 计时前后同步。逐样本保存毫秒值，并报告中位数/均值以及首条是否为冷启动。

禁止用 GT mask 替代模型预测。模型调用异常、OOM、没有 `[SEG]`、mask 数量不匹配和空 mask 都必须进入逐样本 JSONL 和分母。

### 10.4 Phase 2B 验收顺序

1. 再次检查 GPU 0/1 空闲显存、磁盘和 checkpoint 文件完整性。
2. 在固定 revision、BF16、单张 GPU 上只跑 1 条样本，记录实际峰值显存和返回类型/shape。
3. 若契约与静态审计一致，再运行全部 4 条 test 样本。
4. 自动校验 predicted PNG 的尺寸和值域，计算指标并生成 gallery。
5. 将命令、commit、checkpoint revision、真实指标和失败样本追加到 `EXPERIMENTS.md`。

Phase 2B 的完成条件只是“真实模型输出端到端可追溯且失败不被隐藏”，不是达到某个分割精度阈值。

## 11. 风险与回退方案

- **环境尚不可用：**当前 chartground 环境无 torch，且 uv 不存在。回退是按锁文件新建项目隔离环境，不污染现有环境。
- **remote code 版本风险：**下载时固定 revision，执行前检查 checkpoint 自带 `auto_map` 和 Python 文件；运行记录同时保存本地仓库 commit 与 checkpoint revision。
- **峰值显存未知：**只在占用最低的 GPU 0/1 单样本预检。若 OOM，先确认其他进程和实际峰值；不要未经审计切换 4-bit、CPU offload 或多卡。
- **源码硬编码 CUDA：**SAM2 路径使用 CUDA 调用，CPU offload 未验证。需要 offload 时作为单独阶段审计。
- **非等比例 1024 缩放：**细线和极端长宽比图表可能在双线性恢复与 0.5 阈值时损失，这是当前上游语义，应在 metadata 记录原尺寸并用 overlay 检查。
- **文本与 mask 数不一致：**严格失败，不采用 `zip` 静默丢弃。
- **中文合成指令的域差异：**首轮 prompt 固定并记录原文；失败仍是真实基线。之后只能一次改一个 prompt 条件。
- **公开接口没有 logits：**首轮不承诺概率阈值 sweep；需要 logits 时另行设计且不得直接修改上游核心。
- **首选 checkpoint 不兼容：**若固定 revision 在 legacy 环境仍出现可复现的接口错误，保留错误记录，再切换唯一备选 InternVL3-2B/latest 环境；不能同时改变 checkpoint、dtype 和 prompt。

## 12. 下一轮建议命令（本轮均未执行）

以下命令只作为用户批准安装/下载后的执行草案；必须先复查脚本和目标目录。`setup_env.sh` 需要系统已有 `uv`，当前尚无：

```bash
# 1. 用户批准安装依赖且 uv 可用后，从仓库根目录创建隔离的 legacy 环境
cd <REPO_ROOT>
bash setup_env.sh sa2va legacy

# 2. 用户批准下载后，固定 revision 下载首选 checkpoint
huggingface-cli download ByteDance/Sa2VA-1B \
  --revision 82faf06c93f6ce3fdc0ad3d45b57fd52c463daeb \
  --local-dir <MODEL_ROOT>/Sa2VA-1B

# 3. Phase 2B 启动前只读预检
nvidia-smi --query-gpu=index,name,memory.total,memory.used \
  --format=csv,noheader
<REPO_ROOT>/projects/sa2va/.venv/bin/python -c \
  "import torch, transformers; print(torch.__version__, torch.version.cuda, transformers.__version__)"
```

Phase 2B 的 backend/CLI 尚未实现，因此本轮不虚构其运行命令。实现后应先提供单样本 `--limit 1 --device cuda:0 --dtype bfloat16` smoke 命令，再提供 test split 批量命令。

## 13. Phase 2A 结论

- 首选：Sa2VA-1B / InternVL2.5 / BF16 / 单卡 GPU 0 或 1，峰值显存待实测。
- 备选：Sa2VA-InternVL3-2B / BF16。
- 当前运行条件：不具备；缺少项目运行环境和完整 Sa2VA checkpoint。
- Phase 2A 只完成规划。Phase 2B 推理、指标和模型能力全部为“未运行/待验证”。
