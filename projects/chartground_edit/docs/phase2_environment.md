# Phase 2B-0：Sa2VA-InternVL3-2B 推理环境

状态：**环境与本地 checkpoint 校验完成，尚未加载模型或执行推理**

验证日期：2026-09-15

仓库分支：`feature/chartground-edit`

## 1. 安装方案与依据

本环境遵循仓库根目录 `setup_env.sh` 和 `projects/sa2va/pyproject.toml`：

- Python 要求为 `>=3.11,<3.12`。
- `setup_env.sh sa2va latest` 在 `/tmp/sa2va_env` 创建环境，并将
  `projects/sa2va/.venv` 软链接到该目录。
- `.venv` 由根目录 `.gitignore` 排除。`/tmp` 可能在重启后被清理；此时应重新执行同一脚本，而不是复用其他 Conda 环境。
- InternVL3 属于 `latest` 组；`legacy` 只用于 InternVL2.5 或更早版本。因此 Phase 2A 文档中针对当时首选 Sa2VA-1B 的 `legacy` 结论不适用于本轮已确定的 InternVL3-2B。
- 仓库没有更小的、锁定的 InternVL3-only 推理依赖组。`latest` 会连同公共训练、视频和 notebook 依赖一起安装；为避免手工拼装未经验证的环境，本轮仍使用官方完整锁定组。

锁文件在 Linux/CUDA 12.4 路径上解析出的关键版本为 PyTorch 2.6.0、torchvision 0.21.0、Transformers 4.57.1、PEFT 0.17.1 和 FlashAttention 2.7.3。PyTorch/torchvision wheel 来自项目声明的 CUDA 12.4 index。

安装前的环境变量状态为：`HF_HUB_OFFLINE`、`TRANSFORMERS_OFFLINE`、
`CUDA_VISIBLE_DEVICES`、`PIP_INDEX_URL`、`UV_INDEX_URL` 未设置；
`HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 已设置。这里只记录状态，未读取或记录
任何代理值、token 或密钥，也没有修改网络配置。

## 2. 实际执行命令

安装前检查确认工作树干净，Phase 2A 已提交。实际执行记录如下：

```bash
# 此尝试失败：当前 pip 的 SOCKS 支持不完整，未改变环境。
/home/dqwang/miniconda3/envs/chartground/bin/python -m pip install uv

# setup_env.sh 明确给出的 uv 安装方式；成功安装 uv 0.12.14。
curl -LsSf https://astral.sh/uv/install.sh | sh

# 使用当前仓库 lock 和 InternVL3 对应的 latest 组。
bash setup_env.sh sa2va latest

# latest 未声明 pytest。为执行本项目验收测试，单独固定测试工具版本；
# 此命令不修改 pyproject.toml 或 uv.lock，也不改变模型关键依赖。
/home/dqwang/.local/bin/uv pip install \
  --python projects/sa2va/.venv/bin/python pytest==9.1.1
```

第一次在受限执行沙箱内运行 `setup_env.sh` 时，uv 无法写入用户缓存并报只读文件系统；在保留相同命令、依赖组和 lock 的情况下以允许访问用户 uv 缓存的权限重跑后成功。网络、代理和索引配置均未被修改。

激活环境：

```bash
source projects/sa2va/.venv/bin/activate
```

也可以不激活，直接调用：

```bash
projects/sa2va/.venv/bin/python --version
```

## 3. 已验证版本

| 组件 | 实际版本或结果 |
|---|---|
| 环境入口 | `projects/sa2va/.venv` |
| 物理环境目录 | `/tmp/sa2va_env` |
| 环境磁盘占用 | 7.4 GiB（`du -sh` 显示 `7.4G`） |
| Python | 3.11.16 |
| uv | 0.12.14 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torch CUDA runtime | 12.4 |
| transformers | 4.57.1 |
| peft | 0.17.1 |
| flash-attn | 2.7.3 |
| timm | 1.0.17 |
| einops | 0.8.1 |
| NumPy | 2.3.4 |
| Pillow | 11.3.0 |
| mmengine | 0.10.7 |
| opencv-python-headless / `cv2` | 4.11.0.86 / 4.11.0 |
| sentencepiece | 0.2.1 |
| safetensors | 0.6.2 |
| accelerate | 1.10.1 |
| huggingface_hub | 0.35.3 |
| pytest（测试工具，非 lock 依赖） | 9.1.1 |

轻量导入验证覆盖了 `torch`、`torchvision`、`transformers`、`peft`、
`flash_attn`、`timm`、`einops`、`mmengine`、`cv2`、`sentencepiece`、
`safetensors`、`accelerate` 和 `huggingface_hub`，全部成功。先导入 `torch`
后，`flash_attn_2_cuda` 扩展也可成功导入。

在允许访问 GPU 设备的终端中，PyTorch 报告：

```text
torch.cuda.is_available() = True
torch.cuda.device_count() = 7
GPU 0..6 = NVIDIA GeForce RTX 3090
```

用 GPU 0 创建的单元素 FP32 张量成功，PyTorch allocator 报告 512 bytes；没有创建模型或大型张量。受限沙箱中的首次检查因不能初始化 NVML 曾返回 `False/0`，普通设备权限复核结果应作为本机结论。

## 4. 依赖一致性说明

`uv pip check` 检查 243 个包时报告两项 metadata 不一致：

1. `mmengine` 的发布 metadata 要求 `opencv-python>=3`，锁定环境安装的是 `opencv-python-headless==4.11.0.86`。项目 `pyproject.toml` 为 mmengine 显式覆盖的依赖正是 `opencv-python-headless`，且 `import cv2` 已通过，因此本轮不另装有冲突风险的 GUI 版 OpenCV。
2. `xtuner==0.1.23` 的发布 metadata 声明 Python `<3.11`，而 Sa2VA 项目自身要求 Python `>=3.11,<3.12`。这是当前上游锁定配置内部的 metadata 冲突；本轮图像推理不会调用训练侧 xtuner，但不能把整个环境描述为 `pip check` 零告警。

没有擅自升级、降级或替换任何模型关键依赖。FlashAttention 包、顶层模块和 CUDA 扩展均可导入，本轮不存在 FlashAttention 安装失败。

## 5. 本地 checkpoint 校验

正式约定：ChartGround-Edit 后续推理 CLI 必须通过必填参数
`--checkpoint PATH` 接收 checkpoint 路径，不在代码中写入本机默认路径。上游现有
`projects/sa2va/demo/demo.py` 使用等价的 `--model_path PATH` 参数。

本机只读验证示例：

```text
repo_id: ByteDance/Sa2VA-InternVL3-2B
declared revision: 15837dcaecc304714a1f0f069e74f47e47521c7f
local path: /home/dqwang/Model/Sa2VA-InternVL3-2B
```

校验结果：

- 路径是普通目录，不是 Hugging Face `snapshots/<commit>` 软链接目录。
- 顶层 artifact 共 26 个文件、8,673,308,961 bytes；包含本地下载 metadata 后共 75 个文件、8,673,311,398 bytes，`du -sh` 为 `8.1G`。
- `config.json`、`tokenizer_config.json`、`special_tokens_map.json`、`added_tokens.json` 和 `model.safetensors.index.json` 均存在并可解析。
- tokenizer 为 `Qwen2Tokenizer`；`tokenizer.json`、`vocab.json`、`merges.txt`、独立 `chat_template.jinja` 和 tokenizer remote-code 文件均存在。
- `generation_config.json` 不存在。checkpoint config 和 tokenizer 文件可解析，且官方目录 artifact 总字节与 Phase 2A 查询到的仓库总字节一致；本轮不补下载可选文件。
- config 的 `architectures` 为 `Sa2VAChatModel`，`model_type` 为 `sa2va_chat`，`auto_map` 指向 `configuration_sa2va_chat.Sa2VAChatConfig` 与 `modeling_sa2va_chat.Sa2VAChatModel`；LLM 子配置为 `qwen2`，顶层、LLM 和视觉 dtype 均为 `bfloat16`。这与仓库 InternVL Sa2VA HF 路径一致。
- 两个权重分片均存在且非空，大小分别为 4,986,531,024 和 3,670,290,768 bytes。index 含 1,589 个权重映射；其引用的分片无缺失。
- 只解析两个 safetensors 的 JSON 文件头：分别含 565 和 1,024 个 tensor 条目，条目集合与 index 完全一致，data offsets 均在文件范围内。没有读取 tensor data，也没有加载权重。
- checkpoint 自带 13 个配置、建模、tokenizer、SAM2 和模板 Python 文件，remote code 文件齐全。
- `.cache/huggingface/download/` 中有 24 个零字节 `.lock` 文件和 24 个 metadata 文件；未发现 `.incomplete`、`.part` 或 `.tmp` 文件。锁文件未删除。
- 24 个本地 Hugging Face metadata 文件的 revision 首行全部为
  `15837dcaecc304714a1f0f069e74f47e47521c7f`。因此本地 metadata 可以验证预期 revision，而不只是根据目录名推断。

上述检查证明目录结构、JSON、索引到分片映射和 safetensors 文件头一致，但不是对 8.06 GiB tensor data 的逐字节远端哈希证明。

## 6. 运行前检查与离线约定

每次加载模型前先确认有一张足够空闲的 GPU：

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used \
  --format=csv,noheader
```

当前最终检查时 7 张 GPU 均有约 22.96 GiB 显存被占用，因此**目前不应启动 BF16 单样本推理**。环境、CUDA 通信和 checkpoint 已具备条件，但峰值显存仍必须在 GPU 释放后由 Phase 2B-1 实测，不能从权重文件大小推断。

本地 checkpoint 完整后，未来离线调用应只对当前 shell 临时设置离线变量，并显式传入路径：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
CHECKPOINT=/path/to/Sa2VA-InternVL3-2B

# Phase 2B-1 将提供 ChartGround-Edit CLI；接口必须采用：
#   ... --checkpoint "$CHECKPOINT" ...
# 上游 demo 当前对应参数名为 --model_path。
```

本轮没有运行 `AutoModel.from_pretrained`、`trust_remote_code`、模型推理或训练。

## 7. 复核命令与常见问题

导入与版本复核：

```bash
projects/sa2va/.venv/bin/python -c '
import torch, torchvision, transformers, peft, flash_attn
import timm, einops, mmengine, cv2, sentencepiece
import safetensors, accelerate, huggingface_hub
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(transformers.__version__, peft.__version__, flash_attn.__version__)
'
```

项目回归：

```bash
projects/sa2va/.venv/bin/python -m pytest \
  projects/chartground_edit/tests -q
projects/sa2va/.venv/bin/python -m compileall -q \
  projects/chartground_edit/chartground_edit \
  projects/chartground_edit/scripts \
  projects/chartground_edit/tests
```

常见问题：

- `/tmp/sa2va_env` 不存在：重启可能清理了 `/tmp`，重新执行
  `bash setup_env.sh sa2va latest`，不要把依赖混装进旧 Conda 环境。
- uv 用户缓存只读：保证当前用户能写自己的 uv cache；不要使用 sudo，也不要修改系统 Python。
- `import flash_attn` 或 CUDA 扩展失败：保存 traceback，并记录 Python、Torch、CUDA、GCC 和 nvcc 版本；不要随机换版本。仓库 InternVL 实现静态上支持 `use_flash_attn=False`，但是否采用“继续使用官方 FlashAttention”或“显式关闭”必须另行确认。
- `torch.cuda.is_available()` 在受限容器内为 false：先用普通终端的 `nvidia-smi` 和同一环境解释器复核设备权限，不要据此修改 CUDA 包。
- GPU 显存占用过高：等待或选择一张已经释放的 GPU；不在本阶段启用未经验证的 CPU offload、量化或多卡策略。
