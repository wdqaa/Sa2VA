# Synthetic v1 data card

## 用途与边界

`synthetic_v1` 是 Phase 3A 的 320 条平衡合成数据，用于验证分布、标注语义、split 隔离和后续 balanced-val Prompt benchmark 的工程条件。它不是最终真实图表 benchmark，本阶段没有运行模型、Prompt benchmark 或训练。

## 来源与许可

图像、mask、数值序列和文本均由 `chartground_edit.datasets.synthetic_v1` 使用 NumPy 与 Pillow 本地生成。没有外部数据、字体、模型或下载资产。生成器及其生成 fixture 适用仓库许可。

## 可复现生成

- generator version：`synthetic-v1.0.0`
- schema：`chartground-edit-v1`
- 固定 seed：`20260916`
- 图像尺寸：480 × 320

```bash
projects/sa2va/.venv/bin/python projects/chartground_edit/scripts/generate_synthetic_v1.py --output-dir projects/chartground_edit/data/synthetic_v1 --seed 20260916 --clean --gallery-output projects/chartground_edit/assets/synthetic_v1_gallery.png
```

`data/synthetic_v1` 受根 `.gitignore` 的 `data` 规则保护；320 张图和 mask 不提交 Git，只提交紧凑 gallery。`--clean` 只删除通过安全目录名检查后的生成器自有文件，并拒绝仓库根目录、用户目录、空/宽泛路径及名称不含 `synthetic_v1` 的目录。

## 精确组成

四种 chart type × 四种 referring type 形成 16 个组合，每组合恰好 20 条：train 12、val 4、test 4。每个组合内，train 的四种 action 各 3 条，val/test 各 1 条。

| split | samples | 每种 chart | 每种 referring | 每种 action |
|---|---:|---:|---:|---:|
| train | 192 | 48 | 48 | 48 |
| val | 64 | 16 | 16 | 16 |
| test | 64 | 16 | 16 | 16 |
| total | 320 | 80 | 80 | 80 |

## 难度定义

- `easy`：3 个实体（2 个干扰项），颜色区分明显，线/marker 较粗，采样较稀。
- `medium`：4 个实体（3 个干扰项），存在更相近的颜色，图元更密集。
- `hard`：5 个实体（4 个干扰项），颜色更相近、线和 marker 更细、曲线/band 更密集，更容易出现视觉交叠。

全局计数为 easy 106、medium 107、hard 107，最大差 1。train 为 64/64/64；val 为 21/22/21；test 为 21/21/22（顺序均为 easy/medium/hard）。每个 chart/referring/split 子集都覆盖三档难度。`distractor_count` 由实际 entity 数量计算，不是随机标签。

## 指代唯一性

每个实体拥有唯一 category、颜色、legend label 和趋势/排名描述。生成器根据 `referring_type` 选择实际属性，并断言在该 scene 的实体描述中只匹配一次；证据写入 `generation_metadata.reference_key/value/match_count`。`full_instruction` 与 `referring_expression` 在生成时分别写入，后续不做反向抽取。

## Split 隔离

- 每条记录的 seed、scene ID、content ID 均唯一；content ID 来自底层数值数组哈希。
- 不复用底层序列，不通过只改变 action 创建跨 split 副本。
- style family：train=`amber/birch/cedar`，val=`dune/ember/fjord`，test=`grove/harbor/iris`。
- instruction template family：train=`atlas/boreal/cobalt`，val=`delta/equinox/fable`，test=`granite/helix/indigo`。
- sample/file 名称不含 `train`、`val` 或 `test`；instruction 也不暴露 split。

## Mask 策略

沿用 `annotation_spec_v0.md`。mask 为原图尺寸、mode `L`、仅 0/255 的 PNG。编辑动作和颜色绝不改变目标定义。独立审计还检查 legend box 为零、plot box 外为零、line/scatter marker 中心为前景，以及 confidence band 前景面积不是细中心线。

## 已知局限

- 图表仍是 Pillow 规则图形，不覆盖真实论文中的字体、压缩、OCR、复杂坐标轴和任意布局。
- 趋势词来自有限模板；bar 的 trend 指代使用唯一高度排名。
- 近重复指纹会因共享 axes/layout 报告较多候选，需要结合 content/hash/人工图像判断。
- 当前数据适合工程 benchmark 和受控诊断，不足以支持真实世界泛化结论。
