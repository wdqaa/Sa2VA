# ChartGround-Edit JSONL v1 schema

状态：Phase 3A 冻结  
Schema：`chartground-edit-v1`  
生成器：`synthetic-v1.0.0`  
Mask 语义：`chartground-edit-mask-v0`

每行是一个 UTF-8 JSON object。v1 与 v0 使用独立校验器和 Reader；v0 manifest 不迁移、不改写。未知顶层字段会被拒绝，字段错误不会被静默修复。

## 字段

| 字段 | 类型 | 必需 | 约束 / 允许值 |
|---|---|---:|---|
| `schema_version` | string | 是 | 固定 `chartground-edit-v1` |
| `sample_id` | string | 是 | manifest 内唯一；仅字母、数字、`.`、`_`、`-` |
| `image_path` | string | 是 | 唯一、安全的 manifest 相对路径 |
| `mask_path` | string | 是 | 唯一、安全的相对路径；不同于 image path |
| `split` | enum | 是 | `train`、`val`、`test` |
| `chart_type` | enum | 是 | `line`、`bar`、`scatter`、`confidence_band` |
| `referring_type` | enum | 是 | `category`、`appearance`、`legend`、`trend` |
| `full_instruction` | string | 是 | 完整目标指代、编辑动作和参数；须原样包含 `referring_expression` |
| `referring_expression` | string | 是 | 生成时直接保存；Reader 不从完整指令反向提取 |
| `edit_action` | enum | 是 | `highlight`、`recolor`、`extract`、`remove` |
| `edit_parameters` | object | 是 | 与 action 联合校验，见下节 |
| `target_type` | enum | 是 | `curve`、`bar`、`scatter_series`、`confidence_band`；须匹配 chart type |
| `target_attributes` | object | 是 | 非空生成真值；只用于校验/监督，不自动加入模型 Prompt |
| `mask_semantics_version` | string | 是 | 固定 `chartground-edit-mask-v0` |
| `generator_version` | string | 是 | 固定 `synthetic-v1.0.0` |
| `seed` | integer | 是 | 非负且 manifest 内唯一；不接受 boolean |
| `scene_id` | string | 是 | manifest 内唯一 |
| `content_id` | string | 是 | 由底层数值序列哈希产生，manifest 内唯一 |
| `style_family` | string | 是 | 非空；同一 family 不跨 split |
| `instruction_template_family` | string | 是 | 非空；同一 family 不跨 split |
| `difficulty` | enum | 是 | `easy`、`medium`、`hard` |
| `distractor_count` | integer | 是 | 至少 1；等于实际 entity 数减一 |
| `image_width` | integer | 是 | 大于 0；实际为 480 |
| `image_height` | integer | 是 | 大于 0；实际为 320 |
| `generation_metadata` | object | 是 | 实体描述、唯一性证据、plot/legend box、目标几何和内容哈希 |

## 编辑参数联合约束

| action | 精确字段 | 约束 |
|---|---|---|
| `highlight` | `strength` | 数值，范围 0 到 1 |
| `recolor` | `color` | `#RRGGBB` |
| `extract` | `background` | 固定 `transparent` |
| `remove` | `fill_mode`, `color` | `background_color` 与 `#RRGGBB` |

额外或缺失参数均报错。编辑颜色不参与 mask 定义。

## 文件与 manifest 级约束

- `sample_id`、image path、mask path、`seed`、`scene_id`、`content_id` 各自唯一。
- 图像和 mask 尺寸均为 `image_width × image_height`。
- mask 必须是 Pillow mode `L` 的无损 PNG，只含 0/255，且既非空也非全一。
- 相对路径不能含 `..`，也不能使用绝对路径。
- `line → curve`、`bar → bar`、`scatter → scatter_series`、`confidence_band → confidence_band`。

## Mask 语义

v1 继续使用冻结的 v0 语义：mask 只覆盖真实 plot element，不含标题、轴、刻度、文字或图例代理。line 包含线和 marker、不含 band；scatter 包含目标序列全部 marker；bar 包含目标柱体填充与边框；confidence band 只包含 band 填充区域，不退化为中心线。legend 可用于指代，但 legend glyph 永不进入 mask。

## Reader

`ChartGroundV1Dataset` 使用已存的 `full_instruction` 和 `referring_expression`，不调用 synthetic_v0 的字符串提取器。原 `ChartGroundDataset` 与 v0 校验 API 保持不变。
