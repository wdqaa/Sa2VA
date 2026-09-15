# ChartGround-Edit JSONL v0 schema

每行是一个 UTF-8 JSON object。规范版本为 `chartground-edit-v0`，由 `metadata.schema_version` 记录。未知顶层字段会被校验器拒绝，以避免拼写错误静默进入数据。

| 字段 | 类型 | 必需 | 允许值 / 约束 | 示例 |
| --- | --- | --- | --- | --- |
| `sample_id` | string | 是 | 非空；在一个 manifest 中唯一 | `cge_line_category_00` |
| `image_path` | string | 是 | 非空；相对 manifest 或绝对文件路径 | `images/cge_line_category_00.png` |
| `mask_path` | string | 是 | 非空；指向单通道二值 PNG | `masks/cge_line_category_00.png` |
| `width` | integer | 是 | `> 0`；不接受 boolean | `480` |
| `height` | integer | 是 | `> 0`；不接受 boolean | `320` |
| `chart_type` | string enum | 是 | `line`、`bar`、`scatter`、`confidence_band` | `line` |
| `referring_type` | string enum | 是 | `category`、`appearance`、`legend`、`trend` | `legend` |
| `instruction` | string | 是 | 非空自然语言指令 | `把 accuracy 对应的曲线改成红色` |
| `target_type` | string enum | 是 | `curve`、`bar`、`scatter_series`、`confidence_band` | `curve` |
| `target_attributes` | object | 是 | JSON object；保存类别、颜色、趋势、序列名等生成真值 | `{"series_name": "accuracy", "color": "#2878b5"}` |
| `edit_action` | string enum | 是 | `highlight`、`recolor`、`extract`、`remove` | `recolor` |
| `split` | string enum | 是 | `train`、`val`、`test` | `train` |
| `generator_seed` | integer | 是 | `>= 0`；不接受 boolean | `20260915` |
| `metadata` | object | 是 | JSON object；必须含 `schema_version`、`generator_version` 和 `source` | `{"schema_version": "chartground-edit-v0", "generator_version": "synthetic-v0", "source": "synthetic"}` |

## 跨字段约束

- `target_type` 必须与 `chart_type` 对应：`line → curve`、`bar → bar`、`scatter → scatter_series`、`confidence_band → confidence_band`。
- 校验 manifest 时，`sample_id` 必须唯一。
- 启用文件校验时，`image_path` 与 `mask_path` 必须存在；图像与 mask 尺寸必须等于 `width × height`。
- mask 必须为 Pillow 模式 `L`、PNG 格式、非空，且只能含 `0` 和 `255`。

## 完整示例

```json
{"sample_id":"cge_line_legend_00","image_path":"images/cge_line_legend_00.png","mask_path":"masks/cge_line_legend_00.png","width":480,"height":320,"chart_type":"line","referring_type":"legend","instruction":"把图例中 accuracy 对应的曲线改成红色","target_type":"curve","target_attributes":{"series_name":"accuracy","color":"#2878b5","target_index":0,"trend":"increasing"},"edit_action":"recolor","split":"train","generator_seed":20260915,"metadata":{"schema_version":"chartground-edit-v0","generator_version":"synthetic-v0","source":"synthetic"}}
```

