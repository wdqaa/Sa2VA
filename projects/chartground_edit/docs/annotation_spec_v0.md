# ChartGround-Edit v0 标注规范

状态：冻结（Phase 1A）  
Schema 版本：`chartground-edit-v0`

## Mask 文件格式

- 所有 mask 使用与原图宽高完全相同的单通道无损 PNG。
- mask 像素值只允许 `0`（背景）和 `255`（目标）。
- v0 的主标注格式是 PNG mask。polygon 和 RLE 暂不作为主格式；后续如有需要，可由 PNG mask 派生。

## 目标语义

### 曲线序列

- 包含绘图区内属于目标序列的曲线线条。
- 包含绘图区内属于目标序列的 marker。
- 不包含 confidence band。
- 不包含图例中的样例线和 marker。
- 不包含坐标轴、网格线和文字。

### Confidence band

- `confidence_band` 独立于曲线序列标注。
- mask 包含绘图区内 band 的填充区域，不包含图例样例、坐标轴、网格线和文字。
- band 与曲线发生像素重叠时，band mask 仍按 band 的完整几何填充区域标注；曲线 mask 则只包含曲线线条和 marker。

### 柱状图

- 目标 mask 包含目标柱子的填充区域和边框。
- 不包含坐标轴、网格线、文字或图例样例。

### 散点序列

- 目标 mask 包含对应序列在绘图区内的全部散点 marker。
- 不包含图例 marker、坐标轴、网格线或文字。

## 指代表达

- 图例可以作为自然语言指代依据，但图例本身不进入目标 mask。
- v0 支持 `category`、`appearance`、`legend` 和 `trend` 四类指代。
- 一条 annotation 对应一个确定目标；同一目标含多个图元时，mask 是这些图元的 union mask。

## 编辑语义

- `highlight`：突出 mask 内目标。
- `recolor`：只改变 mask 内目标颜色。
- `extract`：输出透明背景 RGBA 目标图，alpha 由 mask 决定。
- `remove`：使用背景色填充 mask 内区域。v0 不宣称进行内容恢复或图像修复。

## 一致性要求

- 图像与 mask 的 resize、crop、horizontal flip 必须复用同一组几何参数。
- 图像允许使用 bilinear 或 bicubic 插值；mask 只能使用 nearest-neighbor 插值。
- 任意变换后 mask 仍必须是单通道且像素值仅为 `0` 和 `255`。

