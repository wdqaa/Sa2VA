# ChartGround-Edit editing v0

Phase 1B 的编辑后端只依赖输入图像和二值 mask，不绑定 Sa2VA、annotation 或训练框架。

## 统一接口

```python
from chartground_edit.editing import edit

edited = edit(image, mask, action, parameters)
```

- `image`：Pillow `RGB` 图像。
- `mask`：Pillow `L`/`1` 图像或二维 NumPy array；接受 bool、`{0, 1}` 或 `{0, 255}`。
- `action`：`highlight`、`recolor`、`extract`、`remove`。
- `parameters`：可选 mapping。未知参数会报错，避免拼写错误被静默忽略。
- 非二值 mask、尺寸不一致或非 RGB 图像会明确报错。
- 空 mask 在 v0 中明确拒绝并抛出 `EmptyMaskError`。全一 mask 合法。

## 动作语义

### `highlight`

参数：`strength`，默认 `0.65`，范围 `[0, 1]`。

mask 内像素保持原样；mask 外像素按强度向较暗的灰度版本插值。因此该操作明确允许并预期修改 mask 外区域。`strength=0` 是完整 no-op；全一 mask 也是 no-op。

### `recolor`

参数：`color`，默认 `(230, 57, 70)` / `#E63946`。支持 `#RGB`、`#RRGGBB` 或三个 `[0, 255]` 整数。

只替换 mask 内像素的色相/饱和度，使用原像素的 HSL lightness 生成新颜色，以尽量保留线条、填充和抗锯齿的明暗结构。mask 外字节级不变。全一 mask 会处理整图。

### `extract`

不接受动作参数。返回与输入同尺寸的 RGBA 图像；RGB 通道保留原图，mask 内 alpha 为 255，mask 外 alpha 为 0。全一 mask 的 alpha 全为 255。

### `remove`

参数：

- `fill_mode`：`color`（默认）或 `neighbor`。
- `color`：固定填充色；在 `neighbor` 无可用邻域时作为 fallback，默认白色。
- `neighbor_radius`：邻域半径，默认 5，整数范围 `[1, 50]`。

`color` 直接以指定 RGB 填充 mask。`neighbor` 对 mask 做确定性方形膨胀，并取膨胀环带中 RGB 的逐通道中位数；环带为空（例如全一 mask）时使用 fallback 色。两种模式都只修改 mask 内像素，不恢复被遮挡的网格线或内容，也不使用生成式修复。

