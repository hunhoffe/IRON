<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SmolVLA on the NPU

A vision-language-action policy (lerobot's `VLAFlowMatching`): it turns a
camera image, a task's tokens and the robot's state into a chunk of
actions. `observe` runs the image through SmolVLM's vision tower, then the
image's, the task's and the state's rows through the VLM's layers once,
caching each layer's keys and values; the image and task rows attend each
other both ways and not the state's, which attends all. `denoise` runs the
action expert, three quarters as wide, over that cache: its self-attention
layers attend the cache and the chunk's own rows causally, its
cross-attention layers project the cached rows with their own keys and
values. `sample` takes ten Euler steps of `denoise` from noise.

## Scope: one camera, toy shape

The port takes one square image and is checked at a toy shape over drawn
weights; loading the released checkpoint is not done yet. At that shape
the device records only at `Config.precision = HIGHEST`: its 64-row
projections need 256 rows at the default precision.

## Testing

```bash
pytest tests/applications/smolvla/
```

`reference/` checks, device-free, each VLM layer's keys and values after
an observation and the action chunk `sample` returns against lerobot's own,
at a toy shape over random weights. Recording the goldens needs lerobot,
torch and transformers.
