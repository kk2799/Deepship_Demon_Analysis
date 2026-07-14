# DeepShip DEMON 数据集

Deepship数据集的舰船辐射噪声 DEMON（Detection of Envelope Modulation on Noise）调制谱数据集，用于训练模型分析螺旋桨叶数与转速。

## 数据规模

| 项 | 数值 |
|---|---|
| 总样本数 | 609 |
| 清晰样本（有标注） | 504 |
| 不清晰样本（无真值） | 105 |
| 桨叶数 Z 分布 | 3 叶: 88, 4 叶: 346, 5 叶: 45, 6 叶: 13 |
| 轴频 fs 范围 | 0.76 – 7.45 Hz（均值 2.48 Hz） |
| 对应转速 RPM | 46 – 447 RPM |

## 目录结构

每个样本一个子目录，以样本 ID 命名：

```
deepship_selected/
├── 0_1/
│   ├── 0_1.json          # 标注标签
│   ├── 0_1.npy           # 深度学习提取的 DEMON 谱（1D 数组）
│   └── 0_1_trad.npy      # 传统算法计算的 DEMON 谱（1D 数组）
├── 0_10/
│   ├── 0_10.json
│   ├── 0_10.npy
│   └── 0_10_trad.npy
└── ... (609 个样本)
```

## 文件说明

### `<id>.json` — 标注标签

```json
{
  "fs": 3.984,                    // 轴频 (Hz)，手动标注的最低峰频率
  "Z": 3,                          // 桨叶数
  "rpm": 239.04,                   // 转速 (RPM = 60 × fs)
  "peaks": [                       // 手动标注的谱峰列表
    {"freq": 3.984, "amp": null}, // 频率 (Hz)，幅值未记录
    {"freq": 7.929, "amp": null},
    {"freq": 11.965, "amp": null}
  ],
  "is_clear": true                 // true=清晰（有真值），false=不清晰（无可靠真值）
}
```

**字段说明：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `fs` | float | 轴频 (Hz)。手动标注的最低谱峰频率。轴频在谱图上不一定清晰可见，可能由倍频间距推断 |
| `Z` | int | 桨叶数 (3-6)。清晰样本有标注，不清晰样本为 0 |
| `rpm` | float | 转速 (RPM)。等于 60 × fs |
| `peaks` | list | 手动标注的谱峰位置。每个元素含 `freq` (Hz) 和 `amp`（未记录，为 null） |
| `is_clear` | bool | 是否为清晰样本。不清晰样本的 fs/Z 不可靠，用于评估模型的拒绝能力 |

### `<id>.npy` — 深度学习提取的 DEMON 谱

- **Shape**: (500,) 一维数组
- **Dtype**: float64
- **频率范围**: 0 – 50 Hz
- **分辨率**: 0.1 Hz（500 个点）
- **归一化**: 未归一化（原始幅值，最大值约 0.1-0.5）
- **来源**: 深度学习方法从原始音频中提取的调制谱

### `<id>_trad.npy` — 传统算法 DEMON 谱

- **Shape**: (499,) 一维数组
- **Dtype**: float32
- **频率范围**: 0 – 49.8 Hz
- **分辨率**: 0.1 Hz（499 个点）
- **归一化**: 未归一化（原始 PSD 幅值，最大值约 1e-4 量级）
- **算法**: 带通滤波 (50-5000 Hz) → Hilbert 包络 → Welch PSD

## 物理关系

```
fs = RPM / 60          (轴频)
fb = Z × fs            (叶频 / 桨叶通过频率)
RPM = 60 × fs          (转速)
Z = fb / fs            (桨叶数)
```

DEMON 谱中，轴频 fs 及其谐波 (k×fs) 构成等间距峰列；叶频 fb 是其中最强峰之一（但不一定是最强峰）。

## 使用方式

### 加载数据

```python
import json
import numpy as np

sample_id = "0_1"
base = f"deepship_selected/{sample_id}/{sample_id}"

# 加载标签
with open(f"{base}.json", encoding="utf-8") as f:
    label = json.load(f)

# 加载 DL 谱
dl_spec = np.load(f"{base}.npy")

# 加载传统谱
trad_spec = np.load(f"{base}_trad.npy")

# 频率轴
freqs_dl = np.arange(len(dl_spec)) * 0.1     # 0, 0.1, 0.2, ..., 49.9 Hz
freqs_trad = np.arange(len(trad_spec)) * 0.1 # 0, 0.1, 0.2, ..., 49.8 Hz

# 归一化（训练前建议归一化到 [0, 1]）
dl_spec_norm = dl_spec / dl_spec.max()
trad_spec_norm = trad_spec / trad_spec.max()
```

### 统计分析

```python
import os, json
from collections import Counter

dirs = sorted(os.listdir("deepship_selected"))
Z_dist = Counter()
for d in dirs:
    with open(f"deepship_selected/{d}/{d}.json", encoding="utf-8") as f:
        r = json.load(f)
    if r["is_clear"]:
        Z_dist[r["Z"]] += 1

print("桨叶数分布:", dict(sorted(Z_dist.items())))
# {3: 88, 4: 346, 5: 45, 6: 13}
```

## 注意事项

1. **DL 谱与传统谱的点数不同**：DL 谱 500 点，传统谱 499 点。使用时注意频率轴长度
2. **传统谱幅值极小**（~1e-4 量级）：Welch PSD 的原始输出，使用前需归一化到 [0, 1]
3. **不清晰样本 (is_clear=false)**：fs/Z 标注为 0 或不可靠，不适合作为训练真值，可用于评估模型的不确定性表达能力
4. **轴频不一定可见**：部分样本的轴频基频在谱图上不显著，需要通过高阶谐波间距反推
5. **本数据集不含原始音频 (.wav)**：仅包含提取后的 1D 谱数组和标注标签

## 引用

如果使用本数据集，请引用相关论文（待补充）。

## 许可

本数据集仅供学术研究使用。
