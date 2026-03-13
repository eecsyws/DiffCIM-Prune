# DiffCIM-Prune

## 提高 FeFET-CIM 对 Variation 的鲁棒性

基于 FeFET 存内计算（Compute-In-Memory, CIM）的神经网络推理，提高对 Variation（工艺波动）鲁棒性。

## 项目简介

本项目主要针对 FeFET 存内计算架构，研究如何提高神经网络推理对器件 Variation（工艺波动）的鲁棒性。项目实现了两个核心创新点：

1. **Differential Mapping（差分映射）**：将权重拆分为正负两部分，降低 Variation 对高位权重的影响
2. **Unstructured Pruning（非结构化剪枝）**：通过剪枝减少受 Variation 影响的器件数量

> 项目方法论详细介绍见 [Methods.md](./Methods.md)

## 项目结构

```
Differential_encode/
├── checkpoints/              # 模型权重文件
│   └── vit_tiny_cifar100_finetune.pt
├── config/
│   ├── __init__.py
│   └── config.py              # 配置文件
├── src/
│   ├── __init__.py
│   ├── CIM_Quant.py           # CIM 量化层
│   ├── Fake_Quant.py          # Fake 量化层
│   ├── quant_layers.py       # 重新导出（向后兼容）
│   ├── global_unstructured_pruning.py  # 全局非结构化剪枝
│   ├── model_utils.py        # 模型替换工具
│   ├── model_loader.py       # 模型加载
│   └── inference.py          # 推理逻辑
├── main.py                   # 主入口
├── Methods.md                # 方法论介绍
└── README.md
```

## 安装

```bash
pip install torch torchvision timm numpy tqdm
```

## 配置

所有参数可在 `config/config.py` 中修改：

### 核心参数

| 参数 | 说明 |
|------|------|
| `QUANT_MODE` | 量化模式：`CIM_Quant` 或 `Fake_Quant` |
| `ENCODE_METHOD` | 编码方式：`single` 或 `differential` |
| `WEIGHT_BITS` | 权重量化位数 |
| `INPUT_BITS` | 输入量化位数 |
| `VARIATION_SIGMA_LIST` | Variation sigma 扫描列表 |
| `PRUNING_ENABLE` | 是否启用剪枝 |
| `PRUNING_RATE` | 剪枝率（0.0 - 1.0） |

### CIM_Quant 特有参数

| 参数 | 说明 |
|------|------|
| `ADC_BITS` | ADC 分辨率 |
| `PARALLEL_READ` | 并行读取行数 |
| `USE_PARTIAL_SUM_QUANT` | 是否使用 Partial Sum 量化 |

### Fake_Quant 特有参数

| 参数 | 说明 |
|------|------|
| `NOISE_ENABLE` | 是否注入噪声 |
| `NOISE_MODE` | 噪声模式：`include` 或 `exclude` |
| `INCLUDE_LAYERS` | 包含噪声的层 |
| `EXCLUDE_LAYERS` | 排除噪声的层 |

## 使用方法

运行推理（使用默认配置）：

```bash
python main.py
```

## 两种量化模式

### CIM_Quant

严格模拟 CIM 行为，包括：
- Parallel Read（并行读取）
- ADC 量化
- Partial Sum 量化

### Fake_Quant

简化版 CIM 评估，仅注入 Variation 噪声。

## 两种编码方式

### Single（单阵列）

传统有符号编码，负权重使用补码（two's complement）表示。

### Differential（差分）

将权重拆分为正权重（w+）和负权重（w-）两部分：
- 将 w+ 和 w- 分别存储在不同的阵列中
- 计算结果：output = out_pos - out_neg
- 减少 Variation 对高位的权重的影响

## 核心类

- `CIM_Linear`：CIM 单编码线性层
- `CIM_SM_Linear`：CIM 差分编码线性层
- `NoisyLinear`：Fake 量化线性层
- `NoisyConv2d`：Fake 量化卷积层
