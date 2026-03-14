# DiffCIM-Prune

## 提高 FeFET-CIM 对 Variation 的鲁棒性

基于 FeFET 存内计算（Compute-In-Memory, CIM）的神经网络推理鲁棒性增强框架，面向器件 Variation（工艺波动）带来的精度退化问题，提出从**权重编码、激活编码与模型剪枝**三个层面协同优化的方法。

## 项目简介

本项目主要针对 FeFET 存内计算架构，研究如何提高神经网络推理对器件 Variation（工艺波动）的鲁棒性。由于 FeFET 具有极高的 on/off ratio，关态电流相较开态电流可忽略，因此 Variation 的主要影响来源于存储逻辑值 **“1”** 的器件单元。基于这一物理特性，本项目围绕“**减少受 Variation 影响的 1-state 单元及其高位贡献**”这一核心目标，提出了三个关键方法：

1. **Differential Weight Mapping（权重差分映射）**  
   将有符号权重拆分为正负两部分 W⁺ 和 W⁻，分别计算后再做差，从而避免负数补码表示带来的大量高位”1”。

2. **Unstructured Pruning（非结构化剪枝）**  
   对除 `patch_embed` 和 `head` 之外的 Linear 层权重进行全局非结构化剪枝，增加零权重比例，使映射后存储“0”的单元更多，减少受 Variation 影响的有效器件数量。

3. **Differential Activation Encoding（激活差分编码）**  
   对动态量化后的有符号激活，不再直接采用二补码 bit-serial 输入，而是拆分为 A⁺ 和 A⁻ 两路非负幅值编码，从而减少小负值在二补码下产生的大量高位”1”，降低激活与阵列”1”单元之间的高位交互误差。

> 项目方法论详细介绍见 [Methods.md](./Methods.md)

## 项目结构

```text
DiffCIM-Prune/
├── checkpoints/                         # 模型权重文件
│   └── vit_tiny_cifar100_finetune.pt
├── config/
│   ├── __init__.py
│   └── config.py                        # 配置文件
├── src/
│   ├── __init__.py
│   ├── CIM_Quant.py                     # 严格 CIM 行为仿真
│   ├── Fake_Quant.py                    # 简化版 CIM 评估
│   ├── quant_layers.py                  # 重新导出（向后兼容）
│   ├── global_unstructured_pruning.py   # 全局非结构化剪枝
│   ├── model_utils.py                   # 模型替换工具
│   ├── model_loader.py                  # 模型加载
│   └── inference.py                     # 推理逻辑
├── main.py                              # 主入口
├── Methods.md                           # 方法论介绍
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
| `WEIGHT_ENCODE_METHOD` | 权重编码方式：`twos_complement` 或 `differential` |
| `ACTIVATION_ENCODE_METHOD` | 激活编码方式：`twos_complement` 或 `differential` |
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

---

## 两种量化模式

### 1. CIM_Quant

严格模拟 CIM 行为，包括：

- Parallel Read（并行读取）
- ADC 量化
- Partial Sum 量化
- Bit-serial 输入与权重位平面交互
- Variation 注入

### 2. Fake_Quant

简化版 CIM 评估，重点关注：

- 动态量化
- Bit-level 交互近似
- Variation 注入
- 精度变化趋势验证

---

## 权重编码方式

本项目支持两种权重编码策略，用于在 FeFET-CIM 推理过程中表示量化后的有符号权重。不同编码方式会直接影响阵列中“1”态单元的分布特征，从而影响系统对器件 Variation 的敏感性。

### 1. Two's Complement（二补码）

Two's Complement 是传统的有符号整数表示方式。量化后的正权重和负权重均以统一的二进制补码形式存储到阵列中。

其优点是表示方式标准、实现直接；但对于 FeFET-CIM 而言，负权重在补码表示下通常会引入更多高位 `1`，从而增加 Variation 敏感单元的数量，并放大高位位平面的误差贡献。

### 2. Differential（差分映射）

Differential 编码将带符号权重拆分为正、负两个非负分量，并分别映射到两组独立阵列中进行计算。对于任意量化权重 $W_q$，有：

$$
W_q = W^{+} - W^{-}
$$

其中：

$$
W^{+} = \max(W_q, 0), \qquad
W^{-} = \max(-W_q, 0)
$$

也就是说：

- `W⁺` 表示原始权重中的正值部分；
- `W⁻` 表示原始权重中的负值幅值部分；
- 两部分均为非负数，可采用非负幅值位平面方式进行映射。

在阵列计算阶段，正负两路分别完成乘加运算，最终输出通过差分恢复：

$$
Y = Y^{+} - Y^{-}
$$

其中：

$$
Y^{+} = XW^{+}, \qquad
Y^{-} = XW^{-}
$$

因此，总输出可写为：

$$
Y = XW_q = X(W^{+} - W^{-}) = XW^{+} - XW^{-}
$$

#### 优势

相较于二补码表示，差分映射具有以下优势：

- 避免负权重在补码表示下产生大量高位 `1`；
- 减少阵列中处于 `1` 态的存储单元数量；
- 降低高位位平面对 Variation 的误差放大效应；
- 提升 FeFET-CIM 推理过程的整体鲁棒性。

---

## 激活编码方式

本项目同样支持两种激活编码策略，用于表示动态量化后的有符号输入激活。由于激活在 CIM 中通常以 bit-serial 形式逐位输入，因此其编码方式会直接影响激活位与权重位之间的有效导通交互。

### 1. Two's Complement（二补码）

在 Two's Complement 模式下，动态量化后的有符号激活直接以二补码形式输入阵列。

例如，在 INT8 场景下，一个数值幅度较小的负激活也可能对应大量高位 `1`。在 bit-serial 计算过程中，这些高位 `1` 会与阵列中的权重 `1` 单元产生更多高位 `1 × 1` 交互，从而增强由器件 Variation 引起的误差传播。

### 2. Differential（差分编码）

为减少负激活在二补码表示下产生的大量高位 `1`，本项目采用 Differential Activation Encoding。对于量化后的激活整数 $A_q$，定义：

$$
A_q = A_q^{+} - A_q^{-}
$$

其中：

$$
A_q^{+} = \max(A_q, 0), \qquad
A_q^{-} = \max(-A_q, 0)
$$

这意味着原始的有符号激活不再通过单路二补码位平面输入，而是被拆分为两路非负幅值分支：

- `A_q⁺`：正激活分支
- `A_q⁻`：负激活幅值分支

若输入位宽为 `b` bit，则两路分支分别采用 `(b - 1)` bit 非负幅值 bit-serial 输入。以 INT8 为例：

- 原始表示：单路 `8-bit signed` 输入
- 差分表示：两路 `7-bit non-negative magnitude` 输入

在计算过程中，有：

$$
A_q W = (A_q^{+} - A_q^{-})W = A_q^{+}W - A_q^{-}W
$$

#### 优势

相较于直接采用二补码输入，差分激活编码具有以下优势：

- 避免小负值在补码表示下引入大量高位 `1`；
- 减少激活侧高位有效导通次数；
- 降低激活位与权重位之间的高位 `1 × 1` 交互；
- 减弱权重侧 FeFET Variation 对最终输出的误差放大。

---

