# 音乐旋律线几何相似性度量及其在曲风区分中的应用

---

## 摘要

> 一段话概括：做了什么（将旋律抽象为三维空间曲线）、用什么方法（Hausdorff 距离）、在什么数据上验证了（ADL Piano MIDI 数据集）、得到了什么主要结果（距离统计、MDS 聚类、分类精度）。

**关键词**：旋律相似度；Hausdorff 距离；几何度量；曲风分类；MIDI 分析

---

## 一、问题重述

### 1.1 问题背景
- 音乐信息检索（MIR）中旋律相似度量的重要性
- 已有方法概述：编辑距离、N-gram、几何方法
- Hausdorff 距离在图像匹配中的成功 → 启发将其引入旋律空间
- 引用 Di Lorenzo & Maio (2006)、Romming & Selfridge-Field (2007)

### 1.2 问题提出
- 将旋律线抽象为三维空间中的曲线（时间、音高、音量）
- 用 Hausdorff 距离量化不同旋律线之间的几何相似性
- 探讨这种度量方式在区分音乐曲风方面的有效性

---

## 二、问题分析

### 2.1 旋律的几何表示
- 为什么选择 (t, pitch, velocity) 三个维度
- 三维空间曲线相比二维表示的优势（音量信息也编码进来）
- 离散音符点 → 分段线性曲线的合理性

### 2.2 相似性度量的选择依据
- Hausdorff 距离对点集大小不敏感 → 适合长度不同的旋律
- 不需要对齐（alignment-free）→ 计算简洁
- 几何直观性

### 2.3 曲风区分的可行性
- 不同曲风在音高分布、节奏密度、力度变化上的系统性差异
- 这些差异是否足以在几何上体现？

---

## 三、模型假设

1. **旋律可离散化**：一段旋律可表示为有限个离散音符的序列
2. **音符三要素**：每个音符由起始时间、音高、音量唯一确定
3. **忽略音色和和声**：仅考虑单声道旋律线（纯钢琴 MIDI 保证这一点）
4. **曲线分段线性**：相邻音符之间用直线段连接
5. **曲风标签可靠**：ADL 数据集中的 genre 分类视为真值

---

## 四、符号说明

| 符号 | 含义 |
|------|------|
| $M=\{p_1,...,p_n\}$ | 旋律点集，$p_i=(t_i, pitch_i, vel_i)$ |
| $H(A,B)$ | 双向 Hausdorff 距离 |
| $h(A,B)$ | 有向 Hausdorff 距离 |
| $d(a,b)$ | 欧氏距离 |
| $C(t)$ | 旋律曲线（分段线性插值） |
| $\overline{M}$ | 重采样后的旋律点集 |

---

## 五、模型建立与求解

### 5.1 旋律曲线的三维空间表示

#### 5.1.1 音符到空间点
- MIDI 格式简介：pitch 编号 0-127（C-1 ~ G9）
- velocity 范围 0-127
- 时间单位为拍或秒
- 定义 $p_i=(t_i, \text{pitch}_i, \text{vel}_i) \in \mathbb{R}^3$

#### 5.1.2 归一化（标准化）
- 时间归一化到 $[0,1]$（不同长度的旋律可比）
- 音高用 MIDI 编号（或可选对数频率 $\log_2(f/440)$）
- 音量归一化到 $[0,1]$

#### 5.1.3 曲线构造
- 将离散点按时间顺序连接成分段线性曲线
- 弧长参数化

### 5.2 Hausdorff 距离的定义与离散计算

#### 5.2.1 数学定义

有向 Hausdorff 距离：

$$h(A,B) = \sup_{a \in A} \inf_{b \in B} \|a-b\|$$

双向 Hausdorff 距离：

$$H(A,B) = \max\{h(A,B), h(B,A)\}$$

#### 5.2.2 离散化与重采样
- 连续曲线 → 沿弧长均匀采样 N 个点（N=500）
- 原因：不同旋律音符数量不同，重采样使点密度一致

#### 5.2.3 算法实现
- K-d 树加速最近邻搜索
- 时间复杂度：重采样 $O(N)$ + 最近邻 $O(N\log N)$
- 伪代码

#### 5.2.4 加权 Hausdorff 距离（可选变体）
- 对不同维度赋予不同权重
- 例如：强调音高轮廓，弱化时间伸缩

### 5.3 数值实验

#### E1：合成数据验证
- 构造已知结构的旋律：音阶、琶音、随机游走
- 验证：自身距离 = 0；加扰版距离小；不同结构距离大
- 结果表格/热力图

#### E2：同类 vs 异类曲风距离分析
- 数据集：ADL Piano MIDI（Classical / Jazz / Rock / Pop / Blues）
- 同类曲风内两两计算 Hausdorff 距离 vs 异类之间
- 统计检验（Mann-Whitney U 检验）
- 箱线图展示分布

#### E3：距离矩阵与 MDS 可视化
- 计算所有曲风间平均距离矩阵
- 多维缩放（MDS）降到 2D
- 散点图：不同颜色标注不同曲风，观察聚类效果

#### E4：基于 Hausdorff 距离的 1-NN 分类
- 留一法（leave-one-out）
- 用 Hausdorff 距离找最近邻 → 预测曲风标签
- 混淆矩阵 + 分类精度
- 与随机基线对比

### 5.4 结果分析

- 汇总表：各实验的主要数值结果
- MDS 聚类图
- 分类混淆矩阵
- 讨论哪些曲风容易混淆（如 Blues 和 Rock），为什么

---

## 六、模型评价

### 6.1 优点
- 几何直观，无需特征工程
- 对旋律长度不敏感
- 计算高效（k-d 树加速）
- 可扩展到三维以上（加入节奏、和声等特征）

### 6.2 局限性
- 忽略旋律的方向性（时间顺序仅通过曲线形状隐式编码）
- 对装饰音等局部细节敏感
- 纯几何度量与人类感知可能存在差距
- 仅使用单声道旋律线，未利用多声部信息

### 6.3 改进方向
1. 引入动态时间规整（DTW）的思想处理时间伸缩
2. 加权 Hausdorff 距离调节各维度贡献
3. 结合持久同调（persistent homology）提取拓扑特征（Callet-Feltz, 2025）
4. 在更大规模数据集上验证

---

## 参考文献

[1] Di Lorenzo, P. & Maio, G. The Hausdorff metric in the melody space: A new approach to melodic similarity. ISMIR, 2006.

[2] Romming, C. A. & Selfridge-Field, E. Algorithms for polyphonic music retrieval: The Hausdorff metric and geometric hashing. ISMIR, 2007.

[3] Mazzola, G., Noll, T. & Lluis-Puebla, E. Perspectives in Mathematical and Computational Music Theory.

[4] Callet-Feltz, V. Persistent Homology and Discrete Fourier Transform. Springer, 2025.

[5] Tzanetakis, G. & Cook, P. Musical genre classification of audio signals. IEEE Trans. Speech and Audio Processing, 2002.

[6] Ferreira, L. N., Lelis, L. H. S. & Whitehead, J. Computer-Generated Music for Tabletop Role-Playing Games. AIIDE, 2020.

---

## 附录

### A. 数据说明
- ADL Piano MIDI 数据集（11,076 首纯钢琴 MIDI，18 类曲风）

### B. 代码结构
- hausdorff.py：Hausdorff 距离核心算法
- midi_parser.py：MIDI 解析与旋律提取
- synthetic.py：合成数据生成
- visualization.py：可视化工具
- experiments.py：实验流程
- main.py：主入口

### C. 补充图表
- 更多 3D 旋律曲线示例
- 不同重采样点数的敏感性分析
