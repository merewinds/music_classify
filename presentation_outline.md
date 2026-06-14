# 答辩PPT大纲

---

## 一、数据处理

### 1. MIDI 文件格式
- MIDI 存的是演奏指令，不是音频波形
- 核心事件：Note On（按键，携带音高+力度）
- 音高 pitch 0-127，每差 1 = 一个半音
- 力度 velocity 0-127，越重越响

### 2. 旋律提取

**Skyline 提取（主旋律近似）**
同一时刻可能有多个音符（和弦），只取音高最高的那个：
$$(p_\tau, v_\tau) = \arg\max_{(p,v)\in C_\tau} (p, v)$$

**时间重采样**
首末音时间对齐 → 均匀采样 N 个点 → 分段线性插值：
$$t_i = \frac{\tau_i - \tau_{\min}}{\tau_{\max} - \tau_{\min}} \in [0, 1]$$

**三维旋律点**
音高减中位数（移调不变），力度加权重：
$$\boldsymbol{x}_i(w_v) = \left(t_i,\; \frac{p_i - \operatorname{median}(p)}{12},\; w_v\frac{v_i - 64}{32}\right) \in \mathbb{R}^3$$

### 3. 样本选择（防泄漏）
- 数据来源：ADL Piano MIDI（五类，共 5317 个候选文件）
- 问题：同一首歌有多个版本，会虚高精度
- 三重去重：文件名规范化、SHA-256、旋律指纹
  - 任一匹配就归为同一连通组（Connected Component）
  - 跨曲风冲突组整组剔除（75 组，187 文件）
- 最终：每类 100 首，共 500 首独立作品

---

## 二、Hausdorff 距离及其变体

### 1. 数学定义

设两条旋律线点集 $A = \{\boldsymbol{a}_i\}_{i=1}^m$, $B = \{\boldsymbol{b}_j\}_{j=1}^n$

**有向 Hausdorff 距离**（Directed Hausdorff Distance）：
$$h(A,B) = \max_{\boldsymbol{a} \in A} \min_{\boldsymbol{b} \in B} \|\boldsymbol{a} - \boldsymbol{b}\|_2$$
A 的每个点找 B 里最近的点，取这些最近距离的最大值。

**双向 Hausdorff 距离**（Bidirectional Hausdorff Distance）：
$$H(A,B) = \max\{h(A,B), h(B,A)\}$$
两个方向各算一次，取最大值。

**95% 分位 Hausdorff 距离**（95% Quantile Hausdorff Distance, Q95-HD）：
$$H_{0.95}(A,B) = \max\left\{Q_{0.95}\!\left(\min_{b\in B}\|a-b\|_2\right),\; Q_{0.95}\!\left(\min_{a\in A}\|b-a\|_2\right)\right\}$$
取分位数代替最大值，去掉最极端的 5%。

**Modified Hausdorff Distance（MHD）**（Dubuisson \& Jain, 1994）：
$$h_{\mathrm{MHD}}(A,B) = \frac{1}{|A|} \sum_{\boldsymbol{a} \in A} \min_{\boldsymbol{b} \in B} \|\boldsymbol{a} - \boldsymbol{b}\|_2$$
$$H_{\mathrm{MHD}}(A,B) = \max\{h_{\mathrm{MHD}}(A,B), h_{\mathrm{MHD}}(B,A)\}$$
取平均值代替最大值，离群点影响小，论文主模型。

### 2. max-Hausdorff 的问题
- 一个离群点就能拉爆距离（由 max 决定，不关心整体）
- 合成实验：单点扰动下 max-HD 线性上升，MHD 几乎不变

### 3. 三种变体对比

| 变体 | 逻辑 | 对离群点 | 论文结果 |
|------|------|---------|---------|
| max-HD | 取最近距离的 max | 极其敏感 | 34.4% |
| Q95-HD | 取最近距离的 95% 分位数 | 稳健 | 34.4% |
| MHD | 取最近距离的 mean | 最稳健 | 40.8% |

---

## 三、模型原理与实验

### 1. 模型体系

| 类别 | 模型 | 作用 |
|------|------|------|
| 几何距离 | 多参数 MHD + K-NN | 主要几何模型，内层选 N/wv/K |
| 几何距离 | 相位对齐 RMSE（Root Mean Square Error） | 严格逐点对应，零时间扭曲 |
| 几何距离 | 多变量 DTW（Dynamic Time Warping） | 允许时间轴伸缩对齐 |
| 描述符 | 多项逻辑回归（Multinomial Logistic Regression） | 检验线性可分性 |
| 描述符 | 随机森林（Random Forest, RF） | 非线性描述符，最强单模型 |
| 融合 | MHD + RF 概率加权融合 | 检验几何与描述符是否互补 |

**相位对齐 RMSE**：两曲线采样网格相同，逐点算欧氏距离再取均方根：
$$\text{RMSE}(A,B) = \sqrt{\frac{1}{N}\sum_{i=1}^N \|\boldsymbol{a}_i - \boldsymbol{b}_i\|^2}$$

**多变量 DTW**（Sakoe \& Chiba, 1978）：
设 $P = (\boldsymbol{p}_1,\ldots,\boldsymbol{p}_m)$, $Q = (\boldsymbol{q}_1,\ldots,\boldsymbol{q}_n)$，动态规划：
$$D(i,j) = \|\boldsymbol{p}_i - \boldsymbol{q}_j\|_2 + \min\{D(i-1,j), D(i,j-1), D(i-1,j-1)\}$$
允许局部时间变形，用 Sakoe-Chiba 窗口限制对齐范围。

**概率融合**（Probabilistic Fusion）：
$$\boldsymbol{p}_{\mathrm{fusion}}(x) = \alpha \boldsymbol{p}_{\mathrm{RF}}(x) + (1-\alpha) \boldsymbol{p}_{\mathrm{MHD}}(x)$$
$\alpha \in \{0.25, 0.5, 0.75\}$，内层选择。

### 2. 实验设计

**验证框架**：Nested Group 5-fold Cross-Validation（嵌套分组交叉验证）
- 外层：Stratified Group 5-Fold（分层+分组，防止同曲跨折）
- 内层：在每折外训练集内，3 个随机种子 × 3-fold 选参数
- 参数选择以平均 Balanced Accuracy（平衡准确率）为标准
- 外层测试折只评估一次，不参与任何参数选择

**参数搜索范围**：
- $N \in \{36, 48, 96\}$（重采样点数）
- $w_v \in \{0, 0.1, 0.25, 0.5\}$（力度权重）
- $K \in \{1, 3, 5, 7, 9\}$（K-NN 近邻数）
- $\alpha \in \{0.25, 0.5, 0.75\}$（融合权重）

**评估指标**：
- Accuracy（准确率）
- Balanced Accuracy（平衡准确率，均衡各类样本权重）
- Macro F1-Score（宏平均 F1）
- Confusion Matrix（混淆矩阵）

### 3. 统计检验（怎么判断结果是不是运气）

**Bootstrap 置信区间**
给精度算一个波动范围。在 500 个测试结果上有放回抽样 2000 次，取 2.5%-97.5% 分位数：
- 例：RF 55.6%（95% CI: 51.2%-60.0%）
- 含义：换一批类似数据，95% 的概率结果落在这个区间里
- 如果区间下限仍然远高于基线，结论可靠

**McNemar 配对检验**
比较两个模型谁更好。只看两个模型有分歧的样本：
- b = RF 对但 MHD 错的样本数
- c = MHD 对但 RF 错的样本数
- 若 b 远大于 c，则 RF 显著更好
- 论文：RF vs MHD：p < 0.0001（显著）
- 论文：RF vs 融合：p = 0.3222（不显著，差异可能是随机波动）

**Holm 校正**
比较多组时显著门槛要更严格，避免多次比较产生的假阳性。

**Permutation Test（置换检验）**
检验"同类距离 < 异类距离"是否显著：
- 把曲风标签随机打乱重复 9999 次
- 每次重新算类内均值 - 类间均值的差值
- p = (比真实值更极端的情况 + 1) / (9999 + 1)
- 结果：p = 0.0001（显著）

**Pair AUC（配对 AUC）**
随机抽一对同类 + 一对异类，同类距离更小的概率：
- AUC = 0.5 → 纯随机
- AUC = 1.0 → 完全可区分
- 论文结果：0.5645 → 略好于随机，但分布高度重叠

### 4. 实验结果

| 模型 | 准确率 | 95% CI |
|------|--------|--------|
| max-HD（逐曲 Min-Max 归一化） | 31.2% | 27.6-35.0 |
| max-HD（相对音高 TPV） | 34.4% | 30.8-38.4 |
| Q95-HD | 34.4% | 30.6-38.4 |
| 固定 MHD | 38.4% | 34.4-42.6 |
| **多参数 MHD** | **40.8%** | **36.8-45.0** |
| 相位对齐 RMSE | 23.6% | 20.4-27.0 |
| 多变量 DTW | 35.2% | 31.2-39.0 |
| 逻辑回归 | 52.2% | 48.0-56.6 |
| **随机森林** | **55.6%** | **51.2-60.0** |
| MHD + RF 融合 | 54.0% | 49.8-58.2 |
| 随机猜测（基线） | 20.0% | — |

### 5. 关键发现
- 类内 MHD 均值 0.3196，类间 0.3487，差值 0.0292（p = 0.0001）
- 但配对 AUC = 0.5645，分布重叠大 → 有统计证据但效应量弱
- 力度有用：五个外层折都选了 wv = 0.5
- 不是越密越好：48 点优于 96 点
- 融合不显著优于 RF（p = 0.3222）

### 6. 随机种子稳定性（Repeated Validation）
- 五组不同外层随机种子重复完整流程
- 多参数 MHD 平均 41.12% ± 1.45%
- 随机森林平均 56.24% ± 1.24%
- 融合平均 55.08% ± 1.06%
- 五次中排序稳定：RF > 融合 > MHD

---

## 四、结论

1. Hausdorff 距离确实能区分曲风，显著优于随机（40.8% > 20%，p < 0.0001）
2. 但区分能力有限，AUC = 0.56，分布重叠大，不能单独做高精度分类
3. 统计特征（节奏、力度、音级）更强，RF 达 55.6%
4. Hausdorff 的价值在于可解释的旋律形状相似性，适合做几何佐证而非独立分类器
