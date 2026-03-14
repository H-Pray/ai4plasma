# RK-PINN 1D 电晕放电仿真算法流程梳理

本文档总结了基于隐式龙格-库塔物理信息神经网络（RK-PINN）求解一维等离子体电晕放电问题的核心逻辑与流程。

---

## 1. 算法核心原理
该算法旨在求解耦合的**泊松方程**（静电场）与**漂移-扩散-反应方程**（电子演化）。

* **时间离散化**：采用 $q$ 阶隐式龙格-库塔（IRK）方案。与标准 PINN 不同，网络输入仅为空间坐标 $r$，输出为该时间步内所有 RK 阶段的解。
* **硬边界约束**：通过数学构造（而非 Penalty Loss）强制满足 Dirichlet 边界条件。
* **物理驱动**：利用自动微分计算空间梯度，并将物理定律（PDEs）转化为优化目标。

---

## 2. 算法详细流程

### 第一阶段：初始化与参数配置
1.  **加载权重**：从 `ButcherTable/` 加载 $q$ 阶 Butcher 矩阵（RK 权重）。
2.  **物理建模**：设定半径 $R$、电压 $V_0$、气压 $P$、温度 $T$ 以及二次电子发射系数 $\gamma$。
3.  **空间采样**：在 $r \in [0, 1]$ 范围内进行配点采样（支持 `uniform`、`lhs` 或 `random` 模式）。

### 第二阶段：网络架构 (Corona1DRKNet)
1.  **前向计算**：输入 $r$，网络输出原始张量。
2.  **边界处理**：
    * **电势 $\Phi$**：通过公式 `out * ((r - R)*r) + (1 - r/R)*V0` 强制使 $r=0$ 时 $\Phi=V_0$，$r=R$ 时 $\Phi=0$。
    * **电子密度 $N_e$**：输出各个 RK 阶段的密度分布。

### 第三阶段：损失函数构造 (Loss Terms)
模型通过最小化以下残差之和进行训练：
* **泊松残差**：$\nabla^2 \Phi + \frac{e(N_p - N_e)}{\epsilon_0} = 0$。
* **电子演化残差**：基于 IRK 公式，结合漂移（$\mu_e E N_e$）、扩散（$D_e \nabla N_e$）和电离源项（$\alpha \mu_e |E| N_e$）。
* **边界残差**：在阴极边界处约束电子通量，满足 $\gamma$ 二次发射物理过程。

### 第四阶段：模型训练与迭代
1.  **优化器**：通常使用 L-BFGS（高精度）或 Adam（快速收敛）。
2.  **回调监控**：`Corona1DRKVisCallback` 实时计算相对 L2 误差并生成演化图。
3.  **时间推进**：完成一个时间步 $\Delta t$ 的收敛后，将当前解作为下一时刻的初始值继续演化。

---

## 3. 关键性能指标 (KPIs)

| 指标 | 说明 |
| :--- | :--- |
| **Total Loss** | 物理残差的总和，反映解对物理方程的服从程度。 |
| **Relative L2 Error** | 与参考解（CSV 数据）的全局偏差，衡量仿真精度。 |
| **Max Relative Error** | 局部最大偏差，通常发生在电场强度最高的尖端区域。 |
| **Convergence Rate** | 达到收敛所需的迭代次数，受 RK 阶数 $q$ 的影响。 |

---

## 4. 物理量计算逻辑 (AutoDiff 链)
1.  **坐标输入**：$r$
2.  **求导层**：
    * $E = - \frac{d\Phi}{dr}$
    * $\text{Laplacian} = \frac{d^2\Phi}{dr^2}$
    * $\text{Flux} = \mu_e E N_e - D_e \frac{dN_e}{dr}$
3.  **系数层**：根据 $E/N$（Td）实时插值获取 $\alpha, \mu_e, D_e$。

---

## 5. 无量纲化归一化方案

为了提高神经网络的训练稳定性，所有物理量均经过归一化处理：

| 物理量 | 归一化因子 | 代码变量 | 典型值 |
| :--- | :--- | :--- | :--- |
| 电子密度 $N_e$ | $N_\text{red}$ | `N_red` | $10^{15}\ \text{m}^{-3}$ |
| 时间 $t$ | $t_\text{red}$ | `t_red` | $5 \times 10^{-9}\ \text{s}$ |
| 电压 $\Phi$ | $V_\text{red}$ | `V_red` | $10^4\ \text{V}$ |
| 空间坐标 $r$ | 半径 $R$ | `R` | $0.01\ \text{m}$ |
| 电场 $E$ | $E_\text{red} = V_\text{red}/R$ | `E_red` | $10^6\ \text{V/m}$ |

泊松方程的无量纲系数为：

$$\text{Phi\_coeff} = \frac{e \cdot N_\text{red} \cdot R^2}{\varepsilon_0 \cdot V_\text{red}}$$

---

## 6. 隐式 RK 时间推进残差

RK-PINN 的核心在于将 IRK 时间离散化嵌入到损失函数中。网络输出 $q+1$ 个 RK 阶段的解，残差形式为：

$$\mathcal{R}_\text{Ne} = N_e^{(k)} + \Delta t \cdot \mathbf{f}^{(k)} \cdot \mathbf{A}^T - N_e^0 = 0$$

其中：
- $N_e^{(k)}$：网络在第 $k$ 个 RK 阶段输出的归一化电子密度，形状 $(N_x,\ q+1)$
- $\mathbf{f}^{(k)}$：各阶段的漂移-扩散-反应算子值，形状 $(N_x,\ q)$
- $\mathbf{A}$：Butcher 矩阵（从 `ButcherTable/Butcher_{q}.npy` 加载），形状 $(q+1,\ q)$
- $N_e^0$：当前时间步初始值（由 `Ne_init_func` 提供并经 $N_\text{red}$ 归一化）

漂移-扩散-反应算子 $\mathbf{f}$ 的完整归一化形式：

$$\mathbf{f} = -\frac{\partial(\mu_e E N_e)}{\partial r} \cdot \frac{E_\text{red} t_\text{red}}{R} - \frac{\partial}{\partial r}\!\left(D_e \frac{\partial N_e}{\partial r}\right) \cdot \frac{t_\text{red}}{R^2} - \alpha \mu_e |E| N_e \cdot t_\text{red} E_\text{red}$$

---

## 7. 梯度计算技巧（向量-雅可比积）

网络同时输出 $q+1$ 列（对应 $q+1$ 个 RK 阶段），对每列单独求导代价极高。代码采用**向量-雅可比积（VJP）**技巧实现高效的批量求导：

```python
# 创建全1辅助张量
Gx0 = ones((N_x, q))      # 对应 q 个中间阶段
Gx1 = ones((N_x, q+1))    # 对应 q+1 个全部阶段

def _grad_f1(G, x):
    # 第一步：G 对 x 的 VJP，等价于对每列单独求梯度后按行累加
    grad_tmp = autograd.grad(G, x, grad_outputs=Gx1, ...)[0]
    # 第二步：通过对 Gx1 再次求导，恢复每列的独立梯度
    grad = autograd.grad(grad_tmp, Gx1, grad_outputs=ones_like(grad_tmp), ...)[0]
    return grad  # shape: (N_x, q+1)
```

此方法将 $q+1$ 次独立反向传播压缩为 **2 次**，显著降低计算开销，适合 $q$ 较大（如 $q=300$）的情形。

---

## 8. 边界条件数学形式

### 电势 $\Phi$ 的 Dirichlet BC（硬约束，由网络结构保证）

$$\Phi(r=0) = V_0, \quad \Phi(r=R) = 0$$

实现方式：网络原始输出 $\hat{u}$ 经构造变换：

$$\Phi = \hat{u} \cdot (r - R) \cdot r + \left(1 - \frac{r}{R}\right) V_0$$

### 电子密度 $N_e$ 的 Neumann BC（软约束，进入损失函数）

**阴极（$r=0$，二次电子发射）**：

$$\left.\frac{\partial N_e}{\partial r}\right|_{r=0} + N_p \cdot \mu_p \cdot |E| \cdot \gamma \cdot R \cdot E_\text{red} = 0$$

其中 $\gamma = 0.066$ 为二次电子发射系数，描述正离子轰击阴极产生二次电子的物理过程。

**阳极（$r=R$）**：

$$\left.\frac{\partial N_e}{\partial r}\right|_{r=R} = 0$$

---

## 9. 代码结构总览

```
ai4plasma/piml/rk_pinn.py
├── load_butcher_table(q)           # 加载 Butcher 矩阵，支持本地与 HuggingFace 自动下载
├── get_PhiNe_func_from_file(csv)   # 从 CSV 读取参考解，返回三次样条插值函数
├── Corona1DRKNet                   # 神经网络包装层（含硬边界约束）
│   └── forward(x)                  # 输入归一化坐标 r，输出 (Phi, Ne)，各含 q+1 列 RK 阶段
├── Corona1DRKModel (继承 PINN)     # 主模型类
│   ├── __init__(...)               # 初始化物理参数、归一化系数、网络、Butcher 矩阵
│   └── _define_loss_terms()        # 定义物理残差（PDE + BC），注册为训练目标
└── Corona1DRKVisCallback           # 训练可视化回调
    ├── __call__(epoch, ...)         # 每 log_freq 轮触发，记录误差并生成图像帧
    ├── save_gif()                   # 将训练帧合并为 GIF 动画
    └── save_final_results(...)      # 保存最终对比图与损失曲线

app/piml/rk_pinn/solve_1d_corona_rk_pinn.py   # 主运行入口脚本
```

---

## 10. 典型训练配置（`solve_1d_corona_rk_pinn.py`）

| 参数 | 值 | 说明 |
| :--- | :--- | :--- |
| `q` | 300 | RK 阶数（高阶对应更精确的时间积分） |
| `layers` | `[1, 300, 300, 300, 300, 602]` | 网络层宽（输出维度 $2(q+1)=602$） |
| `num_epochs` | 100,000 | 总训练轮数 |
| `learning_rate` | $10^{-4}$ | Adam 初始学习率 |
| `milestones` | [50000, 100000] | 多步 LR 调度，每次乘以 0.5 |
| `train_data_x_size` | 500 | 配点数量 |
| `sample_mode` | `'uniform'` | 均匀采样配点 |
| `dt` | 1.0 | 归一化时间步长（对应物理时间 $t_\text{red} = 5\ \text{ns}$） |
| `checkpoint_freq` | 5000 | 每 5000 轮保存一次模型检查点 |
| `gif_freq` | 1000 | 每 1000 轮保存一帧用于生成 GIF 动画 |