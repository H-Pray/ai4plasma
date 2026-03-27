# 结构保辛 RK-PINN 文献调研报告

> **调研目标**：从基础 PINN 和 Zhong et al. (2022) 的 RK-PINN 出发，调研通过**结构化构造保辛性质**的 RK-PINN 方法（SRKPINN），评估该 idea 的可行性、创新性和技术路线。
>
> **调研日期**：2026-03-26

---

## 目录

1. [背景与动机](#1-背景与动机)
2. [基础方法回顾](#2-基础方法回顾)
   - 2.1 PINN 基础
   - 2.2 RK-PINN（Zhong et al. 2022）
   - 2.3 辛积分器理论基础
3. [保结构神经网络方法综述](#3-保结构神经网络方法综述)
   - 3.1 Hamiltonian Neural Networks (HNN)
   - 3.2 Lagrangian Neural Networks (LNN)
   - 3.3 SympNets
   - 3.4 生成函数方法 (GFNN)
   - 3.5 Symplectic Recurrent Neural Networks (SRNN)
   - 3.6 Nonseparable Symplectic Neural Networks (NSSNN)
   - 3.7 HénonNet 与等离子体应用
   - 3.8 SympFlow 与 Hamiltonian Matching
   - 3.9 Port-Hamiltonian Neural Networks
   - 3.10 Contact Hamiltonian 与耗散系统
4. [保结构 PINN 方法综述](#4-保结构-pinn-方法综述)
   - 4.1 Structure-Preserving PINNs (SP-PINN)
   - 4.2 SPINI：PINN + 辛积分器混合方法
   - 4.3 Multi-Symplectic PINN (MPINN)
   - 4.4 Hamilton-Dirac Neural Networks (HDNN)
5. [当前 SRKPINN 实现分析](#5-当前-srkpinn-实现分析)
   - 5.1 架构设计
   - 5.2 已取得的成果
   - 5.3 现有局限性
6. [Idea 评估：结构化构造保辛 RKPINN](#6-idea-评估结构化构造保辛-rkpinn)
   - 6.1 核心思想定位
   - 6.2 与已有方法的对比
   - 6.3 创新性分析
   - 6.4 可行性评估
   - 6.5 潜在风险与挑战
7. [三条技术路线详细分析](#7-三条技术路线详细分析)
   - 7.1 路线 A：可微隐式辛 RK 求解层
   - 7.2 路线 B：生成函数 / 离散 Hamiltonian 参数化
   - 7.3 路线 C：辛子映射复合
8. [原始 RK-PINN 及当前 SRKPINN 的问题批判](#8-原始-rk-pinn-及当前-srkpinn-的问题批判)
   - 8.1 原始 RK-PINN（Zhong et al. 2022）的根本性问题
   - 8.2 当前 SRKPINN 实现中的具体缺陷
   - 8.3 PINN 范式本身的固有困难
9. [拟开展研究面临的核心困难](#9-拟开展研究面临的核心困难)
   - 9.1 可微隐式求解层的训练稳定性
   - 9.2 精确辛性与逼近精度的张力
   - 9.3 多目标损失的梯度冲突
   - 9.4 从低维到高维的可扩展性
   - 9.5 混沌系统中的理论-实践鸿沟
   - 9.6 与已有方法的差异化论证
   - 9.7 工程实现的复杂度壁垒
10. [关键文献对照表](#10-关键文献对照表)
11. [结论与建议](#11-结论与建议)
12. [参考文献](#12-参考文献)

---

## 1. 背景与动机

### 1.1 问题陈述

物理信息神经网络（PINN）自 Raissi et al. (2019) 提出以来，已成为求解正问题与逆问题的重要工具。RK-PINN 将隐式 Runge-Kutta 时间推进嵌入 PINN 框架，允许以大时间步长推进求解，并在等离子体模拟领域展现出良好潜力（Zhong et al. 2022）。

然而，标准 PINN 和 RK-PINN 均**不保证所学映射具有辛性质**。对于 Hamiltonian 系统而言，这意味着：

- 长时间积分中能量漂移不可控；
- 相空间结构（如 KAM 环面）遭到破坏；
- 所学动力学在长时间尺度上定性失真。

**核心 idea**：在 RK-PINN 框架内，通过结构化架构设计而非软约束，确保所学单步映射精确保辛。这一方案将 RK-PINN 大时间步长推进的优势与辛积分器长程结构保持能力有机结合。

### 1.2 为什么保辛很重要

根据 Hairer, Lubich & Wanner (2006) 的几何数值积分理论：

1. **后向误差分析**：辛方法的数值解是某个修正 Hamiltonian 系统的精确解，修正 Hamiltonian （Modified Hamiltonian）与原 Hamiltonian 仅差 $O(h^r)$（$h$ 为步长，$r$ 为方法阶数）。
2. **指数长时间能量守恒**：辛积分器的能量误差在 $O(e^{\gamma/h})$ 量级的时间内保持 $O(h^r)$ 有界（Benettin-Giorgilli 定理）。
3. **KAM 稳定性**：对近可积系统，辛方法的 KAM 环面在数值离散下持续存在（Shang 1999, 2000）。
4. **相空间体积保持**：辛映射自动满足 Liouville 定理，即相空间体积守恒。

这些性质是非辛方法无法提供的——即便单步精度很高，长程行为仍可能出现定性错误。

---

## 2. 基础方法回顾

### 2.1 PINN 基础

**Physics-Informed Neural Networks (PINNs)** (Raissi, Perdikaris & Karniadakis, 2019)

核心思想：将 PDE/ODE 约束直接嵌入神经网络的损失函数中。

- 网络 $\hat{u}_\theta(t, \mathbf{x})$ 充当解的代理模型；
- 损失函数由三部分组成：

$$\mathcal{L} = \omega_f \mathcal{L}_f + \omega_B \mathcal{L}_B + \omega_I \mathcal{L}_I$$

其中 $\mathcal{L}_f$ 为 PDE 残差，$\mathcal{L}_B$ 为边界条件，$\mathcal{L}_I$ 为初始条件。

**优点**：无网格、灵活，可同时处理正问题与逆问题。

**关键局限**（Krishnapriyan et al., NeurIPS 2021）：
- 对非平凡问题（如对流方程、反应方程），PINN 可能收敛至平凡解；
- 损失景观病态，优化困难；
- **不保结构**：所学解不自动满足守恒律。

### 2.2 RK-PINN（Zhong et al. 2022）

Zhong et al. 在论文 "Low-temperature plasma simulation based on physics-informed neural networks" 中提出了两个框架：

**CS-PINN（Coefficient-Subnet PINN）**：
- 使用子网络（神经网络或样条函数）近似等离子体方程中的解依赖系数。

**RK-PINN（Runge-Kutta PINN）**：
- 将隐式 Runge-Kutta 形式主义嵌入 PINN；
- 网络以空间坐标 $\mathbf{x}$ 为输入，输出 $q+1$ 个值，对应 RK 各阶段的解；
- 采用 $q$ 阶段隐式 RK 方法：

$$u(t_0 + \Delta t \cdot c_i) = u(t_0) + \Delta t \sum_{j=1}^q a_{ij} f(u(t_0 + \Delta t \cdot c_j))$$

$$u(t_0 + \Delta t) = u(t_0) + \Delta t \sum_{j=1}^q b_j f(u(t_0 + \Delta t \cdot c_j))$$

**RK-PINN 的核心特点**：
- 允许大时间步长（相比显式方法可提升 1–2 个数量级）；
- 可处理有限或含噪的传感器数据；
- 已在等离子体模拟中得到验证：Boltzmann 方程、漂移扩散方程、Elenbaas-Heller 方程。

**RK-PINN 的局限**：
- RK 结构仅作为时间离散化工具，不承载任何辛结构；
- 未针对 Hamiltonian 系统进行设计；
- 所学映射不具备任何几何保持性质；
- 面向 PDE 空间残差，而非相空间演化。

### 2.3 辛积分器理论基础

#### 2.3.1 辛条件

对正则 Hamiltonian 系统 $\dot{q} = \nabla_p H$，$\dot{p} = -\nabla_q H$，映射 $\varphi: (q,p) \mapsto (Q,P)$ 是辛的当且仅当：

$$J_\varphi^\top \Omega J_\varphi = \Omega, \quad \Omega = \begin{pmatrix} 0 & I \\ -I & 0 \end{pmatrix}$$

#### 2.3.2 辛 Runge-Kutta 方法

$s$ 阶段 RK 方法 $(A, b, c)$ 是辛的当且仅当满足以下条件（Lasagni 1988, Sanz-Serna 1988, Suris 1988）：

$$b_i a_{ij} + b_j a_{ji} - b_i b_j = 0, \quad \forall i, j$$

重要性质：
- 所有辛 RK 方法必然是隐式的；
- **Gauss-Legendre 方法**：$s$ 阶段达到 $2s$ 阶精度，A-稳定，自共轭，且为辛方法；
- **隐式中点法**：最简单的单阶段辛 RK 方法（2 阶精度）。

#### 2.3.3 分裂方法与复合方法

对可分 Hamiltonian $H = T(p) + V(q)$：

- **Störmer-Verlet / 蛙跳法**：2 阶，辛，时间可逆，显式；
- **Yoshida 复合** (1990)：通过 "triple jump" 构造 4 阶方法：

$$\Psi_h = \varphi_{\gamma_1 h} \circ \varphi_{\gamma_2 h} \circ \varphi_{\gamma_1 h}$$

其中 $\gamma_1 = 1/(2 - 2^{1/3})$，$\gamma_2 = -2^{1/3}/(2 - 2^{1/3})$，可推广至任意偶数阶。

#### 2.3.4 生成函数与辛映射

辛正则变换可通过四种类型的生成函数隐式定义：

- **Type 1**: $F_1(q, Q)$ — $p = \partial_q F_1$，$P = -\partial_Q F_1$
- **Type 2**: $F_2(q, P)$ — $p = \partial_q F_2$，$Q = \partial_P F_2$
- **Type 3**: $F_3(p, Q)$
- **Type 4**: $F_4(p, P)$

**核心定理**：任何足够光滑的、接近恒等映射的辛映射均可由至少一种类型的生成函数表示。这为通过参数化生成函数来构造精确辛映射提供了坚实的理论基础。

#### 2.3.5 变分积分器（Marsden & West, 2001）

离散 Lagrangian 方法：

- 对 Hamilton 原理（作用量积分）进行离散化，而非对 ODE 直接离散化；
- **无论离散 Lagrangian 如何选取**，变分积分器均自动满足：
  - 辛性；
  - 动量守恒（当离散 Lagrangian 具有对称性时）。
- Verlet、SHAKE、RATTLE、Newmark 等方案均为变分积分器的特例。

---

## 3. 保结构神经网络方法综述

### 3.1 Hamiltonian Neural Networks (HNN)

**论文**：Greydanus, Dzamba & Yosinski, "Hamiltonian Neural Networks," NeurIPS 2019

**核心思想**：用神经网络参数化 Hamiltonian $H_\theta(q, p)$，通过自动微分计算 Hamilton 方程：

$$\dot{q} = \frac{\partial H_\theta}{\partial p}, \quad \dot{p} = -\frac{\partial H_\theta}{\partial q}$$

**辛性保持方式**：**软性（结构性）** — 所学向量场永远是某个标量函数的辛梯度，但时间积分步骤不一定是辛的。所学 $H_\theta$ 被精确守恒，但 $H_\theta$ 与真实 $H$ 之间的误差难以控制。

**局限**：
- 需要正则坐标 $(q, p)$ 数据；
- 积分器引入的误差可能破坏辛性；
- 前向 Euler 训练会导致人为的损失下界（David & Méhats, 2023）。

**评价**：开创性工作，建立了保结构神经网络的研究范式。但辛性的保持是间接的，不足以保证长程积分质量。

### 3.2 Lagrangian Neural Networks (LNN)

**论文**：Cranmer, Greydanus et al., "Lagrangian Neural Networks," ICLR 2020 Workshop

**核心思想**：用神经网络参数化 Lagrangian $L_\theta(q, \dot{q})$，通过 Euler-Lagrange 方程导出动力学。

**优点**：不依赖正则坐标，适用于含非完整约束的系统。

**相关扩展**：
- DeLaN (Lutter et al., 2019)：假设动能具有速度内积形式；
- Discrete Lagrangian NN (Lishkova et al., 2023)：直接学习离散 Lagrangian，规避离散化误差；
- LieFVIN (Duruisseaux et al., 2023)：在 Lie 群上学习受控 Lagrangian/Hamiltonian 动力学。

### 3.3 SympNets

**论文**：Jin, Zhang, Zhu, Tang & Karniadakis, "SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems," Neural Networks 132, 2020

**核心思想**：构造每一层均精确辛的神经网络，使得整体组合映射自动满足辛性。

**两种架构**：

**LA-SympNets**（线性-激活模块）：
- 线性模块：参数化为单位块三角辛矩阵，可进行无约束优化；
- 激活模块：逐元素非线性辛剪切映射；
- 交替组合线性模块与激活模块。

**G-SympNets**（梯度模块）：
- 每个模块形如：$(q, p) \mapsto (q, p + K^\top \text{diag}(a) \sigma(Kq + b))$；
- 或其转置版本（交换 $q, p$ 的角色）；
- 每个梯度模块均为固有辛的非线性剪切流。

**辛性保持方式**：**精确（by construction）** — 每个模块的 Jacobian 恒满足辛条件，与参数值无关。辛映射的复合仍为辛映射。

**万能逼近定理**：在适当的激活函数下，LA-SympNets 和 G-SympNets 均为辛映射空间上的万能逼近器。

**实验**：单摆、双摆、三体问题。在精度、能量守恒、长期稳定性和训练速度上均优于 MLP 和 HNN。

**评价**：这是**与本项目最直接相关**的方法之一。SympNets 证明了通过网络架构精确保辛的可行性与优越性，但它不是 PINN 框架，而是纯数据驱动的映射学习方法。

### 3.4 生成函数方法 (GFNN)

**论文**：R. Chen & Tao, "Data-driven Prediction of General Hamiltonian Dynamics via Learning Exactly-Symplectic Maps," ICML 2021

**核心思想**：用神经网络参数化 Type-2 生成函数 $S_\theta(q_n, P_{n+1})$，通过偏导数隐式定义辛映射：

$$p_n = \frac{\partial S_\theta}{\partial q_n}, \quad q_{n+1} = \frac{\partial S_\theta}{\partial P_{n+1}}$$

**辛性保持方式**：**精确（by construction）** — 由生成函数理论保证，任何光滑函数所定义的映射均为辛映射，与网络的近似质量无关。

**关键理论结果**：预测误差随时间**线性增长**（而非指数增长），这是精确辛性的直接推论。

**局限**：
- 前向传播中需要求解非线性方程（隐式映射）；
- 需要计算混合高阶导数；
- 对某些系统可能存在局部坐标奇异性。

**后续扩展**：
- GRNN (2022)：生成循环神经网络；
- LSNN (2023)：大步长神经网络，已应用于 25000 年轨道演化。

**评价**：数学上最为简洁的精确辛映射学习方法。本项目 README 中的 Scheme B 即对应此技术路线。

### 3.5 Symplectic Recurrent Neural Networks (SRNN)

**论文**：Z. Chen, Zhang, Arjovsky & Bottou, "Symplectic Recurrent Neural Networks," ICLR 2020

**核心思想**：用 HNN 参数化 Hamiltonian，将蛙跳积分器作为循环计算图展开。

**辛性保持方式**：蛙跳积分器对可分 Hamiltonian 精确保辛；时间可逆性允许以 $O(1)$ 内存开销进行反向传播。

**局限**：仅适用于可分 Hamiltonian $H = T(p) + V(q)$。

### 3.6 Nonseparable Symplectic Neural Networks (NSSNN)

**论文**：Xiong, Tong et al., "Nonseparable Symplectic Neural Networks," ICLR 2021

**核心思想**：通过**增广辛积分器**处理不可分 Hamiltonian $H(q, p)$（即动能与势能耦合的情形）。引入辅助变量和 Lagrange 乘子，在增广相空间中实现位置与动量的"解耦"。

**辛性保持方式**：增广积分器限制到物理子空间上是辛的。

**评价**：解决了先前方法（SRNN、Taylor-net）仅能处理可分 Hamiltonian 这一关键限制。

### 3.7 HénonNet 与等离子体应用

**论文**：Burby, Tang & Maulik, "Fast neural Poincaré maps for toroidal magnetic fields," Plasma Phys. Control. Fusion 63, 2021

**核心思想**：每一层均为 Hénon 型辛映射，复合后保持辛性。专为学习环形磁场的 Poincaré 映射而设计。

**等离子体物理扩展**（Drimalas et al., Physics of Plasmas 32, 2025）：
- **SympMat**（线性）和 **HénonNet**（非线性）用于电磁场中带电粒子动力学；
- SympMat 在亚回旋尺度上优于 Boris 推进器；
- 可作为 PIC 模拟中的快速代理模型。

**评价**：这是**保结构方法在等离子体物理中最直接的应用**，与本项目的长远目标高度契合。

### 3.8 SympFlow 与 Hamiltonian Matching

**论文**：Canizares, Murari et al., "Hamiltonian Matching for Symplectic Neural Integrators," NeurIPS 2024

**核心思想**：将辛神经网络定义为参数化含时 Hamiltonian 的精确流映射的复合，并允许通过后向误差分析识别"影子 Hamiltonian"。

**辛性保持方式**：**精确（by construction）** — Hamiltonian 流映射的复合是辛的，同时构成 Hamiltonian 流空间上的万能逼近器。

**训练目标**：Hamiltonian Matching — 最小化所学影子 Hamiltonian 与真实 Hamiltonian 之间的距离。

### 3.9 Port-Hamiltonian Neural Networks

**论文**：多篇，Physical Review E (2021), NeurIPS 2024, arXiv 2025

**核心思想**：将 Hamiltonian 系统推广至开放系统（具有能量输入/输出端口），学习 port-Hamiltonian 结构。

**关键扩展**：
- **Stable pHNN** (2025)：加入全局 Lyapunov 稳定性保证；
- **Port-Metriplectic NN** (2023)：同时满足热力学第一定律和第二定律；
- **耦合系统 pHNN** (NeurIPS 2024)：支持模块化子系统建模。

**评价**：为非保守系统（含耗散、外力驱动）提供了结构化的建模框架，是保辛方法的自然推广。

### 3.10 Contact Hamiltonian 与耗散系统

**论文**：Testa et al., "Geometric Contact Flows," arXiv:2506.17868, 2025

**核心思想**：以接触几何（contact geometry）取代辛几何，为耗散系统提供几何结构框架。接触 Hamiltonian 模型可以内嵌稳定性与能量耗散，同时保持几何结构。

**其他相关工作**：
- 共形辛方法（conformal symplectic）用于阻尼系统；
- SympFlow 通过相空间加倍处理耗散效应。

---

## 4. 保结构 PINN 方法综述

### 4.1 Structure-Preserving PINNs (SP-PINN)

**论文**：Chu, Miyatake et al., "Structure-Preserving Physics-Informed Neural Networks with Energy or Lyapunov Structure," IJCAI 2024

**核心思想**：在 PINN 损失函数中引入保结构项（能量守恒或 Lyapunov 结构）。

**方式**：**软约束** — 通过额外的损失项鼓励而非强制结构保持。

**局限**：软约束不能保证结构的精确保持，且损失权重的平衡较为困难。

### 4.2 SPINI：PINN + 辛积分器混合方法

**论文**：Liang et al., "SPINI: A Structure-Preserving Neural Integrator for Hamiltonian Dynamics and Parametric Perturbation," Nature Scientific Reports 15, 2025

**核心思想**：**两阶段混合方法** —
1. **阶段一**：无监督 PINN 从控制方程（而非轨迹数据）学习 Hamiltonian 函数；
2. **阶段二**：将所学 Hamiltonian 代理嵌入 4 阶 Yoshida 辛积分器进行时间演化。

**辛性保持方式**：**阶段二精确辛** — 即使所学 $H$ 是近似的，时间步进本身也精确保辛；PINN 阶段本身不强制辛性。

**评价**：这是**与本项目 idea 最为接近的已发表工作**。SPINI 将"学习物理"与"积分动力学"解耦处理，但其辛积分器部分采用的是传统方法（Yoshida），而非神经网络构建。本项目的 idea 可视为在 RK-PINN 框架内实现类似的结构保持，以辛 RK 结构替代 Yoshida 分裂法。

### 4.3 Multi-Symplectic PINN (MPINN)

**论文**：Physics Letters A, 2026

**核心思想**：以多辛结构（而非 PDE 本身）作为 PINN 损失函数中的物理信息，专门针对无穷维 Hamiltonian 系统（如 KdV 方程、非线性 Schrödinger 方程）设计。

**扩展**：GMPINN（广义多辛 PINN）将框架进一步推广至耗散系统。

### 4.4 Hamilton-Dirac Neural Networks (HDNN)

**论文**：Kaltsas, "Constrained Hamiltonian Systems and Physics-Informed Neural Networks," Physical Review E 111, 2025

**核心思想**：运用 Dirac 约束理论处理约束 Hamiltonian 系统，通过 Dirac 括号和约束正则化确保动力学不偏离约束流形。

**应用**：笛卡尔坐标下的非线性摆、导心运动（与等离子体物理直接相关）。

---

## 5. 当前 SRKPINN 实现分析

### 5.1 架构设计

当前 `SRKPINN/` 包实现了一个**正则 Hamiltonian 单步映射学习器**，核心设计如下：

```
输入: z_n = (q_n, p_n)
  ↓
[FNN backbone: 2 → 128 → 128 → 128 → 4]
  ↓
[HamiltonianSRKNet: 输出解构为 (Q_1,P_1), (Q_2,P_2)]
  ↓
[Hard symplectic RK closure:]
  q_{n+1} = q_n + Δt Σ b_i ∂H/∂p(Q_i, P_i)
  p_{n+1} = p_n - Δt Σ b_i ∂H/∂q(Q_i, P_i)
  ↓
输出: z_{n+1} = (q_{n+1}, p_{n+1})
```

**关键设计决策**：
1. 网络仅预测 RK 阶段变量 $(Q_i, P_i)$，不直接预测终态；
2. 终态通过**硬辛 RK 闭合公式**重构；
3. Butcher 表经过辛条件验证；
4. 损失函数包含阶段方程残差（StageDynamics）和数据拟合（InitialOrData）。

### 5.2 已取得的成果

| 指标 | 数值 |
|------|------|
| 单步 RMSE L2 | 3.72e-4 |
| 局部辛缺陷（均值） | 2.76e-4 |
| 中等能量 200 步能量漂移 | 2.22e-3 |
| 近分界线 1000 步误差 | 灾难性增长（~16） |

### 5.3 现有局限性

1. **辛性非精确**：阶段方程仅通过损失最小化软性满足，所学映射近似具有辛性，但并非精确辛映射；
2. **后向误差分析不适用**：由于映射不精确辛，无法从理论上保证长时间能量有界；
3. **分界线附近鲁棒性差**：高能区域的误差累积导致定性错误；
4. **训练数据覆盖不均匀**：当前均匀采样策略未对困难区域加权；
5. **未与基线比较**：尚未与原始 RK-PINN 或其他保结构方法进行基准对比。

### 5.4 当前方法的理论定位

当前 SRKPINN 应被描述为：

> 一个使用辛 RK 表 + 硬闭合但软阶段一致性的 Hamiltonian 基准分支。

它**不是**精确辛映射学习器，但**已经比**通用 RK 风格回归器或早期使用软终态头的设计更接近辛结构。

---

## 6. Idea 评估：结构化构造保辛 RKPINN

### 6.1 核心思想定位

本项目的 idea 可以精确表述为：

> **在 RK-PINN 框架内，通过结构化架构设计（而非损失函数软约束），构造精确保辛的学习型单步映射。**

这一方案融合了两个研究方向的优势：
- **RK-PINN** 的隐式 RK 大步长推进能力与 PINN 的灵活性；
- **辛积分器/辛网络** 的精确结构保持性质。

### 6.2 与已有方法的对比

| 方法 | 辛性保持方式 | 需要轨迹数据 | RK 结构 | PINN 框架 |
|------|------------|-----------|---------|----------|
| HNN (2019) | 软（学 H，不保证积分辛） | 是 | 否 | 否 |
| SympNets (2020) | **精确（架构）** | 是 | 否 | 否 |
| GFNN (2021) | **精确（生成函数）** | 是 | 否 | 否 |
| SRNN (2020) | 精确（蛙跳积分器） | 是 | 否 | 否 |
| SPINI (2025) | 阶段二精确（Yoshida） | 否（无监督） | 否 | **是** |
| SP-PINN (2024) | 软（损失正则化） | 否 | 否 | **是** |
| 原始 RK-PINN (2022) | 无 | 取决于问题 | **是** | **是** |
| **当前 SRKPINN** | 近辛（硬闭合 + 软阶段） | 是（一步对） | **是** | **是** |
| **提议方法（精确辛 RKPINN）** | **精确** | 取决于路线 | **是** | **是** |

### 6.3 创新性分析

**高创新性维度**：

1. **RK-PINN 与精确辛性的结合是尚未被充分探索的方向**。现有文献中，RK-PINN 不关注辛性（Zhong et al. 2022），保辛方法不采用 PINN 框架（SympNets, GFNN），而 SPINI (2025) 虽结合了 PINN 与辛积分器，但将两者解耦为独立阶段，且采用传统辛积分器（Yoshida），不使用 RK 结构。

2. **隐式辛 RK + 神经网络作为求解器/warm start 的思路**（对应 Scheme A）在现有文献中尚未出现。这是将数值分析中的隐式辛 RK 方法与深度学习的函数逼近能力直接融合的创新性探索。

3. **将 PINN 的无数据优势引入保辛框架**：若实现得当，可从物理定律（而非轨迹数据）出发构建精确辛映射，这超越了 SympNets 和 GFNN 的纯数据驱动范式。

**中等创新性维度**：

4. 生成函数路线（Scheme B）已有 GFNN 的先驱工作，但将其纳入 PINN 框架是新的尝试。

5. 辛子映射复合路线（Scheme C）在概念上与 SympNets 相似，但在 RK-PINN 框架下的具体实现是新的。

**需要注意的创新性边界**：

6. 单纯的"辛网络 + Hamiltonian ODE"已有大量文献覆盖。创新点必须着重强调 **PINN 框架的独特优势**（如无监督学习、物理残差驱动、无需轨迹数据）。

### 6.4 可行性评估

**技术可行性：高**

- `SRKPINN/` 包已具备大部分基础设施：Butcher 表验证、Hamiltonian 抽象、硬 RK 闭合、rollout 诊断；
- 三条技术路线均有清晰的数学基础和可参考的已有实现；
- 单摆是成熟的基准问题，便于快速验证。

**工程可行性：中高**

- Scheme C（辛子映射复合）可在 1–2 周内实现原型；
- Scheme B（生成函数）需要处理隐式求解，工程复杂度有所提升；
- Scheme A（可微隐式 RK 求解层）工程量最大，但最忠实于当前设计理念。

**学术发表可行性：高**

- 该方向处于 PINN、几何积分与深度学习的交叉点，2024–2025 年相关论文数量迅速增长；
- 可定位为 "structure-preserving RK-PINN for Hamiltonian systems"；
- 若能在单摆及更复杂系统上展示显著的长程改进，有望在 JCP、CMAME 或 NeurIPS/ICML 等期刊或会议发表。

### 6.5 潜在风险与挑战

1. **隐式求解的训练稳定性**（Scheme A, B）：非线性根求解在训练初期可能无法收敛，尤其在大 $\Delta t$ 或困难相空间区域；

2. **精确辛性 ≠ 精确映射**：SympNets 和 GFNN 的文献已指出，精确辛映射仍可能存在较大的单步误差。辛性保证长程稳定性，但单步精度仍有赖于充足的网络容量和训练质量；

3. **可分性假设的局限**（Scheme C 的简单版本）：对不可分 Hamiltonian，简单的分裂方法表达能力可能不足，需考虑 NSSNN 式的增广空间方法；

4. **与纯辛网络的比较困难**：若 SRKPINN 最终与 SympNets 性能相当，则需清晰论证 PINN 框架所带来的额外价值（如无数据训练能力）；

5. **计算开销**：隐式求解层或生成函数求解可能显著增加前向与反向传播的计算成本。

---

## 7. 三条技术路线详细分析

### 7.1 路线 A：可微隐式辛 RK 求解层

#### 核心思想

保持当前 SRKPINN 的辛 RK 框架，但将阶段一致性从损失函数移入前向映射。

网络不再直接输出最终阶段状态，而是输出**初始猜测**或**求解器热启动**。前向传播中嵌入一个可微非线性求解器，**显式求解辛 RK 阶段方程**。收敛的阶段状态通过现有硬闭合公式得到终态。

#### 实现细节

```
输入: z_n = (q_n, p_n)
  ↓
[FNN backbone → stage initial guess (Q^0, P^0)]
  ↓
[可微非线性求解器（如 Anderson 加速 / Newton-Krylov）:]
  反复迭代直到:
    Q_i = q_n + Δt Σ a_{ij} ∂H/∂p(Q_j, P_j)
    P_i = p_n - Δt Σ a_{ij} ∂H/∂q(Q_j, P_j)
  精确成立
  ↓
[Hard symplectic closure → z_{n+1}]
```

#### 所需改动

- 在 `_predict_tensor()` 中加入批量隐式求解层；
- 网络角色从"阶段预测器"转变为"阶段猜测器"；
- `StageDynamics` 损失大幅降权或移除（由求解器保证一致性）；
- 加入求解器诊断（迭代次数、残差范数、失败率）。

#### 文献支持

- DEQ (Deep Equilibrium Models, Bai et al., NeurIPS 2019)：证明了可微不动点求解层的可训练性；
- Implicit differentiation (Griewank & Walther, 2008)：为反向传播提供理论基础；
- 隐式层的梯度可通过隐函数定理高效计算，无需展开迭代过程。

#### 优劣分析

| 维度 | 评分 |
|------|------|
| 辛忠实度 | ★★★★★（最忠实于 SRK 身份） |
| 实现复杂度 | ★★★★☆（需要可微求解器） |
| 计算成本 | ★★★★☆（每步需要迭代求解） |
| 训练稳定性 | ★★★☆☆（求解器可能不收敛） |
| 创新性 | ★★★★★（未见于文献） |

### 7.2 路线 B：生成函数 / 离散 Hamiltonian 参数化

#### 核心思想

用 Type-2 生成函数替代阶段状态参数化：

$$F_2(q_n, P_{n+1}) = q_n^\top P_{n+1} + G_\theta(q_n, P_{n+1})$$

辛映射通过偏导数隐式定义：

$$p_n = \frac{\partial F_2}{\partial q_n} = P_{n+1} + \frac{\partial G_\theta}{\partial q_n}$$

$$q_{n+1} = \frac{\partial F_2}{\partial P_{n+1}} = q_n + \frac{\partial G_\theta}{\partial P_{n+1}}$$

#### 实现细节

```
输入: z_n = (q_n, p_n)
  ↓
[求解隐式方程得到 P_{n+1}:]
  p_n = P_{n+1} + ∂G_θ/∂q_n(q_n, P_{n+1})
  ↓
[计算 q_{n+1}:]
  q_{n+1} = q_n + ∂G_θ/∂P_{n+1}(q_n, P_{n+1})
  ↓
输出: z_{n+1} = (q_{n+1}, p_{n+1} = P_{n+1})
```

#### 所需改动

- 新增标量输出的生成函数网络 $G_\theta$；
- 前向传播中求解隐式方程（Newton 法）；
- 从梯度构造 $q_{n+1}$；
- 以数据拟合与求解器质量监控替代阶段损失；
- 作为**新模型类**实现（不修改 `HamiltonianSRKPINN`）。

#### 文献支持

- GFNN (Chen & Tao, ICML 2021)：直接先驱；
- LSNN (2023)：大步长扩展；
- 辛高斯过程回归 (Rath et al., 2021)：通过 GP 观测生成函数导数。

#### 优劣分析

| 维度 | 评分 |
|------|------|
| 数学纯粹度 | ★★★★★（最简洁的精确辛路线） |
| 实现复杂度 | ★★★★☆（隐式求解 + 混合高阶导数） |
| 计算成本 | ★★★☆☆（每步求根 + 二阶导数） |
| 与当前设计的兼容性 | ★★☆☆☆（偏离 RK 阶段身份） |
| 创新性 | ★★★☆☆（GFNN 已建立范式） |

### 7.3 路线 C：辛子映射复合

#### 核心思想

将单步映射构建为多个精确辛基元映射的复合。对可分结构：

$$q \leftarrow q + \nabla T_k(p), \quad p \leftarrow p - \nabla V_k(q)$$

每个子映射均为辛映射，其整体复合亦为辛映射。

#### 具体架构选项

**选项 C1：SympNets 风格的梯度模块**

```
Layer_k: (q, p) → (q, p + K_k^T diag(a_k) σ(K_k q + b_k))
Layer_{k+1}: (q, p) → (q + K_{k+1}^T diag(a_{k+1}) σ(K_{k+1} p + b_{k+1}), p)
交替组合 L 层
```

**选项 C2：学习型分裂**

$$q \leftarrow q + \Delta t \cdot f_\theta^T(p), \quad p \leftarrow p - \Delta t \cdot f_\theta^V(q)$$

其中 $f_\theta^T$, $f_\theta^V$ 是某学习标量函数的梯度。

**选项 C3：Yoshida 风格复合**

在基础 2 阶辛方法上应用 Yoshida 系数复合得到 4 阶方法，但基础方法中的 $T$ 和 $V$ 由网络参数化。

#### 所需改动

- 新增辛映射网络后端；
- 新增模型类（如 `HamiltonianSympMapPINN`）；
- 复用训练对、rollout、callback 和辛诊断模块；
- 作为**并行模型族**处理。

#### 文献支持

- SympNets (Jin et al., 2020)：G-SympNets 架构；
- SRNN (Chen et al., 2020)：蛙跳展开；
- Taylor-nets (Tong et al., 2021)：4 阶辛积分器 + Neural ODE；
- P-SympNets (Tapley, 2024)：基于几何积分器设计 SympNets 的统一框架。

#### 优劣分析

| 维度 | 评分 |
|------|------|
| 实现速度 | ★★★★★（最快可获得精确辛原型） |
| 计算成本 | ★★☆☆☆（无隐式求解） |
| 辛忠实度 | ★★★★★（精确辛 by construction） |
| 与 RK 身份的兼容性 | ★★☆☆☆（偏离 SRK 身份） |
| 表达能力 | ★★★☆☆（简单分裂对不可分 H 受限） |
| 创新性 | ★★☆☆☆（SympNets 已建立范式） |

### 7.4 路线比较总结

```
                      路线 A          路线 B          路线 C
                  (隐式 SRK 层)   (生成函数)     (辛子映射复合)
                  ─────────────  ────────────  ──────────────
SRK 身份保持       ★★★★★         ★★☆☆☆        ★★☆☆☆
精确辛性           ★★★★★         ★★★★★        ★★★★★
实现速度           ★★☆☆☆         ★★★☆☆        ★★★★★
计算效率           ★★☆☆☆         ★★★☆☆        ★★★★★
训练稳定性         ★★★☆☆         ★★★☆☆        ★★★★★
创新性             ★★★★★         ★★★☆☆        ★★☆☆☆
可扩展性           ★★★★★         ★★★★☆        ★★★☆☆
学术价值           ★★★★★         ★★★★☆        ★★★☆☆
```

### 7.5 推荐策略

**分阶段双轨策略**：

1. **第一阶段（1–2 周）**：实现路线 C 作为精确辛基线，快速获取对比数据；
2. **第二阶段（2–4 周）**：实现路线 A 作为主要贡献，这是创新性最高的方向；
3. **可选第三阶段**：如需更广泛的方法比较，补充路线 B。

路线 A 是**推荐的主要研究贡献**，原因在于：
- 最忠实于 SRK-PINN 的方法论身份；
- 在现有文献中最为新颖；
- 直接融合了数值分析（隐式辛 RK）与深度学习（可微求解器）的前沿成果；
- 可清晰论证相对于 SympNets（纯数据驱动）和 SPINI（PINN + 传统辛积分器解耦）的独特优势。

---

## 8. 原始 RK-PINN 及当前 SRKPINN 的问题批判

本节对原始 RK-PINN（Zhong et al. 2022）和当前 SRKPINN 实现进行系统性批判分析，明确指出需要解决的根本性问题。

### 8.1 原始 RK-PINN（Zhong et al. 2022）的根本性问题

#### 8.1.1 RK 结构仅是时间离散化工具，不具有几何意义

原始 RK-PINN 使用隐式 Runge-Kutta 形式主义将时间推进嵌入 PINN。然而，其中的 RK 结构**纯粹是时间离散化设备**，不包含任何辛性或几何保持的考量：

- Butcher 表的选取（论文中使用 $q$ 阶段隐式方法）未经辛条件 $b_i a_{ij} + b_j a_{ji} - b_i b_j = 0$ 的验证；
- RK 阶段约束仅作为损失函数中的软约束出现（公式 10–17），不保证精确满足；
- 终态通过网络直接输出（$\hat{u}_0^{i,q+1}$），而非通过硬闭合公式重构；
- 这意味着即使 Butcher 表恰好满足辛条件，所学映射也**不会因此获得辛性质**。

**根本问题**：RK-PINN 将一个具有丰富几何结构的数值方法（隐式 RK）退化为纯代数约束，丢弃了其几何内涵。

#### 8.1.2 训练目标与参考解的自洽性问题

RK-PINN 损失函数中（公式 12），训练目标 $u_0^i$ 来自外部参考解或初始条件。这引入了一个微妙但重要的问题：

- 若参考解由非辛方法（如显式时间推进）生成，则训练目标**本身就不满足辛性质**；
- 网络被迫学习一个非辛映射以匹配非辛参考解；
- 即便后续引入辛约束，目标函数本身的不一致性也会造成梯度冲突。

这一问题在当前 SRKPINN 中部分存在：参考步进器使用 DOP853（8(5,3) 阶显式方法，**非辛**），意味着训练目标 $z_{n+1}^{ref}$ 不是辛映射的像。

#### 8.1.3 大阶段数带来的病态性

原始论文声称使用多达 $q=300$ 甚至 $q=500$ 个 RK 阶段。这在理论上提供了极高的时间精度，但在实践中存在严重问题：

- **输出维度爆炸**：网络须同时输出 $q+1$ 个值。对 $q=300$ 而言，输出维度达 301，网络容量需求和优化难度急剧增加；
- **Butcher 表条件数恶化**：高阶 Gauss-Legendre 方法的 $A$ 矩阵涉及高阶 Legendre 多项式的根，数值条件数随阶段数增长；
- **理论精度无实际意义**：$O(\Delta t^{600})$ 量级的时间误差远低于浮点精度，实际误差完全由网络逼近能力和优化质量决定；
- **损失项耦合剧增**：$q$ 个阶段方程相互耦合，形成 $q \times q$ 的隐式非线性系统作为软约束嵌入损失。

Raissi et al. (2019) 原始 PINN 论文中同样使用了 $q=100$ 乃至 $q=499$ 的阶段数，但后续研究（arXiv:2409.16826）报告了在此配置下"无法准确预测轨迹"的失败案例。

#### 8.1.4 缺乏误差估计机制

传统 RK 方法通过嵌入式方法对（如 Dormand-Prince 的 RK45）提供可靠的局部误差估计，支持自适应步长控制。RK-PINN 完全缺失这一机制：

- 没有可靠的局部误差指标；
- 没有步长自适应能力；
- 无法判断某个时间步的预测是否可信；
- 这对安全关键应用是不可接受的（arXiv:2401.05211）。

#### 8.1.5 PDE 残差驱动 vs. 相空间演化

原始 RK-PINN 面向 PDE（空间残差最小化），其网络以空间坐标 $\mathbf{x}$ 为输入，输出空间上的解。这与 Hamiltonian 系统相空间演化的需求存在根本性的范式差异：

- PDE 残差强调空间逐点满足，而 Hamiltonian 系统强调相空间全局结构；
- PDE-PINN 的边界条件处理（软约束）在 Hamiltonian 系统中没有自然对应；
- 等离子体模拟中的多物理耦合（电子密度、电势、离子密度）在 RK-PINN 中通过多网络实现，但这些网络之间的结构一致性没有保证。

### 8.2 当前 SRKPINN 实现中的具体缺陷

#### 8.2.1 核心矛盾：硬闭合 + 软阶段 = 非辛映射

当前设计的核心矛盾在于：

```
辛 RK 表（精确辛）
  + 硬闭合公式（结构正确）
  + 软阶段方程（仅通过损失最小化近似满足）
  = 非辛映射（辛性取决于阶段方程的满足程度）
```

具体而言，辛 RK 方法的辛性依赖于**阶段方程与闭合公式同时精确满足**。闭合公式

$$q_{n+1} = q_n + \Delta t \sum_i b_i \nabla_p H(Q_i, P_i)$$

$$p_{n+1} = p_n - \Delta t \sum_i b_i \nabla_q H(Q_i, P_i)$$

本身并不保证辛性——它仅是将终态表示为阶段变量的函数。辛性来源于阶段方程

$$Q_i = q_n + \Delta t \sum_j a_{ij} \nabla_p H(Q_j, P_j)$$

的**精确**满足。当阶段方程仅近似满足时（当前 StageDynamics 残差 $\sim 10^{-4}$），辛性同样只是近似的。

**后果**：后向误差分析不再适用，无法保证存在一个被精确守恒的修正 Hamiltonian，因此长时间能量漂移缺乏理论上界。

#### 8.2.2 训练数据的内在不一致性

当前实现中，训练对 $(z_n, z_{n+1}^{ref})$ 由 DOP853 积分器生成：

```python
# model.py: _default_reference_stepper
def _default_reference_stepper(self, states, dt: float):
    return default_reference_stepper(self.system, states, dt)

# utils.py: DOP853, rtol=1e-10, atol=1e-10
```

DOP853 是一个 8(5,3) 阶**显式** Runge-Kutta 方法。它：
- **不是辛方法**（显式 RK 方法不可能是辛的，辛 Euler 是例外）；
- 单步精度极高（$\sim 10^{-10}$），但其输出的 $z_{n+1}^{ref}$ **不是辛映射的像**；
- 这意味着网络试图以辛 RK 结构去匹配一个非辛目标。

这造成根本性矛盾：若网络完美学会了辛 RK 映射，它反而**不应该**完美匹配 DOP853 的输出，因为两者对应不同的映射（辛 vs. 非辛）。当前良好的单步精度（RMSE $3.72 \times 10^{-4}$）实际上表明网络更多地学会了匹配参考解，而非学会了辛映射。

**理想替代方案**：训练目标应由辛积分器（如相同 Gauss-Legendre 方法的高精度实现）生成，以确保目标本身满足辛性质。

#### 8.2.3 采样策略的系统性缺陷

当前均匀网格采样：

```python
# systems.py: PendulumSystem.sample_initial_states, mode="uniform"
grid_side = int(np.ceil(np.sqrt(num_samples)))
q_vals = np.linspace(self.q_range[0], self.q_range[1], grid_side)
p_vals = np.linspace(self.p_range[0], self.p_range[1], grid_side)
```

存在以下具体问题：

1. **相空间动力学的不均匀性被忽略**：分界线（separatrix）附近动力学变化剧烈，需要高密度采样，但均匀网格对此完全无感知；
2. **能量分层缺失**：不同能量层面的轨道具有定性不同的行为（振动 vs. 旋转），均匀采样在低能区过密、在高能区不足；
3. **512 个训练样本可能不足**：对 2D 相空间，512 点约对应 $23 \times 23$ 的网格，分辨率有限；
4. **训练数据一次性生成，不随训练更新**：缺乏自适应重采样或 hard-example mining 策略。

rollout 评估结果直接反映了这一缺陷：

| 区域 | 200 步能量漂移 | 1000 步能量漂移 |
|------|-------------|--------------|
| 中等能量 $(1.7, 0)$ | $2.22 \times 10^{-3}$ | $1.35 \times 10^{-2}$ |
| 近分界线 $(3.0, 0)$ | $3.25 \times 10^{-3}$ | **$8.01$**（灾难性） |

近分界线区域的灾难性失败直接归因于该区域训练数据覆盖不足。

#### 8.2.4 损失权重平衡缺乏理论指导

当前使用手动设定的固定权重：

```python
self.loss_weights = {
    "StageDynamics": 1.0,   # 阶段方程残差
    "InitialOrData": 1.0,   # 数据匹配 (脚本中覆盖为 2.0)
}
```

这存在多层面问题：

- **StageDynamics 和 InitialOrData 的梯度量级可能差异悬殊**：Wang et al. (2021) 已证明 PINN 中不同损失项的梯度量级可相差数个数量级，固定权重无法实现有效平衡；
- **两个损失目标存在内在冲突**：完美满足阶段方程（辛 RK 映射）和完美匹配参考解（DOP853 非辛映射）不可能同时达到，优化器必须在两者之间权衡；
- **缺乏自适应权重策略**：如 NTK-based reweighting（Wang et al. 2021）、GradNorm 或 Pareto 多目标优化；
- **Rohrhofer et al. (2023) 已证明** PINN 训练本质上是多目标优化问题，其 Pareto 前沿可能停留在损失值显著大于零处。

#### 8.2.5 网络架构过于简单

当前 backbone 是标准 3 层 FNN：

```python
backbone_net = FNN(
    layers=[2, 128, 128, 128, 4],  # 输入 z=(q,p), 输出 4 维阶段变量
    act_fun=nn.Tanh(),
)
```

问题：

- **Tanh 激活的频谱偏差**（Rahaman et al., ICML 2019）：网络优先学习低频成分，对分界线附近尖锐的相空间结构（高频特征）学习困难；
- **缺乏周期性编码**：单摆的 $q$ 是角度变量，具有 $2\pi$ 周期性，但网络输入为裸 $q$ 值，未使用 $(\sin q, \cos q)$ 编码，迫使网络在 $q = \pm\pi$ 处处理本不存在的不连续性；
- **未利用 Hamiltonian 的已知解析形式**：$H(q,p) = \frac{1}{2}p^2 + \omega_0^2(1 - \cos q)$ 在训练中已通过 `system.gradients()` 使用，但网络架构未利用这一先验知识来约束输出空间；
- **参数量有限**：$2 \times 128 + 128 \times 128 + 128 \times 128 + 128 \times 4 = 33,\!284$ 个参数。对于需要在整个相空间上逼近非线性映射的任务，这可能不足。

#### 8.2.6 辛性诊断的局限性

`symplectic_map_residual()` 方法计算局部辛缺陷 $J_F^\top \Omega J_F - \Omega$：

```python
# 逐列计算 Jacobian（低效且仅适用于低维）
for col in range(self.state_dim):
    grad = torch.autograd.grad(next_state[0, col], state_tensor, ...)
    jacobian_rows.append(grad[0])
```

问题：

- **仅支持单状态评估**（`state_array.shape[0] != 1` 时报错），无法进行批量计算；
- **逐列 Jacobian 计算**：需要 $O(d)$ 次反向传播，对高维系统不可扩展；
- **局部缺陷 ≠ 全局辛性**：映射在采样点处可以具有较小的局部辛缺陷，但在全局上仍可能严重非辛；
- **未监控辛缺陷随训练的变化趋势**：无法判断训练过程是否在改善辛性质。

### 8.3 PINN 范式本身的固有困难

#### 8.3.1 频谱偏差（Spectral Bias）

Rahaman et al. (ICML 2019) 确立的"频率原理"表明：标准全连接神经网络**先学低频、后学高频**。对 Hamiltonian 系统而言：

- 分界线附近的动力学包含尖锐的相空间结构（高频）；
- 固定点附近的小振动是平滑的（低频）；
- 网络倾向于先学会简单区域，对困难区域的学习被延迟乃至忽略。

这一规律解释了当前 SRKPINN 在中等能量区域表现良好、而在分界线附近失败的现象。

#### 8.3.2 因果性违反

Wang, Sankaran & Perdikaris (2022) 发现，PINN 倾向于**违反时间因果性**：优化器偏好先最小化后期时间步的残差，而非按时间顺序学习。

对单步映射学习器（如 SRKPINN），这一问题体现为：网络可能在某些相空间区域（简单区域）早已收敛，而对困难区域的学习被推迟，但两类区域的训练数据在每个 epoch 中被等权混合处理。

#### 8.3.3 损失景观的病态性

Krishnapriyan et al. (NeurIPS 2021) 的核心发现：PINN 失败**并非因为网络表达能力不足**，而是因为 PINN 损失景观极难优化。

- 软 PDE 正则化使损失景观**病态化**；
- 随问题参数（如对流系数）增大，损失景观复杂度急剧上升；
- PINN 可以收敛至损失接近零、但 L2 误差接近 1 的平凡解。

Cao et al. (JCP, 2024) 进一步证明：PINN 的病态性与 PDE 系统 Jacobian 矩阵的条件数直接相关，且这种病态性**对任何基于 ML 的 PDE 惩罚方法都是不可避免的**。

#### 8.3.4 刚性系统的固有困难

对于刚性 ODE/PDE 系统（等离子体物理中普遍存在）：

- PDE 残差损失因刚性**主导总损失**，迫使模型优先满足 ODE 约束而忽视初始/边界条件，从而收敛至**平凡解**；
- 动力系统的**不动点**（无论稳定性如何）对应物理损失的局部极小值，形成严重的收敛陷阱；
- 标准 PINN 无法处理**自适应步长**，而这是刚性系统数值方法的必要特性。

---

## 9. 拟开展研究面临的核心困难

本节系统分析拟开展的保辛 RKPINN 研究中所面临的具体技术困难与理论挑战。

### 9.1 可微隐式求解层的训练稳定性（路线 A 的核心难点）

路线 A 要求在前向传播中嵌入可微非线性求解器。Deep Equilibrium Models (DEQ, Bai et al., NeurIPS 2019) 的研究揭示了以下关键困难：

#### 9.1.1 训练过程中求解器不稳定性递增

DEQ 文献报告了一个普遍现象：**随着训练进行，达到不动点所需的迭代次数持续增长**，系统逼近不稳定边界。具体表现为：

- 不动点处 Jacobian 矩阵的谱半径在训练中逐渐增大；
- 前向传播与反向传播的收敛质量同步恶化；
- 某些样本可能完全无法收敛，需要在训练中被跳过。

对 SRKPINN 而言，这意味着：随着网络学习到更准确的阶段映射，隐式 RK 方程的求解可能反而变得更加困难（因为 Hamiltonian 梯度的非线性更强）。

#### 9.1.2 早停求解器导致噪声梯度

当求解器因效率需要被提前终止时，梯度在非平衡点处计算，产生**噪声或不正确的梯度信号**。这可能导致：

- 训练震荡，无法收敛；
- 网络参数向使求解器更易收敛（但映射精度更低）的方向偏移；
- 网络退化为"仅提供热启动猜测"，而非真正的映射学习器。

#### 9.1.3 反向传播的实现选择

隐式层的反向传播有多种实现策略，各有利弊：

| 策略 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| 隐函数定理 | $\partial z^*/\partial \theta = -(I - \partial g/\partial z)^{-1} \partial g/\partial \theta$ | 精确、内存恒定 | 需要求解线性系统 |
| 展开反向传播 | 通过迭代计算图传播梯度 | 简单直接 | 内存 $O(T)$，梯度消失/爆炸 |
| Jacobian-free 近似 | 用 $-I$ 替换 Jacobian | 计算最快 | 理论保证有限 |
| Jacobian 正则化 | 对 Jacobian 谱半径加惩罚 | 显著稳定化 | 额外计算开销 |

对于辛 RK 求解层，**隐函数定理方法最为适合**（因为辛结构提供了 Jacobian 的额外结构信息），但需要高效的线性系统求解器。

#### 9.1.4 大 $\Delta t$ 和困难区域的收敛性

隐式辛 RK 方程的非线性程度取决于 $\Delta t \cdot \|\nabla^2 H\|$。当：

- $\Delta t$ 较大（RK-PINN 的核心优势之一），
- 或 $\nabla^2 H$ 较大（分界线附近，$|\nabla^2 H| = \omega_0^2 |\cos q|$ 接近极值）时，

隐式方程可能进入非收缩区域，Newton 迭代不保证收敛。这恰恰是当前 SRKPINN 表现最差的区域；引入隐式求解层可能反而使情况进一步恶化。

**可能的缓解策略**：
- 限制 $\Delta t$ 的范围（以牺牲大步长优势为代价）；
- 使用延续法（continuation method），从小 $\Delta t$ 逐步增大；
- 使用多次重启的 Anderson 加速替代纯 Newton 迭代。

### 9.2 精确辛性与逼近精度的张力

这是贯穿所有三条技术路线的根本性挑战。

#### 9.2.1 精确辛 ≠ 精确映射

SympNets 和 GFNN 的文献已明确指出：**精确辛映射仍可能存在较大的单步误差**。辛性保证的是长程稳定性（能量有界漂移），而非单步精度。

这意味着：
- 一个精确辛但单步精度较差的模型，在能量守恒上可能优于当前 SRKPINN，但在短程轨迹跟踪上可能更差；
- 评估标准需要同时考量单步精度与长程结构保持，两者可能不会同时达到最优。

#### 9.2.2 辛约束对逼近空间的限制

通过架构硬约束辛性等价于将网络的输出空间从全体光滑映射限制到辛映射子空间。这一限制：

- **降低了表达自由度**：辛映射空间是全体映射空间的一个（无穷余维数的）子流形；
- **可能引入逼近困难**：对某些目标映射，辛映射子空间中的最优逼近可能远劣于无约束最优逼近；
- **SympNets 的万能逼近定理**（Jin et al., 2020）保证了理论上的逼近能力，但实际所需的网络深度/宽度可能很大。

JMLR 2024 年的一项研究正式刻画了 port-Hamiltonian 系统中**结构保持与表达能力**之间的根本性权衡：两者之间存在由可控性/可观性决定的不可调和的张力。

#### 9.2.3 修正 Hamiltonian 问题

David & Méhats (JCP, 2023) 的关键发现：当用辛积分器训练 HNN 时，网络学到的不是真实 Hamiltonian $H$，而是一个**修正 Hamiltonian** $\tilde{H}$：

$$\tilde{H} = H + h^r H_{r+1} + h^{r+1} H_{r+2} + \cdots$$

这是后向误差分析的直接推论。其实际含义是：

- 网络所学动力学是某个修正系统的精确动力学；
- 修正系统与原系统的差异为 $O(h^r)$；
- 对于**精确辛**方法，这一差异受到良好控制（指数长时间有界）；
- 但若研究目标是"学习真实 Hamiltonian"，修正 Hamiltonian 不可避免地引入系统偏差。

### 9.3 多目标损失的梯度冲突

#### 9.3.1 当前 SRKPINN 的双目标冲突

当前损失 $\mathcal{L} = w_1 \text{MSE}(R^{stage}) + w_2 \text{MSE}(R^{data})$ 中的两个目标存在**内在冲突**：

- $R^{stage} \to 0$ 意味着网络输出精确满足辛 RK 阶段方程，即学到辛映射；
- $R^{data} \to 0$ 意味着网络输出匹配 DOP853 参考解，即学到非辛映射。

这一矛盾无法通过调整权重解决——两个目标指向不同的最优解，优化器必须在 Pareto 前沿上进行权衡。

Wang et al. (2021) 的 NTK 分析表明，这种不平衡会导致：梯度方向冲突，某一损失项主导优化，另一损失项收敛停滞。

#### 9.3.2 路线 A 中冲突的转化（非消除）

路线 A（可微隐式求解层）将 StageDynamics 从损失移入前向映射，表面上消除了双目标冲突。但实际上冲突以**新的形式**延续：

- 若隐式求解器精确收敛 → 映射精确辛 → 与非辛参考解的不匹配被暴露为更大的 $R^{data}$；
- 数据损失的梯度通过隐式求解层反向传播到网络参数，可能驱使参数向使隐式方程不那么"辛"的方向移动（尽管结构上仍然辛）。

**解决方向**：使用辛参考步进器（如相同 Gauss-Legendre 方法的高精度实现）生成训练目标，从根本上消除这一冲突。

#### 9.3.3 引入更多损失项时的冲突放大

若未来引入额外的结构约束（如能量守恒正则化、辛缺陷惩罚），梯度冲突将进一步加剧：

- 每增加一个损失项，Pareto 前沿的维度增加；
- 多目标间的配对梯度冲突频率增加；
- GC-PINN (2025) 和 ConFIG (ICLR 2025) 提出的梯度投影方法可以缓解冲突，但引入额外计算开销。

### 9.4 从低维到高维的可扩展性

当前在 2D 相空间（单摆，$d=2$）上的工作需要推广到更高维系统。

#### 9.4.1 维度增长的直接影响

| 系统 | 相空间维度 $2n$ | 辛映射参数化复杂度 |
|------|---------------|----------------|
| 单摆 | 2 | 低 |
| 双摆 | 4 | 中 |
| 受限三体 | 4 | 中 |
| 三体问题 | 12 | 高 |
| 粒子束（$N$ 粒子） | $6N$ | 极高 |

- SympNets 需要与系统维度同阶的矩阵，参数量 $O(n^2)$ 增长；
- GFNN 的隐式方程维度随 $n$ 线性增长，求解成本 $O(n^3)$（Newton 迭代中的线性系统求解）；
- 辛 RK 求解层的隐式系统维度为 $2ns$（$n$ 为自由度数，$s$ 为阶段数），计算成本迅速膨胀。

#### 9.4.2 高维 Jacobian 计算的瓶颈

辛缺陷诊断 $J_F^\top \Omega J_F - \Omega$ 需要完整的 $2n \times 2n$ Jacobian 矩阵：

- 当前实现需要 $2n$ 次反向传播（逐列计算）；
- 对三体问题（$n=6$），需要 12 次反向传播；
- 对 $N$ 粒子系统，此方案完全不可行；
- 需要引入随机 Jacobian 估计或结构化计算。

#### 9.4.3 SympGNNs 作为可能出路

Symplectic Graph Neural Networks (SympGNNs, arXiv:2408.16698) 利用图结构处理多体问题，但引入了新的复杂度（图构建、消息传递与辛结构的兼容性）。

### 9.5 混沌系统中的理论-实践鸿沟

#### 9.5.1 辛性不保证混沌轨道的跟踪精度

辛积分器的核心保证是**守恒辛 2-形式**和**近守恒修正 Hamiltonian**。但对于混沌系统：

- 轨迹对初始条件敏感依赖，任何有限精度方法都会在有限时间后偏离真实轨迹；
- 辛性保证的是**统计性质**（如遍历测度、Lyapunov 指数的正确性），而非单条轨迹的跟踪；
- 后向误差分析的修正 Hamiltonian 级数**发散**（仅是渐近级数），最终失效。

对于单摆（可积系统），辛性的优势更为直接（KAM 环面保持）。但在推广至混沌系统（如三体问题）时，辛性的实际收益需要重新评估。

#### 9.5.2 分界线的特殊困难

即使对单摆这样的可积系统，分界线（separatrix）也是动力学中最具挑战性的区域：

- 分界线上 Lyapunov 时间趋于无穷小（$T_{Lyap} \to 0$）；
- 微小的能量误差可以将轨迹从振动翻转为旋转，产生定性错误；
- 辛积分器在分界线附近同样面临困难（步长需极小）；
- 当前 SRKPINN 的 $\Delta t = 0.1$ 在分界线附近可能过大。

这不仅仅是训练数据覆盖不足的问题——即使经过完美训练，在分界线附近以 $\Delta t = 0.1$ 的步长进行单步映射，精度也可能不足。

### 9.6 与已有方法的差异化论证

#### 9.6.1 为什么不直接用 SympNets？

这是将面临的最尖锐的学术质疑。SympNets（Jin et al., 2020）已经提供了：

- 精确辛性（by construction）；
- 万能逼近定理；
- 高效的训练与推理；
- 在单摆、双摆、三体问题上的验证。

需要清晰论证 SRKPINN 相对于 SympNets 的**不可替代优势**。候选论证方向：

1. **PINN 的无监督能力**：SympNets 需要轨迹数据，SRKPINN 可从物理方程出发（但当前实现仍依赖参考步进器生成数据）；
2. **RK 结构的可解释性**：辛 RK 阶段对应具有明确物理意义的中间状态；
3. **灵活性**：可集成更多物理先验（边界条件、守恒律、对称性）。

但这些优势需要**实验证据**支撑，而非仅停留在论证层面。

#### 9.6.2 为什么不直接用 SPINI？

SPINI (2025) 已经实现了 PINN 与辛积分器的结合。相比之下，SRKPINN 需要论证：

1. **端到端训练 vs. 两阶段解耦**的优劣；
2. **辛 RK 结构内嵌网络 vs. 传统辛积分器外接**的差异；
3. 在特定问题类上的性能优势。

#### 9.6.3 差异化的关键论证策略

最有说服力的差异化来自**路线 A（可微隐式辛 RK 求解层）**，因为：

- SympNets 不使用 RK 结构；
- GFNN 不使用 PINN 框架；
- SPINI 不在 PINN 训练循环内嵌入辛结构；
- 路线 A 独特地将隐式辛 RK 方法与可微编程融合在 PINN 训练中。

### 9.7 工程实现的复杂度壁垒

#### 9.7.1 自动微分的嵌套深度

当前 SRKPINN 的计算图已经包含：

```
网络前向 → Hamiltonian 梯度 (autograd) → 闭合公式 → 损失 → 反向传播 (autograd)
```

路线 A 进一步加深为：

```
网络前向 → 初始猜测 → 隐式求解 (多次迭代, 每次需要 autograd) → 闭合 → 损失 → 隐式反向传播 (autograd through autograd)
```

这涉及**至少三层嵌套自动微分**：
1. Hamiltonian 梯度 $\nabla H$；
2. 隐式求解器中的 Jacobian；
3. 损失函数对网络参数的梯度。

PyTorch 的 `create_graph=True` 可以处理上述需求，但：
- 内存消耗显著增大；
- 计算图复杂度急剧增加；
- 调试极为困难（梯度错误难以定位）；
- 数值稳定性更加脆弱（浮点误差在多层微分中放大）。

#### 9.7.2 批量化隐式求解的工程挑战

训练需要对每个 mini-batch 中的所有样本并行求解隐式方程。这要求：

- 批量化 Newton 迭代（不同样本可能需要不同迭代次数）；
- 处理部分样本收敛失败的情况（跳过？惩罚？截断？）；
- 确保批内操作的 GPU 并行效率；
- 在求解器迭代与 autograd 计算图之间正确传播梯度。

现有 DEQ 实现（如 `torchdeq`）可作为参考，但辛 RK 方程的特殊结构（辛矩阵的对称性）需要定制化实现。

#### 9.7.3 调试和验证的复杂度

对于包含可微隐式求解层的辛 RK-PINN，验证其正确性需要依次确认：

1. Butcher 表的辛性（已有）；
2. 隐式求解器的收敛性；
3. 收敛后的阶段状态满足 RK 方程；
4. 闭合后的映射精确辛；
5. 反向传播的梯度正确性（隐式微分 vs. 展开微分的一致性）；
6. 训练收敛到有意义的解（非平凡解）。

每一步均可能引入 bug，且 bug 的表现往往不是直接报错，而是训练质量的隐性退化。

---

## 10. 关键文献对照表

### 10.1 基础方法

| 文献 | 年份 | 会议/期刊 | 核心贡献 | 辛性 |
|------|------|----------|---------|------|
| Raissi et al. "PINNs" | 2019 | JCP 378 | PINN 框架，离散时间 PINN 使用隐式 RK | 无 |
| Zhong et al. "Plasma PINNs" | 2022 | arXiv:2206.15294 | CS-PINN 和 RK-PINN 用于等离子体模拟 | 无 |
| Hairer et al. "Geometric NI" | 2006 | Springer | 几何数值积分教材 | 理论基础 |

### 10.2 保结构神经网络

| 文献 | 年份 | 会议/期刊 | 核心贡献 | 辛性 |
|------|------|----------|---------|------|
| Greydanus et al. "HNN" | 2019 | NeurIPS | 学习 Hamiltonian 函数 | 软 |
| Cranmer et al. "LNN" | 2020 | ICLR WS | 学习 Lagrangian 函数 | 软 |
| Jin et al. "SympNets" | 2020 | Neural Networks 132 | 精确辛网络架构 + 万能逼近定理 | **精确** |
| Z. Chen et al. "SRNN" | 2020 | ICLR | 蛙跳展开 + 初态优化 | 精确（可分 H） |
| R. Chen & Tao "GFNN" | 2021 | ICML | 生成函数参数化辛映射 | **精确** |
| Xiong et al. "NSSNN" | 2021 | ICLR | 不可分 Hamiltonian 的辛网络 | 精确 |
| Burby et al. "HénonNet" | 2021 | PPCF 63 | Poincaré 映射学习（磁场） | **精确** |
| David & Méhats | 2023 | JCP | 辛学习改进 HNN 训练 | 精确（训练中） |
| Tong et al. "Taylor-net" | 2021 | JCP 437 | Taylor 展开 + 4 阶辛积分器 | 精确 |

### 10.3 保结构 PINN 方法

| 文献 | 年份 | 会议/期刊 | 核心贡献 | 辛性 |
|------|------|----------|---------|------|
| Chu et al. "SP-PINN" | 2024 | IJCAI | 能量/Lyapunov 保结构 PINN | 软 |
| Liang et al. "SPINI" | 2025 | Sci. Rep. | PINN 学 H + Yoshida 辛积分器 | 阶段二精确 |
| MPINN | 2026 | Phys. Lett. A | 多辛 PINN（无穷维） | 软 |
| Kaltsas "HDNN" | 2025 | Phys. Rev. E | 约束 Hamiltonian PINN | 软 |

### 10.4 最新进展（2024-2026）

| 文献 | 年份 | 会议/期刊 | 核心贡献 | 辛性 |
|------|------|----------|---------|------|
| SympFlow | 2024 | NeurIPS | Hamiltonian 流映射复合 + Matching | **精确** |
| GHNNs | 2025 | JCP 521 | 统一 SRNN/SympNets/HénonNet | **精确** |
| PSNN | 2025 | JCP | 伪辛 NN，一层显式 + Padé 激活 | 近辛 |
| P-SympNets | 2024 | arXiv | 基于动力系统的 SympNets 框架 | **精确** |
| Drimalas et al. | 2025 | Phys. Plasmas | 辛 NN 用于带电粒子动力学 | **精确** |
| Symplectic Gyroceptron | 2023 | Sci. Rep. | 近周期辛映射的保结构 NN | **精确** |
| Stable pHNN | 2025 | arXiv | 全局 Lyapunov 稳定的 port-Hamiltonian NN | 结构性 |
| GCF | 2025 | arXiv | 接触 Hamiltonian 流用于耗散系统 | 接触辛 |

---

## 11. 结论与建议

### 11.1 总体评价

**本项目的 idea——在 RK-PINN 框架内结构化构造精确保辛映射——是一个具有重要价值且时机成熟的研究方向。**

理由如下：

1. **Gap 明确存在**：现有文献中，RK-PINN 不保辛，保辛网络不使用 PINN 框架，SPINI 虽结合两者但将学习与积分解耦。**在 PINN 训练框架内同时嵌入精确辛 RK 结构**是一个尚未被充分探索的空白。

2. **基础设施已就绪**：`SRKPINN/` 包提供了辛表验证、Hamiltonian 抽象、硬闭合公式、rollout 诊断等完整基础，可直接在此基础上进行升级。

3. **时机成熟**：2024–2025 年该领域论文数量爆发式增长（SympFlow NeurIPS 2024, GHNNs JCP 2025, SPINI Sci. Rep. 2025, Plasma NN Phys. Plasmas 2025），学术界对此方向有强烈兴趣。

4. **等离子体物理直接应用前景**：HénonNet 已在 Poincaré 映射中得到应用，SympMat 优于 Boris 推进器。保辛 RKPINN 自然适用于等离子体中的 Hamiltonian 子系统（导心运动、粒子动力学等）。

### 11.2 创新性等级评估

| 维度 | 评级 | 说明 |
|------|------|------|
| 问题定义 | 中高 | "保辛 + PINN" 已有关注，但 "保辛 + RK-PINN" 更为具体 |
| 路线 A（可微隐式辛 RK 层） | **高** | 未见于文献，融合数值分析与深度学习前沿 |
| 路线 B（生成函数） | 中 | GFNN 已建立范式，在 PINN 框架中是新的尝试 |
| 路线 C（辛子映射复合） | 中低 | SympNets 已建立范式，包装为 PINN 是增量贡献 |
| 等离子体应用 | 中高 | 与 HénonNet/SympMat 并行但方法论不同 |

### 11.3 推荐的研究路径

```
Phase 1: 精确辛基线 (1-2 周)
├── 实现路线 C (SympNets 风格) 作为精确辛基线
├── 在单摆上与当前 SRKPINN 对比
├── 验证精确辛性带来的长程改进
└── 建立基准评估框架

Phase 2: 主要贡献 (3-4 周)
├── 实现路线 A (可微隐式辛 RK 求解层)
├── 这是创新性最高且最忠实于 SRK 身份的方向
├── 与 Phase 1 基线和当前 SRKPINN 三方对比
└── 探索训练稳定性和计算效率

Phase 3: 扩展与论文 (2-3 周)
├── 推广到更复杂系统 (双摆、Kepler 问题)
├── 如需要，补充路线 B 作为额外对比
├── 完善诊断和基准结果
└── 撰写论文

Phase 4: 等离子体应用 (后续)
├── 将最优方法应用于带电粒子动力学
├── 与 Boris 推进器和 HénonNet 对比
└── 连接回 ai4plasma 的整体目标
```

### 11.4 核心评估指标

任何精确辛升级都应报告以下指标：

1. **单步预测误差**（held-out 状态上的 RMSE）；
2. **多步 rollout 误差**（多个时间范围：200, 1000, 5000 步）；
3. **绝对和相对能量漂移**；
4. **局部辛缺陷** $J^\top \Omega J - \Omega$（对精确辛方法应降至浮点精度）；
5. **隐式求解器失败率**（如适用）；
6. **训练墙钟时间和参数量**；
7. **多随机种子的均值与标准差**。

验证标准应从当前的

> "映射具有小的局部辛缺陷"

升级为

> "映射由辛构造定义，任何残留缺陷由求解器容差或浮点误差主导"

---

## 12. 参考文献

### 12.1 基础方法

1. M. Raissi, P. Perdikaris, G.E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs." *J. Comput. Phys.* 378:686-707, 2019. [arXiv:1711.10561](https://arxiv.org/abs/1711.10561)

2. L. Zhong, B. Wu, Y. Wang. "Low-temperature plasma simulation based on physics-informed neural networks: Frameworks and preliminary applications." *arXiv:2206.15294*, 2022. [arXiv](https://arxiv.org/abs/2206.15294)

3. E. Hairer, C. Lubich, G. Wanner. *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*. Springer, 2nd ed., 2006. [Springer](https://link.springer.com/book/10.1007/3-540-30666-8)

4. J.E. Marsden, M. West. "Discrete mechanics and variational integrators." *Acta Numerica* 10:357-514, 2001. [DOI](https://doi.org/10.1017/S096249290100006X)

### 12.2 Hamiltonian/Lagrangian 神经网络

5. S. Greydanus, M. Dzamba, J. Yosinski. "Hamiltonian Neural Networks." *NeurIPS* 2019. [arXiv:1906.01563](https://arxiv.org/abs/1906.01563)

6. M. Cranmer, S. Greydanus, S. Hoyer, P. Battaglia, D. Spergel, S. Ho. "Lagrangian Neural Networks." *ICLR 2020 Workshop*. [arXiv:2003.04630](https://arxiv.org/abs/2003.04630)

7. M. Lutter, C. Ritter, J. Peters. "Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning." *ICLR* 2019. [arXiv:1907.04490](https://arxiv.org/abs/1907.04490)

8. A. Sosanya, S. Greydanus. "Dissipative Hamiltonian Neural Networks." *arXiv:2201.10085*, 2022. [arXiv:2201.10085](https://arxiv.org/abs/2201.10085)

### 12.3 精确辛网络

9. P. Jin, Z. Zhang, A. Zhu, Y. Tang, G.E. Karniadakis. "SympNets: Intrinsic structure-preserving symplectic networks for identifying Hamiltonian systems." *Neural Networks* 132:166-179, 2020. [arXiv:2001.03750](https://arxiv.org/abs/2001.03750)

10. R. Chen, M. Tao. "Data-driven Prediction of General Hamiltonian Dynamics via Learning Exactly-Symplectic Maps." *ICML* 2021. [arXiv:2103.05632](https://arxiv.org/abs/2103.05632)

11. Z. Chen, J. Zhang, M. Arjovsky, L. Bottou. "Symplectic Recurrent Neural Networks." *ICLR* 2020. [arXiv:1909.13334](https://arxiv.org/abs/1909.13334)

12. S. Xiong, Y. Tong, X. He, S. Yang, C. Yang, B. Zhu. "Nonseparable Symplectic Neural Networks." *ICLR* 2021. [arXiv:2010.12636](https://arxiv.org/abs/2010.12636)

13. J.W. Burby, Q. Tang, R. Maulik. "Fast neural Poincaré maps for toroidal magnetic fields." *Plasma Phys. Control. Fusion* 63:024001, 2021. [DOI](https://doi.org/10.1088/1361-6587/abcbaa)

14. Y. Tong, S. Xiong, X. He, G. Pan, B. Zhu. "Symplectic Neural Networks in Taylor Series Form for Hamiltonian Systems." *J. Comput. Phys.* 437:110325, 2021. [arXiv:2005.04986](https://arxiv.org/abs/2005.04986)

### 12.4 最新保结构方法（2023-2026）

15. M. David, F. Méhats. "Symplectic Learning for Hamiltonian Neural Networks." *J. Comput. Phys.* 2023. [arXiv:2106.11753](https://arxiv.org/abs/2106.11753)

16. Canizares, Murari et al. "Hamiltonian Matching for Symplectic Neural Integrators." *NeurIPS* 2024. [arXiv:2410.18262](https://arxiv.org/abs/2410.18262)

17. P. Horn, V. Saz Ulibarrena, B. Koren, S. Portegies Zwart. "A Generalized Framework of Neural Networks for Hamiltonian Systems." *J. Comput. Phys.* 521:113536, 2025. [DOI](https://doi.org/10.1016/j.jcp.2024.113536)

18. B.K. Tapley. "Symplectic Neural Networks Based on Dynamical Systems." *arXiv:2408.09821*, 2024. [arXiv:2408.09821](https://arxiv.org/abs/2408.09821)

19. J. Bajars. "Locally-Symplectic Neural Networks for Learning Volume-Preserving Dynamics." *J. Comput. Phys.* 476:111911, 2023. [arXiv:2109.09151](https://arxiv.org/abs/2109.09151)

20. V. Duruisseaux, J.W. Burby, Q. Tang. "Approximation of Nearly-Periodic Symplectic Maps via Structure-Preserving Neural Networks." *Sci. Rep.* 13:8351, 2023. [DOI](https://doi.org/10.1038/s41598-023-34862-w)

### 12.5 保结构 PINN 方法

21. C. Liang et al. "SPINI: A Structure-Preserving Neural Integrator for Hamiltonian Dynamics and Parametric Perturbation." *Sci. Rep.* 15:43842, 2025. [Nature](https://www.nature.com/articles/s41598-025-28710-2)

22. H. Chu, Y. Miyatake, W. Cui, S. Wei, D. Furihata. "Structure-Preserving Physics-Informed Neural Networks with Energy or Lyapunov Structure." *IJCAI* 2024. [DOI](https://doi.org/10.24963/ijcai.2024/428)

23. S. Yildiz, P. Goyal, P. Benner. "Structure-preserving learning for multi-symplectic PDEs." *Adv. Model. Simul. Eng. Sci.* 12:6, 2025. [DOI](https://doi.org/10.1186/s40323-025-00287-5)

24. D.A. Kaltsas. "Hamilton-Dirac Neural Networks." *Phys. Rev. E* 111:025301, 2025. [arXiv:2401.15485](https://arxiv.org/abs/2401.15485)

### 12.6 等离子体应用

25. E.G. Drimalas, F. Fraschetti, C. Huang, Q. Tang. "Symplectic neural network and its application to charged particle dynamics in electromagnetic fields." *Phys. Plasmas* 32:103901, 2025. [DOI](https://doi.org/10.1063/5.0283551)

### 12.7 辛积分器理论

26. F.M. Lasagni. "Canonical Runge-Kutta methods." *ZAMP* 39:952-953, 1988. [DOI](https://doi.org/10.1007/BF00945133)

27. J.M. Sanz-Serna. "Runge-Kutta schemes for Hamiltonian systems." *BIT* 28:877-883, 1988. [URL](https://sanzserna.org/1988/01/05/j-m-sanz-serna-runge-kutta-schemes-for-hamiltonian-systems-bit-281988-877-883/)

28. H. Yoshida. "Construction of higher order symplectic integrators." *Phys. Lett. A* 150:262-268, 1990. [DOI](https://doi.org/10.1016/0375-9601(90)90092-3)

29. G. Benettin, A. Giorgilli. "On the Hamiltonian interpolation of near-to-the-identity symplectic mappings with application to symplectic integration algorithms." *J. Statist. Phys.* 74:1117-1143, 1994. [DOI](https://doi.org/10.1007/BF02188219)

30. M. Tao. "Explicit symplectic approximation of nonseparable Hamiltonians." *Phys. Rev. E* 94:043303, 2016. [arXiv:1609.02212](https://arxiv.org/abs/1609.02212)

### 12.8 其他重要文献

31. R.T.Q. Chen, Y. Rubanova, J. Bettencourt, D.K. Duvenaud. "Neural Ordinary Differential Equations." *NeurIPS* 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)

32. S. Bai, J.Z. Kolter, V. Koltun. "Deep Equilibrium Models." *NeurIPS* 2019. [arXiv:1909.01377](https://arxiv.org/abs/1909.01377)

33. A.S. Krishnapriyan et al. "Characterizing possible failure modes in physics-informed neural networks." *NeurIPS* 2021. [arXiv:2109.01050](https://arxiv.org/abs/2109.01050)

34. Y.D. Zhong, B. Dey, A. Chakraborty. "Symplectic ODE-Net: Learning Hamiltonian Dynamics with Control." *ICLR* 2020. [arXiv:1909.12077](https://arxiv.org/abs/1909.12077)

35. Y. Chen, T. Matsubara, T. Yaguchi. "Neural Symplectic Form: Learning Hamiltonian Equations on General Coordinate Systems." *NeurIPS* 2021. [NeurIPS](https://proceedings.neurips.cc/paper/2021/hash/8b519f198dd26772e3e82874826b04aa-Abstract.html)

36. X. Li, J. Li, Z.J. Xia, N. Georgakarakos. "Large-step neural network for learning the symplectic evolution from partitioned data." *MNRAS* 524(1):1374-1385, 2023. [DOI](https://doi.org/10.1093/mnras/stad1948)

37. A. Testa, S. Hauberg, T. Asfour, L. Rozo. "Geometric Contact Flows: Contactomorphisms for Dynamics and Control." *arXiv:2506.17868*, 2025. [arXiv:2506.17868](https://arxiv.org/abs/2506.17868)

38. K. Rath, C.G. Albert, B. Bischl, U. von Toussaint. "Symplectic Gaussian process regression of maps in Hamiltonian systems." *Chaos* 31(5):053121, 2021. [DOI](https://doi.org/10.1063/5.0048129)

39. Y. Lishkova, P. Scherer, S. Ridderbusch, M. Jamnik, P. Lio, S. Ober-Blobaum, C. Offen. "Discrete Lagrangian Neural Networks with Automatic Symmetry Discovery." *IFAC-PapersOnLine* 56(2):3203-3210, 2023. [DOI](https://doi.org/10.1016/j.ifacol.2023.10.1457)

40. X. Cheng, L. Wang, Y. Cao, C. Chen. "Learning non-separable Hamiltonian systems with pseudo-symplectic neural network." *J. Comput. Phys.* 550:114630, 2026. [DOI](https://doi.org/10.1016/j.jcp.2025.114630)
