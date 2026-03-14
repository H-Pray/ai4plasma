# AI4Plasma — Project Structure

A PyTorch-based library for physics-informed machine learning and operator learning,
specifically designed for plasma physics simulation.

---

## How to Start

```bash
# Option 1 — install from PyPI
pip install ai4plasma

# Option 2 — development install (recommended for this repo)
conda create -n ai4plasma python=3.12
conda activate ai4plasma
pip install -e .

# Run an example
python app/piml/pinn/solve_1d_pinn.py
```

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, NumPy, SciPy, Pandas, Matplotlib,
TensorBoard, Imageio, FiPy, Shapely, Huggingface-hub

---

## Directory Tree

```
ai4plasma/
│
├── ai4plasma/                          ← Main Python package
│   ├── __init__.py
│   ├── config.py                       ← 全局变量设置: REAL (浮点数类型) & DEVICE (CPU/GPU)
│   │
│   ├── core/                           ← 所有模型共享的核心基础包
│   │   ├── __init__.py
│   │   ├── model.py                    ← BaseModel + CfgBaseModel: training loop, checkpoints, TensorBoard
│   │   └── network.py                  ← 网络架构: FNN, CNN, RelaxLayer, RelaxFNN
│   │
│   ├── piml/                           ← Physics-Informed Machine Learning methods
│   │   ├── __init__.py
│   │   ├── geo.py                      ← 几何与采样: Geo1D, GeoRect2D, GeoPoly2D + boundary samplers
│   │   ├── pinn.py                     ← 标准PINN模块: EquationTerm, VisualizationCallback, PINN class
│   │   ├── cs_pinn.py                  ← CS-PINN: 系数子网PINN for plasma arc (steady + transient)
│   │   ├── rk_pinn.py                  ← RK-PINN: 电晕放电场景应用
│   │   ├── meta_pinn.py                ← Meta-PINN: 模型训练加速
│   │   └── nas_pinn.py                 ← NAS-PINN: 模型架构搜索
│   │
│   ├── operator/                       ← Operator learning (function-to-function mappings)
│   │   ├── __init__.py
│   │   ├── deeponet.py                 ← DeepONet: 算子学习
│   │   └── deepcsnet.py                ← DeepCSNet: 系数子网算子回归预测碰撞截面
│   │
│   ├── plasma/                         ← Plasma physics domain models (FVM-based, no ML)
│   │   ├── __init__.py
│   │   ├── arc.py                      ← 传统方法的Arc plasma solvers: StaArc1D, TraArc1DNoV, TraArc1D (FiPy-based)
│   │   └── prop.py                     ← 插值方法: ArcPropSpline, CoronaPropSpline
│   │
│   └── utils/                          ← General-purpose utilities
│       ├── __init__.py
│       ├── common.py                   ← set_seed, numpy2torch, list2torch, Timer, physical constants
│       ├── device.py                   ← GPU/CPU management: check_gpu, Device class, torch_device
│       ├── io.py                       ← File I/O: read_json (config loading), img2gif (animation export)
│       └── math.py                     ← Auto-diff: df_dX; error metrics: calc_l2_err, calc_relative_l2_err; Real class
│
├── app/                                ← Ready-to-run example scripts (not installed as a package)
│   │
│   ├── readme/                         ← Minimal examples from the README
│   │   ├── example1.py                 ← Solve 1D ODE with a simple PINN
│   │   └── example2.py                 ← Learn solution operator with DeepONet
│   │
│   ├── piml/
│   │   ├── pinn/                       ← Basic PINN examples
│   │   │   ├── solve_1d_pinn.py        ← 1D Poisson / ODE with standard PINN
│   │   │   ├── solve_2d_rect_pinn.py   ← 2D Poisson on a rectangular domain
│   │   │   └── solve_2d_poly_pinn.py   ← 2D Poisson on a polygonal domain
│   │   │
│   │   ├── cs_pinn/                    ← CS-PINN examples (SF6 arc plasma)
│   │   │   ├── data/                   ← Input data: SF6 plasma properties + reference simulation CSVs
│   │   │   ├── solve_1d_arc_steady_cs_pinn.py          ← Steady-state arc with CS-PINN
│   │   │   ├── solve_1d_arc_transient_cs_pinn.py       ← Transient arc (with radial velocity)
│   │   │   ├── solve_1d_arc_transient_noV_cs_pinn.py   ← Transient arc (no radial velocity)
│   │   │   └── resume_1d_arc_transient_cs_pinn.py      ← Resume training from a saved checkpoint
│   │   │
│   │   ├── rk_pinn/                    ← RK-PINN examples (Argon corona discharge)
│   │   │   ├── data/                   ← Ar transport coefficients + reference simulation data
│   │   │   └── solve_1d_corona_rk_pinn.py              ← 1D transient corona discharge
│   │   │
│   │   ├── meta_pinn/                  ← Meta-PINN examples (SF6-N2 mixtures)
│   │   │   ├── data/
│   │   │   │   ├── prop/               ← Plasma properties for SF6-N2 at 10 different mixing ratios
│   │   │   │   └── *.csv               ← Reference arc simulation data per mixture ratio
│   │   │   └── solve_1d_arc_steady_meta_pinn.py        ← Meta-learning across gas compositions
│   │   │
│   │   └── nas_pinn/                   ← NAS-PINN examples
│   │       └── search_pinn_2d_poisson.py               ← Architecture search on 2D Poisson
│   │
│   ├── operator/
│   │   ├── deeponet/                   ← DeepONet examples (Poisson operator)
│   │   │   ├── solve_1d_poisson.py         ← Basic 1D DeepONet
│   │   │   ├── solve_1d_poisson_batch.py   ← 1D DeepONet with batch training
│   │   │   ├── solve_1d_poisson_test.py    ← 1D DeepONet evaluation / testing
│   │   │   ├── solve_2d_poisson.py         ← 2D DeepONet with FNN branch
│   │   │   └── solve_2d_poisson_cnn.py     ← 2D DeepONet with CNN branch
│   │   │
│   │   └── deepcsnet/                  ← DeepCSNet examples (ionization cross sections)
│   │       ├── data/csv/               ← Cross-section data CSVs for 100+ molecules (C, N, O, F, H compounds)
│   │       └── predict_total_ionxsec.py    ← Predict total ionization cross sections
│   │
│   └── plasma/
│       └── arc/                        ← Pure FVM arc solvers (no ML, for comparison / data generation)
│           ├── data/                   ← SF6 plasma property input files (.dat)
│           ├── solve_1d_arc_steady.py              ← Steady-state 1D arc via FVM
│           ├── solve_1d_arc_transient_explicit.py  ← Transient 1D arc (explicit time stepping)
│           └── solve_1d_arc_transient_noV.py       ← Transient 1D arc (no radial velocity)
│
├── docs/                               ← Sphinx documentation source
│   ├── Makefile / make.bat             ← Build docs with `make html`
│   ├── requirements.txt                ← Docs-only dependencies (sphinx, etc.)
│   └── source/
│       ├── conf.py                     ← Sphinx configuration
│       ├── index.md                    ← Docs home page
│       ├── api/                        ← Auto-generated API reference pages (core, piml, operator, plasma, utils)
│       ├── guides/                     ← User guides: installation, quickstart, concepts, training, config
│       ├── examples/                   ← Walkthrough pages for piml, operator, plasma examples
│       └── dev/                        ← Developer / contribution guide
│
├── docs/images/                        ← SVG logos + GIF animations for README
│   ├── AI4Plasma_Logo.svg
│   ├── AI4Plasma_Code.svg
│   ├── CS-PINN-Sta-Arc.gif             ← Steady arc CS-PINN training animation
│   ├── CS-PINN-Tra-Arc.gif             ← Transient arc CS-PINN training animation
│   └── RK-PINN.gif                     ← RK-PINN corona discharge training animation
│
├── pyproject.toml                      ← Package metadata, dependencies, setuptools-scm version
├── requirements.txt                    ← Flat dependency list for pip install
├── LICENSE                             ← MIT License
└── README.md                           ← Project overview, quick-start, examples, citations
```

---

## Module Roles at a Glance

| Module | Role |
|---|---|
| `config.py` | Single place to set float precision (`REAL`) and compute device (`DEVICE`) |
| `core/model.py` | All models inherit from here — handles training loop, optimizer, checkpoints, TensorBoard |
| `core/network.py` | Network zoo: `FNN` (MLP), `CNN` (1D/2D/3D), `RelaxFNN` / `RelaxLayer` (NAS) |
| `piml/geo.py` | Define 1D/2D domains and sample collocation + boundary points for PINNs |
| `piml/pinn.py` | Base PINN — add equation terms with `add_equation()`, then call `train()` |
| `piml/cs_pinn.py` | CS-PINN specialised for plasma arcs with variable transport coefficients |
| `piml/rk_pinn.py` | Adds implicit Runge-Kutta time integration on top of PINN for temporal problems |
| `piml/meta_pinn.py` | MAML outer loop — train once, adapt quickly to new gas compositions |
| `piml/nas_pinn.py` | Learns the network architecture (depth, width) alongside PDE weights |
| `operator/deeponet.py` | DeepONet branch+trunk architecture — learns operator mappings from data |
| `operator/deepcsnet.py` | DeepCSNet — coefficient-subnet operator for predicting ionization cross sections |
| `plasma/arc.py` | FiPy-based FVM arc plasma solvers — used to generate reference/ground-truth data |
| `plasma/prop.py` | Spline interpolators for temperature-dependent plasma transport properties |
| `utils/common.py` | `set_seed`, `numpy2torch`, `Timer`, physical constants (Boltzmann, electron charge) |
| `utils/device.py` | `check_gpu()`, `Device` class — call `DEVICE.set_device(0)` to switch to GPU |
| `utils/io.py` | `read_json` (load training configs), `img2gif` (export training animations) |
| `utils/math.py` | `df_dX` (auto-differentiation), `calc_relative_l2_err`, `Real` (precision manager) |
