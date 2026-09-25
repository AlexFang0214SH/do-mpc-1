# 03 · 核心 API 参考

按模块组织的速查手册。每一项都标注了源码位置，需要完整 docstring 时可直接跳转。

> **符号约定**：`★` = 必须由用户调用；`○` = 可选；`⚙` = 内部方法，一般不用直接调。

---

## 目录

- [`do_mpc.model`](#do_mpcmodel) — 建模
- [`do_mpc.optimizer`](#do_mpcoptimizer) — 优化器基类
- [`do_mpc.controller`](#do_mpccontroller) — MPC / LQR
- [`do_mpc.estimator`](#do_mpcestimator) — StateFeedback / EKF / MHE
- [`do_mpc.simulator`](#do_mpcsimulator) — 仿真器
- [`do_mpc.data`](#do_mpcdata) — 数据容器与存取
- [`do_mpc.graphics`](#do_mpcgraphics) — 绘图与动画
- [`do_mpc.tools`](#do_mpctools) — 辅助工具

---

## `do_mpc.model`

源码：[`do_mpc/model/`](../do_mpc/model) · 导出：`Model`, `LinearModel`, `IteratedVariables`, `linearize`, `dae2odeconversion`

### `Model(model_type, symvar_type='SX')`

[`model/_model.py`](../do_mpc/model/_model.py)

| 参数 | 取值 | 说明 |
|---|---|---|
| `model_type` | `'continuous'` \| `'discrete'` | **必填**，非字符串会断言失败 |
| `symvar_type` | `'SX'`（默认）\| `'MX'` | CasADi 符号类型，会被所有派生类继承 |

#### 变量声明

| 方法 | 说明 |
|---|---|
| ★ `set_variable(var_type, var_name, shape=(1,1), input_type_integer=False)` | 声明变量。`var_type` ∈ `_x`/`_u`/`_z`/`_p`/`_tvp`/`_w`/`_v`。`input_type_integer=True` 会产生 **MINLP** 问题（变量名会被登记进 `model.integer`）。返回符号变量，可直接参与表达式运算。 |

#### 方程与表达式

| 方法 | 说明 |
|---|---|
| ★ `set_rhs(var_name, expr, process_noise=False)` | 为**每个** `_x` 状态设置右端表达式。连续模型是 $\dot x$，离散模型是 $x_{k+1}$。`process_noise=True` 会自动生成对应的 `_w` 噪声变量。 |
| ○ `set_alg(expr_name, expr)` | 设置代数方程 $0 = g(\cdot)$，用于 DAE。 |
| ○ `set_expression(expr_name, expr)` | 定义**辅助表达式**（→ `_aux`）。会被记录进 `Data`，可绘图、可复用于代价函数与约束。 |
| ○ `set_meas(meas_name, expr, meas_noise=True)` | 定义测量方程（→ `_y`）。`meas_noise=True` 会自动生成对应的 `_v` 噪声变量。**不调用时默认全状态可测。** |

#### 查询属性（只读，返回 CasADi 结构体，可按名索引）

| 属性 | 内容 | 典型用途 |
|---|---|---|
| `model.x` | 微分状态 | 写代价函数 `model.x['C_b']` |
| `model.u` | 控制输入 | 写约束 |
| `model.z` | 代数状态 | DAE |
| `model.p` | 参数 | 鲁棒 MPC / MHE 权重矩阵 |
| `model.tvp` | 时变参数 | 参考轨迹跟踪 |
| `model.y` | 测量 | MHE |
| `model.aux` | 辅助表达式 | 绘图 |
| `model.w` / `model.v` | 过程 / 测量噪声 | MHE |
| `model.n_x`, `n_u`, `n_z`, `n_p`, `n_tvp`, `n_y`, `n_aux`, `n_w`, `n_v` | 各变量维度 | 构造权重矩阵 |
| `model.keys()` | 全部变量名 | 遍历 |
| `model.integer` | 整数输入名列表 | MINLP |
| `model.flags['setup']` | 是否已 `setup` | 状态检查 |

索引语法（`Model.__getitem__`）：

```python
model['_x']                 # 整个 _x 结构体
model['_x', 'C_a']          # 单个变量
model.x['C_a']              # 等价写法
model.x[['C_a','C_b']]      # 多个变量（列表索引）
```

#### 生命周期

| 方法 | 说明 |
|---|---|
| ★ `setup()` | **冻结模型**。内部生成 `_rhs_fun`、`_alg_fun`、`_meas_fun` 等 CasADi `Function`，并做一致性检查。之后任何修改都会触发断言。 |
| ○ `get_linear_system_matrices(xss, uss, tvp=None, p=None)` | 在给定工作点求 $(A,B,C,D)$ 雅可比矩阵，返回 `numpy` 数组或符号表达式。 |
| ⚙ `__getstate__` / `__setstate__` | 支持 pickle（会剔除不可序列化的 CasADi 对象）。 |
| ⚙ `_transfer_variables(old, new, transfer=['_x','_u','_tvp','_p'])` | 在模型转换工具间搬运变量定义。 |

### `LinearModel(model_type, symvar_type='SX')`

[`model/_linearmodel.py`](../do_mpc/model/_linearmodel.py) — 继承自 `Model`，表示 LTI 系统 $\dot x = Ax + Bu,\ y = Cx + Du$。

**两种建模路线：**

1. **和 `Model` 一样**：`set_variable` → `set_rhs` → `set_meas` → `setup()`（不传矩阵）
2. **直接给矩阵**：`set_variable` → `setup(A, B, C, D)`

| 成员 | 说明 |
|---|---|
| `sys_A` / `sys_B` / `sys_C` / `sys_D` | 只读属性，返回系统矩阵 |
| `setup(A=None, B=None, C=None, D=None)` | ★ 传入矩阵完成设置 |
| `set_rhs(name, rhs)` | ⚠️ 被重写：只接受 $A x + B u$ 形式 |
| `set_meas(name, meas)` | ⚠️ 被重写：只接受 $C x + D u$ 形式 |
| `set_alg(...)` | ⚠️ **不支持**（线性模型不含代数变量） |
| `discretize(t_step=0, conv_method='zoh')` | 连续 → 离散转换，返回新的 `LinearModel`。`conv_method` 见 CasADi 文档（`'zoh'` 零阶保持等）。 |
| `get_steady_state(xss=None, uss=None)` | 求稳态点 |

### `linearize(model, xss=None, uss=None, tvp0=None, p0=None) -> LinearModel`

[`model/_linearize.py`](../do_mpc/model/_linearize.py)

在稳态工作点 $(x_{ss}, u_{ss})$ 用泰勒展开把非线性 `Model` 线性化，得到 $\Delta\dot x = A\Delta x + B\Delta u$。

- **前提**：`model.setup()` 已调用；**不支持 DAE**（`assert model.z.size == 0`）。
- **不支持 LTV**：若 $(A,B,C,D)$ 不是常数而是符号表达式，抛 `NotImplementedError`。
- 变量名与原模型保持一致；增量模型中新增输入 `q` 表示 $\dot u$。
- ⚠️ 返回的是**增量形式**模型。要得到真实的 $\Delta u$、$\Delta x$，需从 LQR 解中减去设定值。

### `dae2odeconversion(model) -> Model`

[`model/_dae2odeconversion.py`](../do_mpc/model/_dae2odeconversion.py)

把 **index-1 DAE** 转成等价 ODE（微分法）：

$$
\dot z = -\left(\frac{\partial g}{\partial z}\right)^{-1}\frac{\partial g}{\partial x}f
         -\left(\frac{\partial g}{\partial z}\right)^{-1}\frac{\partial g}{\partial u}\dot u
$$

转换后新模型的状态为 $[\dot x, \dot u, \dot z]$，输入为 $q$（即 $\dot u$）。

- **只支持 index-1 DAE**，高指标 DAE 无法处理。
- 主要用于让 **LQR** 能处理 DAE 系统（LQR 需要 `LinearModel`，而 `LinearModel` 不支持 `_z`）。
- 假设转换后的代数状态与状态测量都可获得。
- 调用后会 `print` 新模型的状态名。

### `IteratedVariables`

[`model/_iteratedvariables.py`](../do_mpc/model/_iteratedvariables.py) — **不可独立使用的基类**。

| 属性 | 说明 |
|---|---|
| `x0` | 初始状态 / 当前迭代值。可读写，支持按名索引。 |
| `u0` | 初始输入 |
| `z0` | 初始代数状态 |
| `t0` | 初始时刻 |

赋值时会自动做维度检查与类型转换（接受 `int`/`float`/`np.ndarray`/`casadi.DM`/`DMStruct`）：

```python
mpc.x0 = np.array([0.8, 0.5, 134.14, 130.0]).reshape(-1,1)   # 全部状态
mpc.x0['C_a'] = 0.8                                          # 单个状态
x0_val = mpc.x0.cat                                          # 取扁平向量
```

---

## `do_mpc.optimizer`

源码：[`do_mpc/optimizer.py`](../do_mpc/optimizer.py) — `MPC` 与 `MHE` 的共同基类，**不可独立使用**。

### 索引属性

| 属性 | 幂索引格式 | 说明 |
|---|---|---|
| `bounds` | `['lower'\|'upper', '_x'\|'_u'\|'_z'\|'_p_est', 名称, ...]` | 读写双向。`_p_est` 仅 MHE 有效。默认 `±inf`。 |
| `scaling` | `['_x'\|'_u'\|'_z'\|'_p', 名称, ...]` | 读写双向。默认 `1.0`。 |

```python
mpc.bounds['lower', '_x', 'phi_1'] = -2*np.pi
mpc.bounds['upper', '_x', 'phi_1'] =  2*np.pi
val = mpc.bounds['lower', '_x', 'phi_1']      # 也可以读
```

对应的底层结构：`lb_opt_x(ind)` / `ub_opt_x(ind)`（按整个优化变量向量索引），以及 `_x_lb`、`_x_ub`、`_u_lb`、`_u_ub`、`_z_lb`、`_z_ub`。

### 约束与目标

| 方法 | 说明 |
|---|---|
| ○ `set_nl_cons(expr_name, expr, ub=inf, soft_constraint=False, penalty_term_cons=1, maximum_violation=inf)` | 添加非线性约束 $m(x,u,z,p_{tv},p) \le m_{ub}$。`soft_constraint=True` 引入松弛变量 $\epsilon$ 并加入代价。返回新建的表达式。 |

### 时变参数

| 方法 | 说明 |
|---|---|
| ★ `get_tvp_template()` | 返回形状为 `(n_tvp, n_horizon+1)` 的数值结构体模板。 |
| ★ `set_tvp_fun(tvp_fun)` | `tvp_fun(t_now) -> template`。每次 `make_step` 内部调用。**若模型无 `_tvp` 可省略。** |

### NLP 生命周期

| 方法 | 说明 |
|---|---|
| ★ `setup()` | = `prepare_nlp()` + `create_nlp()`。一步完成。 |
| ○ `prepare_nlp()` | 只做离散化、场景树、变量结构组装，**不编译求解器**。之后可修改 NLP。 |
| ○ `create_nlp()` | 编译 CasADi `nlpsol` 求解器。 |
| ○ `compile_nlp(overwrite=False, cname='nlp.c', libname='nlp.so', compiler_command=None)` | 生成 C 代码并编译为共享库（部署用）。 |
| ⚙ `solve()` | 实际调用求解器。由 `make_step` 内部触发。 |
| ○ `reset_history()` | 清空 `data` 存储。 |

### NLP 符号属性（`prepare_nlp()` 之后可用）

| 属性 | 可读写 | 内容 |
|---|---|---|
| `nlp_obj` | ✔ | 目标函数（CasADi 符号） |
| `nlp_cons` | ✔ | 约束表达式 |
| `nlp_cons_lb` | ✔ | 约束下界（数值） |
| `nlp_cons_ub` | ✔ | 约束上界（数值） |
| `opt_x` | 只读 | 优化变量符号结构（`_x`, `_u`, `_z`, `_slack`, ...） |
| `opt_p` | 只读 | 优化参数符号结构（`_tvp`, `_p`, `x0`, `y_meas`, ...） |
| `opt_x_num` | ✔ | 优化变量的**数值**结构（当前解 / 初值） |
| `opt_p_num` | ✔ | 优化参数的数值结构 |
| `lb_opt_x` / `ub_opt_x` | 只读 | 扁平化的变量下/上界向量 |

⚙ 内部方法：`_setup_discretization()`（正交配置）、`_setup_scenario_tree()`（场景树）、`_setup_nl_cons()`、`_prepare_data()`。

---

## `do_mpc.controller`

源码：[`do_mpc/controller/`](../do_mpc/controller) · 导出：`MPC`, `MPCSettings`, `LQR`, `LQRSettings`

### `MPC(model, settings=None)`

[`controller/_mpc.py`](../do_mpc/controller/_mpc.py) — 继承 `Optimizer` + `IteratedVariables`

**配置流程（源码 docstring 中的 7 步）：**

1. ★ 通过 `mpc.settings.*` 配置（见 [04 · 配置参数速查](04-配置参数速查.md#mpcsettings)）
2. ★ `set_objective(mterm, lterm)` + `set_rterm(...)`
3. ○ `bounds[...]` 设置盒式约束
4. ○ `set_nl_cons(...)` 设置非线性/软约束
5. ○ 鲁棒 MPC：`set_uncertainty_values(...)`（高层）或 `get_p_template(n)` + `set_p_fun(...)`（低层）
6. ○ `get_tvp_template()` + `set_tvp_fun(...)`
7. ★ `setup()`（或 `prepare_nlp()` → 改 → `create_nlp()`）

| 方法 / 属性 | 说明 |
|---|---|
| ★ `set_objective(mterm=None, lterm=None)` | `lterm` = 阶段代价 $\sum_{k=0}^{N-1} l(x_k,u_k,z_k,p,p_{tv,k})$；`mterm` = 终端代价 $m(x_{N+1},\cdot)$。**两者都可以是任意 CasADi 表达式**（经济型 MPC 的关键）。至少要给一个。 |
| ★ `set_rterm(rterm=None, **kwargs)` | 输入增量惩罚 $\sum_k\sum_i r_i \Delta u_{k,i}^2$，其中 $\Delta u_k = u_k - u_{k-1}$。用关键字参数：`mpc.set_rterm(F=0.1, Q_dot=1e-3)`。v4.6.3 起可传 `rterm=<符号表达式>` 自定义惩罚形式（可引用自动生成的 `_u_prev`）。 |
| ○ `set_param(**kwargs)` | **旧接口**，内部转调 `settings`。仍可用。 |
| ○ `terminal_bounds['lower'\|'upper', 状态名]` | 终端约束。需配合 `settings.use_terminal_bounds = True`。索引只需 2 元组（无 `var_type`）。 |
| ○ `get_p_template(n_combinations)` | 低层鲁棒 API：返回形状含 `n_combinations` 维的参数模板。 |
| ○ `set_p_fun(p_fun)` | `p_fun(t_now) -> template`。 |
| ○ `set_uncertainty_values(**kwargs)` | 高层鲁棒 API：`set_uncertainty_values(alpha=[1.,1.05,.95])`，自动取笛卡尔积。 |
| ○ `copy_struct(original_struct)` | 从另一个结构体拷贝数值。 |
| ★ `set_initial_guess()` | 用当前 `x0`/`u0`/`z0` 填满整个时域作为初值。**`make_step` 前必须调用。** |
| ★ `make_step(x0)` | 求解 OCP，返回 $u_0$。内部会：调用 `tvp_fun`/`p_fun` → 更新 `opt_p_num` → `solve()` → 记录 `data` → 平移初值（warm start）。 |
| ○ `reset_history()` | 清空数据。 |
| `mpc.data` | `MPCData` 实例 |
| `mpc.settings` | `MPCSettings` 实例，可直接 `print` |

**运行时警告（源码原文）**：调用 `make_step` 前务必给 `x0`、`z0`、`u0` 赋有效初值并调用 `set_initial_guess()`；若要完全掌控初值，直接改 `opt_x_num`。

### `LQR(model)`

[`controller/_lqr.py`](../do_mpc/controller/_lqr.py) — 只接受 `LinearModel`，继承 `IteratedVariables`。

| 时域类型 | 设置 |
|---|---|
| **有限时域** | `lqr.settings.n_horizon = 20` |
| **无限时域** | `lqr.settings.n_horizon = None`（默认，求解离散代数 Riccati 方程） |

**两种工作模式：**

| 模式 | 步骤 |
|---|---|
| **标准模式** | `set_setpoint(xss, uss)` → `set_objective(Q, R, P=None)` → `setup()` |
| **输入增量惩罚模式** | `set_setpoint(...)` → `set_rterm(delR)` → `set_objective(Q, R)` → `setup()` |

| 方法 | 说明 |
|---|---|
| ★ `set_objective(Q, R, P=None)` | 状态权重 $Q$、输入权重 $R$、终端权重 $P$（有限时域用；不给则自动求解）。 |
| ○ `set_setpoint(xss=None, uss=None)` | 设定点，默认 `0`。**运行时可反复调用**以更新设定值。 |
| ○ `set_rterm(delR)` | 输入增量惩罚矩阵。⚠️ 若模型是由 DAE 转换而来，**不推荐**使用（转换后的模型已经是增量输入形式）。 |
| ○ `set_param(**kwargs)` | 旧接口（`n_horizon`, `t_step`）。 |
| ★ `setup()` | 完成配置。 |
| ★ `make_step(x0)` | 返回最优输入 $u$。 |
| ○ `discrete_gain(A, B)` | 由连续 $(A,B)$ 求离散增益。 |
| ○ `reset_history()` | 清空数据。 |
| ⚙ `_retreive_augmented_states(x, u)` | 增量模式下的增广状态处理。 |

---

## `do_mpc.estimator`

源码：[`do_mpc/estimator/`](../do_mpc/estimator) · 导出：`StateFeedback`, `Estimator`, `EKF`, `EstimatorSettings`, `MHE`, `MHESettings`

### `Estimator(model)` — 基类

[`estimator/_base.py`](../do_mpc/estimator/_base.py)

| 成员 | 说明 |
|---|---|
| `model` | 模型引用 |
| `data` | `Data` 实例，`data.dtype = 'Estimator'` |
| `x0`/`u0`/`z0`/`t0` | 来自 `IteratedVariables` |
| `reset_history()` | 清空 `data`（MHE 中被 `Optimizer` 版本覆盖） |

### `StateFeedback(model)`

最简单的"估计器"——`make_step(y0)` **原样返回** `y0`。

> 源码 docstring 自嘲："Why do you even bother to use this class?" —— 存在的意义是**保持闭环代码接口统一**，这样从全状态可测切换到 MHE/EKF 时，`main.py` 一行都不用改。

### `EKF(model)`

[`estimator/_ekf.py`](../do_mpc/estimator/_ekf.py) — 扩展卡尔曼滤波。

> ⚠️ **源码标注 "Work in progress"，且当前实现不支持 DAE 系统。**

| 成员 | 说明 |
|---|---|
| `settings` | `EstimatorSettings`（`n_horizon`, `t_step`） |
| `P0` | 初始误差协方差矩阵，默认 `np.eye(n_x)`，可读写 |
| ★ `setup()` | 生成 CasADi 雅可比函数 |
| ★ `make_step(y_next, u_next, Q_k, R_k)` | **注意签名与其他估计器不同**：需要显式传入过程噪声协方差 `Q_k` 与测量噪声协方差 `R_k`。 |
| ○ `set_initial_guess()` | 设置初值 |
| ○ `get_p_template()` / `set_p_fun(p_fun)` | 参数 |
| ○ `get_tvp_template()` / `set_tvp_fun(tvp_fun)` | 时变参数 |
| `flags` | `setup` / `set_initial_guess` / `set_tvp_fun` / `set_p_fun` / `first_step` |

示例见 [`examples/triple_tank_ekf/`](../examples/triple_tank_ekf)。

### `MHE(model, p_est_list=[])`

[`estimator/_mhe.py`](../do_mpc/estimator/_mhe.py) — 移动视界估计，多继承 `Optimizer` + `Estimator`。

| 参数 | 说明 |
|---|---|
| `model` | 已 `setup` 的模型 |
| `p_est_list` | **要估计的参数名列表**，必须是 `model` 中 `_p` 的子集。空列表 = 只估状态。 |

> **为什么参数要单独列出来？** 模型里可能同时有"外部影响参数"（如天气预报，不该被估计）和"内部系统参数"（如反应速率常数，应该被估计）。`p_est_list` 就是二者的分界线。参数在整个时域内恒定（不像状态那样逐步变化），所以放进 `_p` 而不是 `_x` 能显著减少优化变量数。

**配置流程：**

1. ★ `mhe.settings.*`
2. ★ `set_default_objective(P_x, P_v, P_p, P_w)`（推荐）或 `set_objective(stage_cost, arrival_cost)`（自定义）
3. ★ `bounds[...]`
4. ○ `set_nl_cons(...)`
5. ★ `get_p_template()` + `set_p_fun(...)` — **只需给出不被估计的那些参数**
6. ★ `get_y_template()` + `set_y_fun(...)` — 提供历史测量序列
7. ★ `get_tvp_template()` + `set_tvp_fun(...)`
8. ★ `setup()`

| 方法 / 属性 | 说明 |
|---|---|
| ★ `set_default_objective(P_x, P_v=None, P_p=None, P_w=None)` | 推荐的标准 MHE 目标：到达代价 $m(x_0,\tilde x_0,p,\tilde p) + \sum_{k=0}^{N-1} l(v_k,w_k,\cdot)$。<br/>• `P_x`：$n\times n$，初值偏差权重（到达代价）<br/>• `P_v`：$m\times m$，测量噪声权重<br/>• `P_p`：参数先验权重（**不估参数时不需要**）<br/>• `P_w`：过程噪声权重（**模型无 `process_noise` 时不需要**）<br/>💡 权重矩阵可以传 `numpy` 数组，**也可以传模型中定义的 `_p` / `_tvp` 符号**，从而实现时变权重。 |
| ○ `set_objective(stage_cost, arrival_cost)` | 低层接口，完全自定义两个代价项。 |
| ★ `get_y_template()` / `set_y_fun(y_fun)` | `y_fun(t_now)` 需返回最近 `n_horizon` 步的测量。典型实现直接从 `mhe.data._y` 取：<br/>`n_steps = min(mhe.data._y.shape[0], mhe.settings.n_horizon)`<br/>`for k in range(-n_steps, 0): y_template['y_meas', k] = mhe.data._y[k]` |
| ○ `get_p_template()` / `set_p_fun(p_fun)` | 只填**非估计**参数。 |
| ○ `set_param(**kwargs)` | 旧接口。 |
| ★ `set_initial_guess()` | 用 `x0`/`u0`/`z0`/`p_est0` 填满时域。 |
| ★ `make_step(y0)` | 传入最新测量，返回状态估计。 |
| `p_est0` | 可读写属性：待估参数的当前值/初值。 |
| `mhe._p_est` | 待估参数的符号结构，可用于 `set_nl_cons`（见示例中给 `Theta_1` 加界）。 |
| `opt_x_num` / `opt_p_num` / `opt_x` / `opt_p` | 同 `Optimizer`。 |
| `mhe.data` | `Data` 实例 |

示例见 [`examples/rotating_oscillating_masses_mhe_mpc/template_mhe.py`](../examples/rotating_oscillating_masses_mhe_mpc/template_mhe.py)。

---

## `do_mpc.simulator`

源码：[`do_mpc/simulator.py`](../do_mpc/simulator.py) · 导出：`Simulator`, `SimulatorSettings`, `ContinousSimulatorSettings`

### `Simulator(model)`

继承 `IteratedVariables`。用 CasADi 接口调用 SUNDIALS **CVODES**（ODE）/ **IDAS**（DAE）积分器。

**配置流程：**

1. ★ `simulator.settings.*`（连续模型自动得到 `ContinousSimulatorSettings`，离散模型得到 `SimulatorSettings`）
2. ○ `get_p_template()` + `set_p_fun(p_fun)`
3. ○ `get_tvp_template()` + `set_tvp_fun(tvp_fun)`
4. ★ `setup()`

| 方法 | 说明 |
|---|---|
| ○ `set_param(**kwargs)` | 旧接口（`t_step`, `abstol`, `reltol`, `integration_tool`, `integration_opts`）。示例中广泛使用。 |
| `scaling['_x'\|'_z', 名称]` | ⚠️ **只支持 `_x` 和 `_z`** —— 积分期间 `u` 和 `p` 恒定，无需缩放。 |
| ○ `get_p_template()` / `set_p_fun(p_fun)` | `p_fun(t_now) -> 单值结构体` |
| ○ `get_tvp_template()` / `set_tvp_fun(tvp_fun)` | `tvp_fun(t_now) -> 单值结构体`（**一维**，与 MPC/MHE 的二维不同） |
| ○ `set_initial_guess()` | 为代数变量提供初值 |
| ○ `init_algebraic_variables()` | 初始化 DAE 的代数变量（首步自动调用） |
| ★ `setup()` | 生成积分器 |
| ★ `make_step(u0=None, v0=None, w0=None)` | 积分一个 `t_step`，返回测量值 $y_{next}$。<br/>• `u0`：控制输入<br/>• `v0`：**测量噪声采样值**（需用户自己生成随机数，维度 = `model.n_v`）<br/>• `w0`：**过程噪声采样值**（维度 = `model.n_w`） |
| ⚙ `simulate()` | 底层积分调用 |
| ○ `reset_history()` | 清空 `data` |
| `simulator.data` | `Data` 实例 |
| ⚙ `_check_validity()` | 内部一致性检查 |

---

## `do_mpc.data`

源码：[`do_mpc/data.py`](../do_mpc/data.py) · 导出：`Data`, `MPCData`, `save_results`, `load_results`

### `Data(model)`

| 成员 | 说明 |
|---|---|
| `data[var_type, var_name]` | `__getitem__`，例如 `data['_x','C_a']` → 形状 `(n_t, ...)` |
| `data[var_type, t_ind, var_name]` | 指定时刻，`t_ind=-1` 为最新 |
| `data.dtype` | `'default'` / `'Estimator'` / `'MPC'` |
| `data.model` | 模型引用 |
| `data.meta` | 元信息字典 |
| ⚙ `init_storage()` | 重置存储 |
| ⚙ `set_meta(**kwargs)` | 写元信息 |
| ⚙ `update(**kwargs)` | 追加一个时刻的数据（由 `make_step` 内部调用） |
| ○ `export()` | 导出为字典 |

可查询的键：`_x`, `_u`, `_z`, `_aux`, `_tvp`, `_p`, `_y`, `_w`, `_v`（视类而定），MPC 额外有 `_time`, `_solver_stats`, `opt_x` 等。

### `MPCData(model)` — 继承 `Data`

| 方法 | 说明 |
|---|---|
| `prediction(ind, t_ind=-1)` | 查询 MPC 预测轨迹。`ind` 是 `('_x', 'C_a')` 这样的元组，`t_ind` 是哪个时刻做的预测。内部有 `prediction_queries` 缓存以加速重复查询。 |

> ⚠️ **鲁棒多阶段 MPC 下默认只存名义参数对应的预测轨迹。**

### 模块级函数

| 函数 | 说明 |
|---|---|
| `save_results(save_list, result_name='results', result_path='./results/', overwrite=False)` | 把 `MPC` / `Simulator` / (`StateFeedback`\|`EKF`\|`MHE`) 的 `.data` 打包成 pickle。字典键为 `'mpc'` / `'simulator'` / `'estimator'`。<br/>`overwrite=False` 时若文件已存在会自动加 `001_`、`002_` 序号前缀。<br/>目录不存在会自动创建。传入其他类型对象抛 `Exception`。 |
| `load_results(file_name)` | 反序列化，返回上述字典。 |

```python
do_mpc.data.save_results([mpc, simulator], 'CSTR_robust_MPC')
res = do_mpc.data.load_results('./results/CSTR_robust_MPC.pkl')
graphics = do_mpc.graphics.Graphics(res['mpc'])
```

---

## `do_mpc.graphics`

源码：[`do_mpc/graphics.py`](../do_mpc/graphics.py) · 导出：`Graphics`, `default_plot`, `animate`

**完全独立于其他模块**，可选使用；也能直接吃 pickle 结果文件做离线后处理。基于 Matplotlib，输出可达出版级质量。

### `Graphics(data)`

`data` 是 `Data` 或 `MPCData` 实例。

| 方法 | 说明 |
|---|---|
| ★ `add_line(var_type, var_name, axis, **pltkwargs)` | 把某个变量绑定到某个 `matplotlib.axes.Axes`。`pltkwargs` 直接透传给 `ax.plot`（颜色、线型等）。返回创建的空 `Line2D` 对象。 |
| ★ `plot_results(t_ind=-1)` | 用最新数据刷新所有结果曲线。`t_ind` 指定画到哪个时刻。 |
| ○ `plot_predictions(t_ind=-1)` | 画 MPC 预测轨迹。**仅对 `MPCData` 有效**，且需要 `store_full_solution=True`。 |
| ○ `reset_axes()` | 重置坐标轴范围（配合 `plt.pause` 做动画必需）。 |
| ○ `reset_prop_cycle()` | 重置颜色循环。 |
| ○ `clear(lines=None)` | 清除曲线数据。 |
| `result_lines[var_type, var_name]` | 已创建的结果曲线对象列表，可用于 `ax.legend(...)`。 |
| `pred_lines[var_type, var_name]` | 预测曲线对象列表。 |
| `result_lines.full` / `pred_lines.full` | 全部曲线，便于批量改样式：<br/>`for line_i in graphics.pred_lines.full: line_i.set_linewidth(1)` |

**典型用法：**

```python
graphics = do_mpc.graphics.Graphics(mpc.data)
fig, ax = plt.subplots(5, sharex=True)
graphics.add_line(var_type='_x',   var_name='C_a',   axis=ax[0])
graphics.add_line(var_type='_aux', var_name='T_dif', axis=ax[2])
graphics.add_line(var_type='_u',   var_name='Q_dot', axis=ax[3])
ax[0].set_ylabel('c [mol/l]')
fig.align_ylabels(); fig.tight_layout(); plt.ion()

for k in range(50):
    ...                                   # 闭环三步
    graphics.plot_results(t_ind=k)
    graphics.plot_predictions(t_ind=k)
    graphics.reset_axes()
    plt.show(); plt.pause(0.01)
```

### `default_plot(data, states_list=None, dae_states_list=None, inputs_list=None, aux_list=None, **kwargs)`

**高层快捷 API**：一次画出全部状态、输入、辅助表达式（各占一个子图）。返回 `(fig, ax, graphics)`。

- 传入名称列表可只画子集（必须是模型中已定义名称的子集，否则断言失败）。
- `**kwargs` 透传给 `plt.subplots(n_plot, 1, sharex=True, **kwargs)`，例如 `figsize=(16,9)`。
- **推荐在开发调试阶段用它**，成型后再换成手工配置的 `Graphics`。

```python
fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data, figsize=(16,9))
```

### `animate(graphics, ...)`

把 `plot_results` / `plot_predictions` / `reset_axes` 封装成 `matplotlib.animation` 的 `update` 回调，用于导出 GIF/MP4。需要 `store_full_solution=True`。

参考 [`documentation/source/Graphics.rst`](../documentation/source/Graphics.rst)。

---

## `do_mpc.tools`

源码：[`do_mpc/tools/`](../do_mpc/tools) · 导出：`Timer`, `Structure`, `IndexedProperty`, `_struct_SX`, `_struct_MX`, `save_pickle`, `load_pickle`, `printProgressBar`

| 名称 | 位置 | 说明 |
|---|---|---|
| **`Timer(name='timer', unit='ms')`** | [`_timer.py`](../do_mpc/tools/_timer.py) | 计时器。`tic()` 开始、`toc()` 结束并记录、`info()` 打印统计（均值/最坏/最好）、`hist(**kwargs)` 画直方图。**所有示例的 `main.py` 都用它测求解耗时。** |
| **`Structure`** | [`_structure.py`](../do_mpc/tools/_structure.py) | 支持多元组幂索引的通用容器基类，`bounds` / `scaling` / `result_lines` 都基于它。`full` 属性返回全部条目。 |
| **`IndexedProperty`** | [`_indexedproperty.py`](../do_mpc/tools/_indexedproperty.py) | 装饰器，把 `property` 扩展成"带索引的 property"，从而支持 `obj.bounds['lower','_x','name'] = v` 这种赋值语法。 |
| **`_struct_SX` / `_struct_MX`** | [`_casstructure.py`](../do_mpc/tools/_casstructure.py) | 对 `casadi.tools.struct_SX/MX` 的薄封装。 |
| `save_pickle(filename, data)` | [`__init__.py`](../do_mpc/tools/__init__.py) | 自动补 `.pkl` 后缀并写盘。 |
| `load_pickle(path_to_file)` | 同上 | 读盘。 |
| `printProgressBar(iteration, total, ...)` | 同上 | 控制台进度条（采样工具中大量使用）。 |

```python
from do_mpc.tools import Timer
timer = Timer()
for k in range(50):
    timer.tic()
    u0 = mpc.make_step(x0)
    timer.toc()
timer.info()      # Average time / worst case / best case
timer.hist()      # matplotlib 直方图
```

---

## 附：完整调用顺序备忘

```
Model:      set_variable* → set_expression/set_meas/set_alg → set_rhs* → setup()
                                                                            │
        ┌───────────────────────────────────────────────────────────────────┤
        ▼                                                                   ▼
MPC:  settings.* → set_objective → set_rterm → bounds → set_nl_cons   Simulator:
      → set_uncertainty_values / (get_p_template + set_p_fun)           settings.* →
      → get_tvp_template + set_tvp_fun → setup()                        (get_p_template + set_p_fun) →
      → x0=... → set_initial_guess()                                    (get_tvp_template + set_tvp_fun) →
        │                                                               setup() → x0=... → set_initial_guess()
        ▼                                                                 │
MHE:  settings.* → set_default_objective → bounds → set_nl_cons           │
      → get_p_template + set_p_fun → get_y_template + set_y_fun           │
      → get_tvp_template + set_tvp_fun → setup()                          │
      → x0=/p_est0=... → set_initial_guess()                              │
        │                                                                 │
        └────────────►  闭环循环: mpc.make_step → sim.make_step → est.make_step  ◄──┘
                                     │
                                     ▼
                        data.save_results / Graphics
```
