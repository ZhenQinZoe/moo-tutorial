
# Chapter 3: Distributed Multi-Objective Optimization

[TOC]

Gradient-based multi-objective optimization (MOO) methods such as MGDA assume that one machine can evaluate every objective's gradient at the current model and combine these gradients into a common descent direction. In many applications, however, data and objectives are spread across devices or organizations that train collaboratively without sharing raw data. In an e-commerce recommender system, for example, different customer groups may care about different combinations of price, brand, and delivery speed (Yang et al., 2023); mobile robots planning trajectories may each weigh fuel usage against travel time differently (Blondin & Hale, 2021). An update that improves one participant's objective may then hurt another's. Minimizing a single average loss can hide these differences, while a fixed weighted sum commits to a particular trade-off that may not suit all participants. Distributed MOO makes these competing goals explicit and seeks Pareto trade-offs, where improving one objective requires sacrificing another. This allows collaboration to account for individual objectives alongside overall performance.

Different architectures organize this collaboration in different ways. In **federated learning**, a central server typically collects local updates and coordinates how objectives are balanced. In **decentralized learning**, participants communicate directly with their neighbors and must coordinate both their models and the trade-offs they pursue without a central server. **Hierarchical architectures** combine multiple levels of coordination, such as local groups and a global server. This chapter focuses on federated and decentralized MOO, examining how each architecture balances competing objectives using local information and limited communication.

Distribution changes the MGDA recipe in three ways.

- **Inexact gradients.** No participant sees the global gradients. In federated learning, the server receives updates from a sample of clients, computed along separate local trajectories, so the gradients that MGDA combines are only estimated.
- **Communication.** With $S$ objectives and a model of $d$ parameters, sending one update per objective costs $\Theta(Sd)$ values per participant per round, which grows with the number of objectives.
- **No common coordinator.** Without a server, agents must agree on both the model and the trade-off through neighbor exchanges alone.

:::danger
**Central question.** Can participants that hold only local data, and coordinate through a server or only with their neighbors, still converge to a Pareto-stationary shared model with an agreed trade-off, while tolerating local training and sending close to one model-sized message per round?
:::

### Roadmap

**Section 3.1** studies federated MOO, where a server coordinates the clients. It first identifies where one federated round departs from MGDA and then follows four methods that address them. **Section 3.2** studies decentralized MOO, where agents communicate only with their neighbors, and **Section 3.3** compares the two architectures and discusses open directions.

| Section | Architecture | Coordination problem | Representative methods |
|---|---|---|---|
| 3.1 | **Federated**: a server aggregates client updates | Combine updates that favor different objectives, computed by heterogeneous clients after local training | FedMGDA+ (Hu et al., 2022), FMOL (Yang et al., 2023), FedCMOO (Askin et al., 2025), DREAM (Zhou et al., 2026) |
| 3.2 | **Decentralized**: agents exchange information with their neighbors | Agree on both the model and the preferences over objectives without a server | Blondin & Hale (2021) |

For each method, we explain the key idea, present the algorithm, and state what its guarantee certifies and what it leaves out.

---

## 3.1 Federated Multi-Objective Learning

### From Client Conflicts to Communication Efficiency and Drift Correction

Federated learning distributes model training across clients while keeping their datasets local. A server coordinates training by collecting client updates and producing a shared model. When several objectives must be optimized, this coordination involves an additional decision: how should the system combine updates that favor different objectives?

:::info
**Insight.** The difficulty comes from two interacting dimensions. Different objectives may prefer different directions, while different clients may provide different estimates of the same objective. Local training introduces further changes because clients evaluate gradients along their own optimization trajectories. Consequently, the server must coordinate objectives using information that is both distributed and approximate.
:::

We assume familiarity with MGDA and focus on what changes when objectives and data are distributed.

<div style="background-color:#f2f2f2; border-left:5px solid #9e9e9e; padding:12px 16px; border-radius:4px; margin-bottom:16px;">

**Notation.** Following Chapter 0, we use $\mathbf{x}\in\mathbb{R}^d$ for the model parameters, $S$ for the number of objectives, and $\mathbf{F}(\mathbf{x})=[f_1(\mathbf{x}),\ldots,f_S(\mathbf{x})]^\top$ for the objective vector. We additionally use $N$ for the number of clients, $f_{s,i}(\mathbf{x})$ for the local loss of objective $s$ at client $i$, and $\mathbf{J}(\mathbf{x}):=[\nabla f_1(\mathbf{x}),\ldots,\nabla f_S(\mathbf{x})]\in\mathbb{R}^{d\times S}$ for the matrix of global objective gradients, so that $\mathbf{J}(\mathbf{x})\boldsymbol{\lambda}=\sum_s\lambda_s\nabla f_s(\mathbf{x})$. Vectors and matrices are written in bold; individual losses and weight components are scalars. The probability simplex is $\Delta_m:=\{\boldsymbol{\lambda}\in\mathbb{R}^m:\lambda_j\geq0,\ \sum_{j=1}^{m}\lambda_j=1\}$ for any positive integer $m$.

</div>

### Where a Federated Round Departs from MGDA

With access to all data, MGDA would compute the global gradients at the current model and set
$$
\boldsymbol{\lambda}^t=\arg\min_{\boldsymbol{\lambda}\in\Delta_S}\big\|\mathbf{J}(\mathbf{x}^t)\boldsymbol{\lambda}\big\|^2,
\qquad
\mathbf{x}^{t+1}=\mathbf{x}^t-\eta\,\mathbf{J}(\mathbf{x}^t)\boldsymbol{\lambda}^t.
$$
The MGDA direction vanishes exactly when the Pareto-stationarity residual
$$
\mathcal{R}(\mathbf{x}):=\min_{\boldsymbol{\lambda}\in\Delta_S}\big\|\mathbf{J}(\mathbf{x})\boldsymbol{\lambda}\big\|^2
$$
is zero, that is, at Pareto-stationary points. Most of the guarantees in this section bound $\mathcal{R}$ or a quantity that upper-bounds it.

A federated round proceeds differently: the server broadcasts $\mathbf{x}^t$, sampled clients train locally for several steps, the clients upload their updates, and the server computes weights from these updates and moves the model. The server therefore never sees $\mathbf{J}(\mathbf{x}^t)$, only an estimate $\widehat{\mathbf{J}}^t$ assembled from client updates. Five departures from MGDA shape the methods in this section.

| Departure | What changes in the weight problem | Treated in |
|---|---|---|
| **Client sampling** | The weights are computed from a random subset of clients, whose objectives may conflict | FedMGDA+ (3.1.1); FedCMOO (3.1.3) also bounds its error |
| **Uneven task assignment** | Each task's gradient is averaged over a different subset of clients | FMOL (3.1.2) |
| **Local drift** | Updates are computed along local trajectories rather than at $\mathbf{x}^t$ | FMOL (3.1.2) bounds it; FedCMOO (3.1.3) reduces it; DREAM (3.1.4) corrects it |
| **Communication** | One update per objective costs $\Theta(Sd)$ per client per round | FedCMOO (3.1.3) |
| **Nonlinear weighting** | Weights computed from noisy estimates are biased even when the estimates are unbiased | DREAM (3.1.4) |

The four methods appear in chronological order; the table shows which departures each one targets.

### 3.1.1 FedMGDA+: Treating Clients as Objectives

FedMGDA+ is the earliest of the four methods. It brings MGDA into federated learning by treating each client as an objective, and its analysis accounts for the first departure, client sampling.

#### Why Average Performance Is Not Enough

A standard federated objective takes the form

$$
\min_{\mathbf{x}\in\mathbb{R}^d} f_{\mathrm{avg}}(\mathbf{x})
=\sum_{i=1}^{N}p_i f_i(\mathbf{x}),
\qquad
p_i\geq0,\quad \sum_{i=1}^{N}p_i=1,
$$

where $f_i$ is the loss of client $i$.

A decrease in $f_{\mathrm{avg}}$ can hide an increase in an individual client's loss. To see this, consider an update $\mathbf{x}^+=\mathbf{x}-\eta\mathbf{d}$. Its first-order effect on client $i$ is

$$
f_i(\mathbf{x}^+)-f_i(\mathbf{x})
\approx-\eta\langle\nabla f_i(\mathbf{x}),\mathbf{d}\rangle.
$$

The weighted average can improve even when some inner products are negative. The aggregation weights therefore influence which clients benefit from training.

#### Preserve Individual Client Objectives

Hu et al. (2022) instead treat each client's loss as a separate objective, which gives the vector problem

$$
\min_{\mathbf{x}\in\mathbb{R}^d}\mathbf{F}(\mathbf{x})
:=\begin{bmatrix}f_1(\mathbf{x})&\cdots&f_N(\mathbf{x})\end{bmatrix}^\top
$$

with $S=N$ objectives.

:::warning
**Key idea.** Run MGDA on the sampled clients' updates, treating each update as one objective's gradient, so that aggregation seeks a common descent direction instead of a fixed average. Normalize the updates first so that no client dominates by scale (Hu et al., 2022).
:::

FedMGDA+ also restricts the adaptive weights to stay near a prescribed vector $\mathbf{p}$, for example the FedAvg weights:

$$
\boldsymbol{\lambda}\in\Delta_N,
\qquad
\|\boldsymbol{\lambda}-\mathbf{p}\|_\infty\leq\epsilon.
$$

The radius $\epsilon$ sets how far aggregation may move away from these weights, balancing average performance against client-level trade-offs: $\epsilon=0$ recovers FedAvg with weights $\mathbf{p}$, and $\epsilon=1$ recovers MGDA (Hu et al., 2022).

:::info
**Algorithm 1: FedMGDA+ (one round)**

1. **Client sampling.** The server chooses a subset $I_t$ of the clients and broadcasts $\mathbf{x}^t$.
2. **Local training.** Each client $i\in I_t$ runs $k$ local epochs of minibatch gradient steps on $f_i$, starting from $\mathbf{x}^t$, and returns its accumulated update $\mathbf{g}_i$, its starting model minus its final local model.
3. **Normalization.** The server normalizes each update, $\bar{\mathbf{g}}_i=\mathbf{g}_i/\|\mathbf{g}_i\|$.
4. **Weights.** The server solves
   $$\boldsymbol{\lambda}^*=\arg\min_{\boldsymbol{\lambda}\in\Delta_{|I_t|},\ \|\boldsymbol{\lambda}-\mathbf{p}_{I_t}\|_\infty\leq\epsilon}\Big\|\sum_{i\in I_t}\lambda_i\bar{\mathbf{g}}_i\Big\|^2,$$
   where $\mathbf{p}_{I_t}\in\Delta_{|I_t|}$ is the prescribed weighting of the sampled clients.
5. **Model update.** $\mathbf{x}^{t+1}=\mathbf{x}^t-\eta_t\sum_{i\in I_t}\lambda^*_i\bar{\mathbf{g}}_i$.
:::

#### Guarantees

**Assumptions.** Client losses are smooth and Lipschitz continuous, and at least one client loss is bounded from below. The analysis covers the single-local-update setting, with one local gradient step per round, and requires suitable step sizes and a sufficiently small error in the estimated common descent direction. Theorem 1a takes the weights from the whole simplex, that is, $\epsilon=1$.

**Metric.** The analysis bounds the Pareto-stationarity residual $\mathcal{R}(\mathbf{x})$ defined at the start of Section 3.1, with the client losses as the objectives ($S=N$).

:::success
**Theorem (Hu et al., 2022, Theorems 1a and 2).** Under these assumptions, FedMGDA+ satisfies
$$
\lim_{T\to\infty}
\min_{0\leq t\leq T}
\mathbb{E}\bigl[\mathcal{R}(\mathbf{x}^t)\bigr]=0.
$$
In the same single-local-update setting, if all client losses are convex and each round includes a strongly convex client objective whose weight is bounded away from zero, choosing $\eta_t=2/[c(t+2)]$ yields
$$
\mathbb{E}\!\left[
\operatorname{dist}(\mathbf{x}^t,\mathcal{X}_{\mathrm{PS}})^2
\right]
\leq \frac{4M^2}{c^2(t+3)}
=O(1/t),
$$
where $\mathcal{X}_{\mathrm{PS}}$ is the Pareto-stationary set, $M$ is the Lipschitz constant, and $c>0$ captures the curvature condition. The distance to this set also converges to zero almost surely (Hu et al., 2022).
:::

Theorem 1a gives no rate. How fast the residual vanishes depends on how quickly the error of the estimated common descent direction decreases, which in turn depends on how aggressively clients are subsampled and how heterogeneous they are (Hu et al., 2022).

#### What Common Descent Does Not Guarantee

:::danger
**Limitations.** With exact gradients and suitable steps, a common descent direction improves participating objectives. Stochastic estimates and multiple local steps complicate this property; the practical algorithm should not be interpreted as guaranteeing that every client's loss decreases in every round. Common descent is also a different notion of fairness from uniform test accuracy across clients; the paper evaluates the two separately (Hu et al., 2022).
:::

### 3.1.2 FMOL: Separating Clients and Tasks

FedMGDA+ associates one objective with each client. A broader setting requires separate representations for clients and tasks: several clients may contribute to the same task, and one client may contribute to several tasks.

#### A General Federated Formulation

Yang et al. (2023) introduce a framework with $N$ clients and $S$ objectives, which they call tasks. Let $f_{s,i}(\mathbf{x})$ denote the local loss for task $s$ at client $i$, and let $\mathcal{C}_s$ contain the clients contributing to that task. The global objective of task $s$ is

$$
f_s(\mathbf{x})=\frac{1}{|\mathcal{C}_s|}
\sum_{i\in\mathcal{C}_s}f_{s,i}(\mathbf{x}).
$$

With these global scalar losses $f_s$, the system jointly optimizes

$$
\min_{\mathbf{x}\in\mathbb{R}^d}\mathbf{F}(\mathbf{x})
:=\begin{bmatrix}f_1(\mathbf{x})&\cdots&f_S(\mathbf{x})\end{bmatrix}^\top.
$$

:::info
**Insight.** Clients may possess different task subsets. FMOL therefore distinguishes objective heterogeneity from data heterogeneity (Yang et al., 2023).
:::

| Dimension | What varies? | Optimization consequence |
|---|---|---|
| Across tasks | The global objectives $f_s$ | Task directions can conflict |
| Across clients | Local functions $f_{s,i}$ for a given task | Local updates can differ |
| Across task assignments | The sets $\mathcal{C}_s$ | Tasks receive information from different clients |

:::warning
**Key idea.** Aggregate in two stages: average the updates *within* each task to estimate that task's global gradient, then run MGDA *across* tasks on these averages to balance the objectives.
:::

#### How One Training Round Works

FMOL introduces FMGDA and FSMGDA, using full and stochastic gradients, respectively (Yang et al., 2023).

:::info
**Algorithm 2: FMGDA / FSMGDA (one round)**

1. **Local training.** Each client initializes a separate local trajectory for each available task from the broadcast model $\mathbf{x}^t$ and performs local updates.
2. **Upload.** Each client returns one accumulated update per available task, the sum of its local gradients for that task.
3. **Task averaging.** The server averages these updates within each task, giving $\mathbf{u}_s^t$.
4. **Weights.** The server solves a simplex-constrained minimum-norm problem across tasks:
   $$\boldsymbol{\lambda}_t=\arg\min_{\boldsymbol{\lambda}\in\Delta_S}\left\|\sum_{s=1}^{S}\lambda_s\mathbf{u}_s^t\right\|^2.$$
5. **Model update.** $\mathbf{x}^{t+1}=\mathbf{x}^t-\eta_g\sum_{s=1}^{S}\lambda_{t,s}\mathbf{u}_s^t$.
:::

#### Why Local Training Changes the Geometry

For a local trajectory $\mathbf{x}_{s,i}^{t,k}$, the discrepancy from the gradient at the broadcast model can be decomposed as

$$
\begin{aligned}
&\nabla f_{s,i}(\mathbf{x}_{s,i}^{t,k})-\nabla f_s(\mathbf{x}^t)\\
&=\underbrace{
\nabla f_{s,i}(\mathbf{x}_{s,i}^{t,k})
-\nabla f_{s,i}(\mathbf{x}^t)
}_{\text{movement along the local trajectory}}
+\underbrace{
\nabla f_{s,i}(\mathbf{x}^t)-\nabla f_s(\mathbf{x}^t)
}_{\text{client heterogeneity}}.
\end{aligned}
$$

Under smoothness, the first term grows with the distance traveled locally. Even when averaging cancels client differences at a shared point, clients subsequently evaluate gradients at different points. Accumulated updates can therefore differ from the columns of $\mathbf{J}(\mathbf{x}^t)$ that MGDA needs.

FMOL separates local and server learning rates to keep these errors small (Yang et al., 2023); DREAM (Section 3.1.4) instead corrects the drift itself.

#### Guarantees

**Assumptions.** For non-convex objectives, FMGDA assumes $L$-smooth global losses, bounded local gradients, and at least one lower-bounded objective. FSMGDA instead assumes unbiased stochastic gradients with bounded second moments, at least one lower-bounded objective, and the regularity condition
$$
\mathbb{E}\left[
\|\nabla f_{s,i}(\mathbf{x};\xi)
-\nabla f_{s,i}(\mathbf{y};\xi')\|^2
\right]
\leq
\alpha\|\mathbf{x}-\mathbf{y}\|^2+\beta\sigma^2,
$$
where $\xi$ and $\xi'$ are independent samples.

**Metric.** The analysis measures the squared norm of the true global gradients combined using the algorithm's weights:
$$
\mathbf{g}_t
:=\sum_{s=1}^S\lambda_{t,s}\nabla f_s(\mathbf{x}^t).
$$
Since $\boldsymbol{\lambda}_t\in\Delta_S$, this upper-bounds the Pareto-stationarity residual:
$$
\mathcal{R}(\mathbf{x}^t)\leq\|\mathbf{g}_t\|^2.
$$

:::success
**Theorem (Yang et al., 2023, Corollaries 2 and 6).** For a fixed number of local steps, FMGDA uses a suitable constant server learning rate and a local learning rate $\eta_L=O(T^{-1/2})$, yielding
$$
\min_{0\leq t<T}\mathcal{R}(\mathbf{x}^t)
\leq
\min_{0\leq t<T}\|\mathbf{g}_t\|^2
=O(T^{-1}).
$$
FSMGDA uses $\eta_g=O(T^{-1/2})$, $\eta_L=O(T^{-1/4})$, and additionally requires $\beta=O(\eta_g)$, yielding
$$
\min_{0\leq t<T}\mathbb{E}\!\left[
\mathcal{R}(\mathbf{x}^t)\right]
\leq
\min_{0\leq t<T}\mathbb{E}\|\mathbf{g}_t\|^2
=O(T^{-1/2}).
$$
In the $(\alpha,\beta)$ condition, $\sigma^2$ is a generic constant rather than the gradient variance (Yang et al., 2023).
:::

FMGDA's $O(1/T)$ rate matches that of centralized MGD, and with a single client FMGDA reduces to MGD. The underlying bound (Yang et al., 2023, Theorem 1) separates an optimization error, which vanishes as $T$ grows, from an error caused by local steps and data heterogeneity, which the local learning rate $\eta_L$ controls.

### 3.1.3 FedCMOO: Computing Weights from Gradient Relationships

Local steps reduce how often clients communicate, but not how much: in FMOL each client still uploads one $d$-dimensional update per task, $\Theta(Sd)$ values per round. FedCMOO removes this dependence on $S$.

#### What the Server Actually Needs

The minimum-norm weight problem depends on the relationships between gradients. With the gradient matrix $\mathbf{J}(\mathbf{x})$ from the notation, define

$$
\mathbf{Q}(\mathbf{x})=\mathbf{J}(\mathbf{x})^\top\mathbf{J}(\mathbf{x}).
$$

Then

$$
\|\mathbf{J}(\mathbf{x})\boldsymbol{\lambda}\|^2
=\boldsymbol{\lambda}^\top\mathbf{Q}(\mathbf{x})\boldsymbol{\lambda}.
$$

The diagonal entries of $\mathbf{Q}$ describe gradient magnitudes. Its off-diagonal entries describe alignment or conflict. The min-norm weights therefore depend on the gradients only through the $S\times S$ matrix $\mathbf{Q}$, so the server needs $\mathbf{Q}$ rather than the $d\times S$ matrix $\mathbf{J}$.

A useful distinction is that a global Gram matrix generally cannot be obtained by averaging local Gram matrices. If $\mathbf{J}=\sum_i p_i\mathbf{J}_i$, then

$$
\mathbf{J}^\top\mathbf{J}
=\sum_{i,j}p_ip_j\mathbf{J}_i^\top\mathbf{J}_j.
$$

The cross-client terms also contribute.

#### Determine Weights Before Local Training

Askin et al. (2025) propose FedCMOO, which estimates $\mathbf{Q}$ from compressed client information at the start of each round and fixes the task weights before local training. Because local Gram matrices cannot simply be averaged, the clients send compressed Jacobians; the server averages these and then forms the Gram matrix of the average.

:::info
**Algorithm 3: FedCMOO (one round)**

1. **Jacobian sketch.** The server samples clients, which send compressed Jacobian information that helps the server approximate the global Gram matrix.
2. **Weights.** The server updates the task weights $\boldsymbol{\lambda}_t$ and broadcasts them.
3. **Local training.** Each client trains on the weighted local loss
   $$L_i^t(\mathbf{x})=\sum_{s=1}^{S}\lambda_{t,s}f_{s,i}(\mathbf{x}).$$
4. **Upload.** Each client uploads one aggregate model update.
5. **Model update.** The server averages these updates and takes a global step.
:::

:::warning
**Key idea.** Fix the task weights at the start of each round, before local training, so that each client trains on one weighted loss and uploads one update, instead of training and uploading one update per task as in FMOL.
:::

Because each client follows one trajectory instead of one per task, local models drift less than in FSMGDA; the experiments of Askin et al. (2025) illustrate this effect.

#### Communication Savings

Each sampled client uploads two model-sized messages per round: its compressed Jacobian, whose rank is chosen so that it has about $d$ entries, and its model update. The upload is therefore $\Theta(d)$ per client per round instead of $\Theta(Sd)$. The savings are in communication only: local training still computes stochastic gradients for all $S$ objectives (Askin et al., 2025).

#### Guarantees

**Assumptions.** Local losses are smooth, with bounded gradients and bounded client heterogeneity. Stochastic gradients are unbiased with bounded variance. The analyzed variant uses unbiased compression with controlled variance, independent samples for Gram-matrix estimation, and one weight-update step per round.

**Metric.** The analysis measures the time-averaged expected squared norm of the weighted global gradient:
$$
\frac{1}{T}\sum_{t=0}^{T-1}
\mathbb{E}\|\mathbf{g}_t\|^2,
\qquad
\mathbf{g}_t
:=\sum_{s=1}^S\lambda_{t,s}\nabla f_s(\mathbf{x}^t).
$$
Since $\boldsymbol{\lambda}_t\in\Delta_S$, the Pareto-stationarity residual satisfies
$$
\mathcal{R}(\mathbf{x}^t)\leq\|\mathbf{g}_t\|^2.
$$

:::success
**Theorem (Askin et al., 2025, Corollary 1.1).** Let $K$ denote the number of local steps. With client learning rate $\eta_L=1/(LK\sqrt{KT})$, server learning rate $\eta_g=\sqrt{K}$, and weight-update step size $\eta_\lambda=1/(S\sqrt{T})$, FedCMOO satisfies
$$
\frac{1}{T}\sum_{t=0}^{T-1}
\mathbb{E}\!\left[\mathcal{R}(\mathbf{x}^t)\right]
\leq
\frac{1}{T}\sum_{t=0}^{T-1}
\mathbb{E}\|\mathbf{g}_t\|^2
\leq \frac{C}{\sqrt{T}},
$$
where $C$ is independent of $T$ but depends on problem and algorithm parameters. Consequently, achieving an expected squared stationarity residual at most $\delta$ for a uniformly sampled iterate requires $T=O(\delta^{-2})$ communication rounds (Askin et al., 2025).
:::

The underlying bound (Askin et al., 2025, Theorem 1) separates the errors caused by changing weights, partial client participation, and local drift. The rate matches the best-known rate for single-objective federated learning under similar conditions. Because local training computes stochastic gradients for all $S$ objectives, the sample complexity grows linearly in $S$, whereas Askin et al. (2025) report $O(S^4)$ for FSMGDA. The analysis also drops the requirement $\beta=O(\eta_g)$ that FSMGDA's stochastic rate needs.

:::danger
**Limitations.** The practical implementation compresses with randomized SVD, which is biased, so the theorem covers the simplified variant described above rather than the algorithm run in the experiments. In addition, with $\eta_\lambda=0$ FedCMOO reduces to FedAvg on a fixed weighted sum, and the bound keeps the same dominant terms. The $O(1/\sqrt{T})$ rate therefore certifies convergence to Pareto stationarity but not a benefit from adapting the weights; the paper supports that benefit with experiments (Askin et al., 2025).
:::

### 3.1.4 DREAM: Correcting Drift Before Adaptive Aggregation

Communication efficiency answers how much information is transmitted. Directional reliability asks whether the transmitted information leads to an appropriate multi-objective update.

#### Two Sources of Directional Error

Local training can shift task updates away from global gradients. Adaptive aggregation introduces another difficulty because its weights depend nonlinearly on the estimated gradients.

Let $\mathcal{G}(\mathbf{J})$ denote the operation that computes weights from $\mathbf{J}$ and returns their combined direction. Even if $\widehat{\mathbf{J}}$ is unbiased,

$$
\mathbb E[\widehat{\mathbf{J}}]=\mathbf{J},
$$

we generally cannot conclude that

$$
\mathbb E[\mathcal{G}(\widehat{\mathbf{J}})]
=\mathcal{G}(\mathbf{J}).
$$

Unbiased inputs do not automatically produce an unbiased output after a nonlinear transformation.

DREAM studies this interaction between intra-task client drift and inter-task aggregation bias (Zhou et al., 2026).

#### Improve Inputs to the Weight Solver

:::warning
**Key idea.** Clean the inputs before weighting them: task-wise control variates remove the systematic drift of local updates, and momentum smooths their stochastic fluctuations, the noise behind the aggregation bias above, so the weight solver receives more accurate task updates (Zhou et al., 2026).
:::

Let the columns of $\mathbf{D}_t$ be the aggregated task updates, which play the role of $\widehat{\mathbf{J}}^t$. The server computes the weights from a regularized min-norm problem, in its basic form

$$
\boldsymbol{\lambda}_t
=\arg\min_{\boldsymbol{\lambda}\in\Delta_S}
\frac12\|\mathbf{D}_t\boldsymbol{\lambda}\|^2
+\frac{\rho}{2}\|\boldsymbol{\lambda}\|^2.
$$

The general framework replaces $\mathbf{D}_t$ by $\mathbf{D}_t\mathbf{A}^\top$, where the task correction matrix $\mathbf{A}$ can encode scalarization, prioritization, or interactions between tasks; the basic form is $\mathbf{A}=\mathbf{I}$. The server either solves the problem exactly (**Exact-Update**) or takes a single projected gradient step on the weights per round (**Step-Update**) (Zhou et al., 2026).

The regularizer makes the weight problem strongly convex, so the weights change smoothly with the task updates, which the analysis needs. The price is a bias that vanishes only as $\rho\to0$, which is why $\rho$ decreases with $T$ in the guarantees below (Zhou et al., 2026).

#### Guarantees

**Assumptions.** Local losses are smooth, global objective gradients are bounded, and stochastic gradients are unbiased with bounded variance. The analysis assumes full participation, with every client contributing to every task, and lower-bounded objectives. Step-Update additionally requires bounded client heterogeneity and almost-surely bounded stochastic gradient noise; Exact-Update does not require these additional conditions.

**Metric.** For $\mathbf{A}=\mathbf{I}$, the analysis bounds the Pareto-stationarity residual $\mathcal{R}(\mathbf{x})$ itself.

:::success
**Theorem (Zhou et al., 2026, Theorem 4.8 and Corollary 4.9).** With momentum parameter $\gamma$, local learning rate $\eta_L$, server learning rate $\eta_g$, and regularization strength $\rho$ chosen as
$$
\gamma=\Theta\!\left(\sqrt{\frac{N}{T}}\right),
\qquad
\eta_L=\Theta(T^{-1/2}),
\qquad
\eta_g=\Theta\!\left(\sqrt{\frac{N}{T}}\right),
\qquad
\rho=\Theta\!\left(\frac{1}{\sqrt{NT}}\right),
$$
DREAM achieves
$$
\min_{0\leq t<T}
\mathbb{E}\!\left[\mathcal{R}(\mathbf{x}^t)\right]
=O\!\left(\frac{1}{\sqrt{NT}}\right),
$$
for sufficiently large $T$ and constants satisfying the theorem's stability conditions. This holds for Exact-Update and, with a sufficiently small constant weight-update step size, Step-Update (Zhou et al., 2026).
:::

The rate improves with the number of clients: for fixed $T$, doubling $N$ shrinks the bound by a factor of $1/\sqrt{2}$. Zhou et al. (2026) describe this as the first such linear speedup for federated MOO that does not rely on bounded data heterogeneity, for Exact-Update. Momentum is essential to the result: with $\gamma=1$ the variance term in the bound does not vanish, even as $T\to\infty$.

#### Stationarity and Direction Accuracy

Beyond stationarity, DREAM bounds the conflict-avoidant (CA) distance between its update direction $\mathbf{d}^{t+1}$ and the ideal direction $\mathbf{J}(\mathbf{x}^t)\boldsymbol{\lambda}_0^*(\mathbf{x}^t)$, where $\boldsymbol{\lambda}_0^*$ solves the weight problem with $\rho=0$.

:::success
**Theorem (Zhou et al., 2026, Theorem 4.11 and Corollary 4.12).** For Exact-Update with the parameter schedule above, the CA distance vanishes at the same rate $O(1/\sqrt{NT})$ as stationarity. For Step-Update, choosing $\eta_g=\Theta(T^{-3/5})$, $\gamma=\Theta(T^{-2/5})$, and $\rho=\Theta(T^{-2/5})$ gives $O(T^{-2/5})$, reflecting a trade-off between tracking error and regularization bias.
:::

:::info
**Remark.** These results distinguish progress toward a stationary solution from accurate tracking of the intended direction. Both rely on full participation, with every client contributing to every task, so the client-count speedup should not be read as a guarantee for arbitrary participation patterns. The strongly convex analysis (Theorem 4.13) instead shows contraction toward a regularization-dependent neighborhood (Zhou et al., 2026).
:::

### 3.1.5 Comparing Federated Multi-Objective Methods

| Method | Main coordination problem | Central design choice | Upload per client per round | Non-convex rate |
|---|---|---|---|---|
| FedMGDA+ | Conflicting client objectives | Adaptive client aggregation and normalization | $\Theta(d)$ | Asymptotic only |
| FMOL | Tasks distributed across heterogeneous clients | Task-specific local training and two-stage aggregation | $\Theta(Sd)$ | $O(1/T)$ (FMGDA), $O(1/\sqrt{T})$ (FSMGDA) |
| FedCMOO | Communication that grows with objective count | Compressed gradient relationships and weighted local training | $\Theta(d)$ | $O(1/\sqrt{T})$ |
| DREAM | Drifted inputs to nonlinear aggregation | Control variates, momentum, and regularized weights | $\Theta(Sd)$ | $O(1/\sqrt{NT})$ |

Upload counts the model-sized vectors a client sends per round (for FMOL and DREAM, a client holding all $S$ objectives). Rates are for the Pareto-stationarity measures in the guarantees above, under each paper's assumptions.

:::info
**Takeaway.** The four methods place the multi-objective decision at different points of a federated round. FedMGDA+ treats each client as an objective at aggregation, FMOL separates tasks from clients and combines task updates after local training, FedCMOO fixes the weights before local training to reduce the per-client upload from $\Theta(Sd)$ to $\Theta(d)$, and DREAM corrects drift before the weights are computed. All four certify convergence toward Pareto stationarity, not the quality of the selected trade-off. Section 3.2 removes the server.
:::

---

## 3.2 Decentralized Multi-Objective Optimization

### From Neighbor Communication to Preference Coordination

Decentralized learning organizes agents over a communication graph. Each agent performs local computation and exchanges information with its neighbors, while maintaining its own estimate of a shared model. Without a central server, information must propagate through successive exchanges before it can influence the entire network. When agents optimize competing objectives, these exchanges must support both agreement on the model and coordination of the trade-off.

:::info
**Insight.** The difficulty comes from two interacting sources of disagreement. Agents can hold different local estimates because they use local information. They can also prefer different compromises among objectives, even when their local estimates coincide. Neighbor averaging helps reconcile decisions, but does not by itself specify which objectives should receive greater importance. A decentralized MOO method must therefore explain how local interactions produce a common decision with an interpretable trade-off.
:::

This section uses Blondin and Hale's preference-aware algorithm as a representative example and focuses on what neighbor-only communication changes (Blondin & Hale, 2021).

:::danger
**Key question.** How can agents coordinate their objective priorities while optimizing, rather than assuming that a common preference vector has already been agreed upon?
:::

### 3.2.1 Problem Formulation: Decisions and Preferences

Following Chapter 0, consider

$$
\min_{\mathbf{x}\in\mathcal{X}}\mathbf{F}(\mathbf{x}),
\qquad
\mathbf{F}(\mathbf{x}):=[f_1(\mathbf{x}),\ldots,f_S(\mathbf{x})]^\top,
$$

where $\mathbf{x}\in\mathbb{R}^d$ and $\mathcal{X}$ is a common feasible set. We call $\mathbf{x}$ the decision, use $N$ for the number of agents, and write $\mathbf{x}_i^t$ for agent $i$'s local estimate of the decision.

In the one-objective-per-agent setting, $S=N$ and agent $i$ knows only $f_i$. A preference vector $\boldsymbol{\lambda}\in\Delta_S$ with positive entries defines the scalarized objective

$$
f_{\boldsymbol{\lambda}}(\mathbf{x})
=\sum_{s=1}^S\lambda_s f_s(\mathbf{x}).
$$

Different preference vectors can select different solutions. Consequently, agents that agree on $\mathbf{x}$ may still disagree about the objective they intend to optimize. Conversely, agreeing on preferences does not ensure that their local estimates have converged.

### 3.2.2 Interlacing Preference Consensus and Optimization

Blondin and Hale study differentiable convex objectives, a common nonempty compact convex feasible set, and a connected undirected graph. Each agent knows its own objective and assigns a positive priority to every objective, collected in its preference vector $\boldsymbol{\lambda}_i\in\Delta_S$ (Blondin & Hale, 2021).

:::warning
**Key idea.** Interlace two consensus processes. In every iteration, agents average their priorities with their neighbors and use the current priorities to weight their neighbors' estimates, so agreement on the trade-off and on the model is reached together rather than one after the other.
:::

In our notation, each agent $i$ runs the following steps at iteration $t$.

:::info
**Algorithm 4: Preference consensus interlaced with optimization (agent $i$, iteration $t$)**

1. **Preference consensus.** Average preferences with the neighbors $\mathcal{N}_i$:
   $$\boldsymbol{\lambda}_i^{t+1}=\boldsymbol{\lambda}_i^t+c\sum_{j\in\mathcal{N}_i}\left(\boldsymbol{\lambda}_j^t-\boldsymbol{\lambda}_i^t\right).$$
2. **Decision mixing.** Combine the neighbors' local estimates, using information weights determined by the current preferences.
3. **Local step.** Take a local gradient step and project onto $\mathcal{X}$.
:::

With a small enough step size $c$, given in the assumptions below, preferences converge to

$$
\overline{\boldsymbol{\lambda}}
=\frac{1}{N}\sum_{i=1}^N\boldsymbol{\lambda}_i^0.
$$

The resulting information matrix is stochastic but generally not doubly stochastic: each agent's weights sum to one, but different agents' weights for the same objective need not. Many distributed optimization analyses assume doubly stochastic mixing, so the paper develops a new convergence analysis (Blondin & Hale, 2021).

#### Guarantees

**Assumptions.** Each agent's objective is continuously differentiable and convex. Agents share a nonempty compact convex feasible set and communicate over a connected undirected graph. Initial priorities are strictly positive, and the consensus step size $c$ is smaller than the inverse of the largest node degree. The optimization step sizes decrease and satisfy
$$
\eta_t\to0,
\qquad
\sum_{t=0}^{\infty}\eta_t=\infty,
\qquad
\sum_{t=0}^{\infty}\eta_t^2<\infty.
$$

**Metric.** The analysis measures decision disagreement
$$
E_i^t:=\|\mathbf{x}_i^t-\overline{\mathbf{x}}^t\|,
\qquad
\overline{\mathbf{x}}^t
:=\frac{1}{N}\sum_{i=1}^N\mathbf{x}_i^t,
$$
and the distance of each local estimate to a common limit $\mathbf{x}^{\star}$.

:::success
**Theorem (Blondin & Hale, 2021, Theorems 4.1 and 4.2).** The consensus-error bound has the form
$$
E_i^t
\leq
C_1\beta^{t-1}
+C_2\sum_{r=0}^{t-2}\beta^{t-r}\eta_r
+C_3\eta_{t-1},
\qquad 0<\beta<1,
$$
where the constants and mixing factor depend on the problem, network, and initial preferences. The first term captures geometric decay of initial disagreement; the remaining terms capture perturbations from optimization steps.

Consequently, $E_i^t\to0$, and all agents converge to a common solution $\mathbf{x}^{\star}$, which the paper describes as optimal with respect to the agreed preferences:
$$
\lim_{t\to\infty}\|\mathbf{x}_i^t-\mathbf{x}^{\star}\|=0,
\qquad \forall i.
$$
The paper also provides an optimization-distance bound involving accumulated consensus errors and step sizes, rather than a single universal $O(1/T)$ rate (Blondin & Hale, 2021).
:::

The mixing factor $\beta$ depends on the smallest initial priority and the number of agents. Convergence slows as the smallest priority decreases or the number of agents grows, so the agents' priorities affect not only the final trade-off but also how fast agreement is reached (Blondin & Hale, 2021).

### 3.2.3 Discussion: What Does Decentralized Coordination Achieve?

#### The Selected Trade-Off Matters

:::danger
**Limitations.** For strictly positive weights, a global minimizer of the weighted objective is Pareto optimal: a dominating point would necessarily have a smaller weighted objective. However, this property does not identify which Pareto solution is desirable. Preference selection remains consequential, and agreement on an average preference vector need not satisfy every agent's original priorities.

Preference averaging also serves a different purpose from MGDA's gradient-based weight selection. It negotiates supplied priorities, whereas MGDA chooses weights from the current gradient geometry. A run targeting one compromise should therefore not be interpreted as recovering the entire Pareto front.
:::

#### Extending Beyond One Objective per Agent

In decentralized multi-task learning, objectives and agents need not coincide. A global task loss may instead take the form

$$
f_s(\mathbf{x})=\frac{1}{N}\sum_{i=1}^N f_{s,i}(\mathbf{x}).
$$

Now the network must also estimate each task's global gradient from local contributions. Parameter consensus alone does not remove differences between those contributions. This introduces an additional coordination problem beyond negotiating preferences, so the one-objective-per-agent result should not be transferred directly to this setting.

:::info
**Takeaway.** Without a server, agents must reconcile both their model estimates and their preferences through neighbor exchanges. Blondin and Hale interlace preference averaging with projected gradient steps, so all agents converge to a common solution, with a trade-off influenced by their initial preferences. Extending this guarantee beyond one objective per agent also requires estimating each task's global gradient from local contributions.
:::

---

## 3.3 Summary and Open Directions

Returning to the central question, the answer is a partial yes. In the federated setting, FMOL, FedCMOO, and DREAM converge, under their respective assumptions, to Pareto-stationary points despite several local steps per round. FedCMOO also cuts the upload to $\Theta(d)$ per client per round, although its guarantee covers a simplified variant, and DREAM corrects drift. In none of the four federated methods, however, do the participants negotiate the trade-off: the server derives the weights from the current gradients, possibly modified by DREAM's task correction matrix. In the decentralized setting, Blondin and Hale's method reaches agreement on both the model and the trade-off, but only for convex objectives with one objective per agent.

Federated and decentralized MOO differ in who coordinates the trade-off and in what their guarantees certify.

| Setting | Representative methods | Who sets the weights | What is kept | What is given up |
|---|---|---|---|---|
| Federated, adaptive aggregation (3.1.1, 3.1.2, 3.1.4) | FedMGDA+, FMOL, DREAM | The server, from client updates in each round | Common-descent weighting; convergence to Pareto stationarity | Exact gradients, since updates are sampled, drifted, or noisy; FMOL and DREAM upload $\Theta(Sd)$ per client |
| Federated, communication-efficient (3.1.3) | FedCMOO | The server, from a compressed Gram-matrix estimate, before local training | $\Theta(d)$ upload per client; $O(1/\sqrt{T})$ stationarity rate | A guarantee for the practical compressor; a theoretical benefit of adaptive over fixed weights |
| Decentralized (3.2) | Blondin & Hale | Agents, by averaging their initial priorities | Agreement on the model and the priorities without a server | Non-convex objectives are not covered; the weights follow the average priority rather than the current gradient conflicts |

**Which method to use.** When each client is a stakeholder and the concern is that averaging sacrifices some of them, FedMGDA+ applies MGDA directly to client updates, and $\epsilon$ interpolates between FedAvg and MGDA. When clients hold different subsets of tasks, the FMOL formulation is needed, with FMGDA and FSMGDA as baseline algorithms. When the number of objectives makes uploads the bottleneck, FedCMOO reduces the per-client upload from $\Theta(Sd)$ to $\Theta(d)$. When local drift and stochastic noise dominate and all clients participate, DREAM corrects the drift and also bounds the error of its update direction. Without a server, Blondin and Hale's method applies to convex objectives with one objective per agent, provided the agents accept the average of their initial priorities.

**How to evaluate.** Methods with the same convergence rate can differ by a factor of $S$ in communication, so round counts alone are misleading. Comparisons should report cumulative uploaded data, counting both messages and payload sizes, together with local computation, individual objective values rather than only their average, and sensitivity to the number of local steps. Decentralized methods should also report agreement between local models and the preferences finally selected. Theoretical comparisons should state the stationarity metric, the participation assumptions, and the parameter schedules. Pareto stationarity is a first-order condition; it does not establish global Pareto optimality or coverage of the Pareto front.

**Open directions.**
- **Partial participation.** DREAM's guarantees, including its speedup in $N$, assume that every client contributes to every task in every round. Extending drift correction to sampled clients and to uneven task assignments, the setting that FMOL formulates, is a natural next step.
- **Combining compression and drift correction.** FedCMOO reduces communication and DREAM corrects drift, but each analysis covers only its own mechanism. Whether compressed Gram-matrix estimates can feed a drift-corrected, regularized weight solver while keeping both guarantees is not covered by either paper.
- **Decentralized MOO beyond one convex objective per agent.** Blondin and Hale's analysis covers convex objectives with one objective per agent. Decentralized multi-task settings additionally require each task's global gradient to be estimated through neighbors (Section 3.2.3); combining this estimation with preference consensus, and extending the analysis to non-convex objectives, remain to be worked out.

---

## References

<!-- Our own MOO papers -->

- Yang, H., Liu, Z., Liu, J., Dong, C., & Momma, M. (2023). Federated multi-objective learning. *NeurIPS 2023*. [arXiv:2310.09866](https://arxiv.org/abs/2310.09866)

<!-- Other references -->

- Askin, B., Sharma, P., Joshi, G., & Joe-Wong, C. (2025). Federated communication-efficient multi-objective optimization. *AISTATS 2025*. [PMLR](https://proceedings.mlr.press/v258/askin25a.html)
- Blondin, M. J., & Hale, M. (2021). A decentralized multi-objective optimization algorithm. *Journal of Optimization Theory and Applications*, 189(2), 458–485. [doi:10.1007/s10957-021-01840-z](https://doi.org/10.1007/s10957-021-01840-z)
- Hu, Z., Shaloudegi, K., Zhang, G., & Yu, Y. (2022). Federated learning meets multi-objective optimization. *IEEE Transactions on Network Science and Engineering*, 9(4), 2039–2051. [arXiv:2006.11489](https://arxiv.org/abs/2006.11489)
- Zhou, Y., Ou, Y., & Shi, X. (2026). DREAM: A unified framework for drift-corrected federated multi-objective learning. *ICML 2026*. [PMLR](https://proceedings.mlr.press/v306/zhou26ak.html)
