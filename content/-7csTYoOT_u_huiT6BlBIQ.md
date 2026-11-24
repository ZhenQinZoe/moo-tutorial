---
title: "MGDA: Convergence Theory and Algorithmic Extensions"
author: Liuyuan Jiang
create date: 2026-10-26
update date: 2026-10-26
description: A tutorial and survey of deterministic MGDA convergence, weaker smoothness, stochastic gradient bias, efficient variants, and preference-aware multiobjective descent.
tags:
  - Multi-objective-optimization
  - Pareto-stationarity
  - MGDA
  - Convergence-analysis
  - Multi-task-learning
type: blog
status: draft
---

# MGDA: Convergence Theory and Algorithmic Extensions

The companion note *Introduction to Multi-objective Optimization: Pareto Optimality and Stationarity* develops Pareto optimality and stationarity, and [*Introduction to MGDA*](Intro%20Deterministic%20MGDA.md) derives the multiple-gradient descent direction.
This tutorial establishes the **convergence rate of deterministic MGDA** and surveys extensions based on curvature, nonsmoothness, stochastic gradients, computational efficiency, and preferences.
The basic proof provides a common framework for understanding the progress measures and guarantees of these methods.

::: {.question-highlight style="background:#f6f8fa; border-left:4px solid #64748b; padding:0.7em 1em;"}

*Q: How quickly does MGDA approach Pareto stationarity, and how do its main algorithmic extensions change the convergence analysis?*

:::

Let $f_m:\mathbb{R}^n\to\mathbb{R}$, $m\in[M]=\{1,\ldots,M\}$, be differentiable objectives to be minimized.
As in the companion notes, write $F(x)=(f_1(x),\ldots,f_M(x))^\top$ and define the simplex $\Delta_M=\{\lambda\in\mathbb{R}^M:\lambda_m\geq0\text{ for all }m\in[M],\ \sum_{m\in[M]}\lambda_m=1\}$.
The **MGDA direction** $d(x)$ and the **squared Pareto-stationarity measure** $R(x)$ are
$$
\begin{aligned}
\lambda^\star(x)&\in\operatorname*{argmin}_{\lambda\in\Delta_M}
\frac12\left\|\sum_{m\in[M]}\lambda_m\nabla f_m(x)\right\|^2,\\
d(x)&=-\sum_{m\in[M]}\lambda_m^\star(x)\nabla f_m(x),\\
R(x)&:=\min_{\lambda\in\Delta_M}\left\|\sum_{m\in[M]}\lambda_m\nabla f_m(x)\right\|^2
=\|d(x)\|^2.
\end{aligned}
\tag{1}
$$
^eq-mgda-residual

The direction is unique, although its weight representation may be nonunique.
The **common-descent inequality** from the introductory MGDA note is
$$
\nabla f_m(x)^\top d(x)\leq-\|d(x)\|^2=-R(x),
\qquad m\in[M],
\tag{2}
$$
^eq-common-descent

Moreover, $R(x)=0$ exactly when $x$ is **Pareto stationary** [Fliege and Svaiter (2000)](https://doi.org/10.1007/s001860000043) [\[1\]](#^ref-fliege-svaiter-2000); [Désidéri (2012)](https://doi.org/10.1016/j.crma.2012.03.014) [\[2\]](#^ref-desideri-2012).
The convergence proof combines this inequality with a lower bound on an objective.

## Convergence Theory and Fundamental Analysis

### Convergence Rates and Iteration Complexity

Assume each objective is **$L$-smooth** with a common $L>0$:
$\|\nabla f_m(x)-\nabla f_m(y)\|\leq L\|x-y\|$ for all $x,y$ and $m$.
Suppose **at least one objective**, say $f_j$, is bounded below by $f_j^{\inf}>-\infty$.
The analysis applies to **nonconvex objectives** and uses only these smoothness and lower-boundedness assumptions.

::: {.proposition #prop-basic-rate}

**Proposition 1 (Nonconvex rate of deterministic MGDA).**

Let $d_t=d(x_t)$ and $x_{t+1}=x_t+L^{-1}d_t$, and suppose the assumptions above hold.
Then for every integer $T\geq1$,
$$
\boxed{
\min_{0\leq t<T}R(x_t)
\leq\frac{2L\bigl(f_j(x_0)-f_j^{\inf}\bigr)}{T}.
}
\tag{3}
$$
^eq-basic-rate

Consequently, an iterate with $\|d(x_t)\|\leq\varepsilon$ is obtained in at most $O(\varepsilon^{-2})$ iterations, while $R(x_t)\to0$ along the entire sequence.
Every accumulation point, if one exists, is Pareto stationary.

:::

::: {.proof #proof-basic-rate}

**Proof of Proposition 1.**
By $L$-smoothness and the common-descent inequality [[#^eq-common-descent|(2)]],
$$
\begin{aligned}
f_m(x_{t+1})
&\leq f_m(x_t)+\frac1L\nabla f_m(x_t)^\top d(x_t)
+\frac{1}{2L}\|d(x_t)\|^2\\
&\leq f_m(x_t)-\frac{1}{2L}R(x_t).
\end{aligned}
\tag{4}
$$
^eq-one-step-decrease

Apply this bound only to the lower-bounded objective $f_j$ and sum over $t=0,\ldots,T-1$:
$$
\frac{1}{2L}\sum_{t=0}^{T-1}R(x_t)
\leq f_j(x_0)-f_j(x_T)
\leq f_j(x_0)-f_j^{\inf}.
\tag{5}
$$
^eq-telescoping

The minimum residual is at most its average, giving [[#^eq-basic-rate|(3)]].
Letting $T\to\infty$ shows $\sum_{t=0}^\infty R(x_t)<\infty$, hence $R(x_t)\to0$.
The gradients are continuous, so the minimum-norm distance from their convex hull to the origin is continuous in $x$; any convergent subsequence therefore has a Pareto-stationary limit. <span class="qed" style="float:right;">&#9633;</span>

:::

The proof follows a useful pattern: **common descent**, **sufficient decrease**, and a **lower-bounded objective** imply summability of $R(x_t)$.
The complexity is $O(\varepsilon^{-2})$ for $\|d(x_t)\|\leq\varepsilon$, or $O(\varepsilon^{-1})$ for the squared criterion $R(x_t)\leq\varepsilon$.
Each iteration computes $M$ gradients and solves a simplex-constrained quadratic subproblem; the bound counts **full-gradient iterations**.
This matches the nonconvex complexity framework of [Fliege, Vaz, and Vicente (2019)](https://doi.org/10.1080/10556788.2018.1510928) [\[3\]](#^ref-fliege-vaz-vicente-2019).

**Stationarity and iterate convergence.**
The theorem establishes convergence of the stationarity measure and Pareto stationarity of accumulation points.
Bounded iterates ensure such accumulation points exist, while convergence of the full sequence and Pareto optimality require additional structure.
The residual measures **first-order stationarity**; its relation to solution distance depends on further error-bound conditions.

#### Armijo Backtracking and Related Descent Directions

**Armijo backtracking** replaces the fixed step $1/L$ with a step chosen from objective values.
Starting from step size $1$, choose $\alpha_t\in\{1,\rho,\rho^2,\ldots\}$, with $\rho\in(0,1)$, until **every** objective satisfies
$$
f_m(x_t+\alpha_t d_t)
\leq f_m(x_t)+c\alpha_t\nabla f_m(x_t)^\top d_t,
\qquad c\in(0,1).
\tag{6}
$$
^eq-armijo

Smoothness guarantees acceptance whenever $\alpha_t\leq2(1-c)/L$.
Thus $\alpha_t\geq\underline\alpha:=\min\{1,2\rho(1-c)/L\}>0$, and
$$
\min_{0\leq t<T}R(x_t)
\leq\frac{f_j(x_0)-f_j^{\inf}}{c\underline\alpha\,T}.
\tag{7}
$$
^eq-armijo-rate

The rate is again $O(\varepsilon^{-2})$ for the direction norm.
The same proof extends to approximate directions through two quantitative properties: **componentwise descent** and **comparability with the exact MGDA residual**.
[Lapucci (2024)](https://doi.org/10.1016/j.orl.2024.107115) [\[4\]](#^ref-lapucci-2024) formalizes this observation using *steepest-descent-related directions* and obtains convergence and complexity guarantees for a broad family of nonconvex methods.
For example, suppose an approximate direction $\widetilde d_t$ satisfies $\nabla f_m(x_t)^\top\widetilde d_t\leq-a\|\widetilde d_t\|^2$ for every $m$, together with $\|\widetilde d_t\|\geq b\|d(x_t)\|$, for constants $a,b>0$ independent of $t$.
The update $x_{t+1}=x_t+(a/L)\widetilde d_t$ then satisfies $f_m(x_{t+1})\leq f_m(x_t)-a^2b^2R(x_t)/(2L)$ by smoothness, giving $O(\varepsilon^{-2})$ stationarity complexity.
The two inequalities give a practical **relative-quality criterion** for an approximate direction solver.

For convex objectives, appropriate **multiobjective merit functions** can have faster convergence rates; strong convexity supports linear rates with further regularity [Fliege, Vaz, and Vicente (2019)](https://doi.org/10.1080/10556788.2018.1510928) [\[3\]](#^ref-fliege-vaz-vicente-2019).
These rates describe the particular **merit function or scalarization** analyzed in each theorem.

### MGDA Under Weaker Smoothness Assumptions

The minimum-norm geometry [[#^eq-mgda-residual|(1)]] and common descent [[#^eq-common-descent|(2)]] hold for **differentiable objectives**.
Convergence rates follow by combining them with a quantitative bound on local objective changes.
Two important settings are Hölder-continuous gradients and generalized smoothness.

**Hölder-continuous gradients.**
Suppose, for $0<\nu\leq1$ and $H>0$,
$\|\nabla f_m(x)-\nabla f_m(y)\|\leq H\|x-y\|^\nu$ for every $m$.
The descent lemma becomes
$$
f_m(x+\alpha d)
\leq f_m(x)+\alpha\nabla f_m(x)^\top d
+\frac{H}{1+\nu}\alpha^{1+\nu}\|d\|^{1+\nu}.
\tag{8}
$$
^eq-holder-descent

For $d=d(x)\neq0$, choose
$$
\alpha(x)=\left(\frac{1+\nu}{2H}\right)^{1/\nu}
\|d(x)\|^{(1-\nu)/\nu}.
\tag{9}
$$
^eq-holder-step

Substituting [[#^eq-common-descent|(2)]] and [[#^eq-holder-step|(9)]] into [[#^eq-holder-descent|(8)]] gives
$$
f_m(x+\alpha(x)d(x))
\leq f_m(x)-\frac12\left(\frac{1+\nu}{2H}\right)^{1/\nu}
\|d(x)\|^{(1+\nu)/\nu}.
\tag{10}
$$
^eq-holder-progress

Telescoping a lower-bounded objective therefore yields
$$
\min_{0\leq t<T}\|d(x_t)\|
=O\!\left(T^{-\nu/(1+\nu)}\right),
\qquad
T=O\!\left(\varepsilon^{-(1+\nu)/\nu}\right).
\tag{11}
$$
^eq-holder-rate

This **Hölder-smoothness rate** follows directly from the common-descent proof.
For $\nu=1$, it recovers $O(\varepsilon^{-2})$; smaller $\nu$ leads to a slower rate.
The step length in [[#^eq-holder-step|(9)]] adapts to the local norm of the MGDA direction.

**Generalized $\ell$-smoothness.**
Neural-network objectives may have curvature that grows with the gradient magnitude, making global $L$-smoothness unrealistic.
A representative twice-differentiable condition bounds $\|\nabla^2 f_m(x)\|$ by $\ell(\|\nabla f_m(x)\|)$ for a nondecreasing function $\ell$, rather than by one global constant.
[Zhang, Xiao, Zou, and Ji (2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html) [\[5\]](#^ref-zhang-2025) analyze MGDA under their specified generalized-smoothness conditions.
Their analysis gives $O(\varepsilon^{-2})$ deterministic and $O(\varepsilon^{-4})$ stochastic sample bounds under specified **Pareto-stationarity and conflict-avoidance criteria**.
The analysis combines generalized smoothness with tailored step rules and stochastic-oracle conditions.

## Deterministic Algorithmic Improvements

### Newton, Quasi-Newton, and Curvature-Aware Directions

Standard MGDA uses a common Euclidean quadratic regularizer $\|v\|^2/2$.
When objectives are ill-conditioned, the resulting direction may be too conservative even though it decreases each objective.
Curvature-aware variants modify the local **direction subproblem**, not merely the outer learning rate.

**Multiobjective Newton.**
A natural local model replaces the linearized changes by second-order approximations:
$$
d_N(x)\in\operatorname*{argmin}_{v\in\mathbb R^n}
\max_{m\in[M]}\left\{
\nabla f_m(x)^\top v
+\frac12v^\top\nabla^2f_m(x)v
\right\}.
\tag{12}
$$
^eq-newton-model

Under **local strong convexity** and the paper's other regularity assumptions, [Fliege, Graña Drummond, and Svaiter (2009)](https://doi.org/10.1137/08071692X) [\[6\]](#^ref-fliege-newton-2009) establish local superlinear convergence, and quadratic convergence with Lipschitz second derivatives.
The stronger local rates arise from the **curvature and regularity** of the second-order model near the solution.

**Limited-memory quasi-Newton.**
[Lapucci and Mansueto (2023)](https://doi.org/10.1007/s10589-023-00454-7) [\[7\]](#^ref-lapucci-mansueto-lbfgs-2023) use a positive-definite limited-memory approximation to a combination of objective Hessians.
Their Wolfe-type line search is well defined in the nonconvex setting, and stronger convexity conditions yield global and $R$-linear convergence results.
This avoids forming $M$ full Hessians and is particularly relevant when the decision dimension is large.

**Barzilai–Borwein scaling.**
[Chen, Tang, and Yang (2023)](https://doi.org/10.1016/j.ejor.2023.04.022) [\[8\]](#^ref-chen-tang-yang-2023) address imbalanced objective scales using Barzilai–Borwein-inspired gradient scaling.
Their descent framework establishes Pareto-critical accumulation points under the stated conditions and can allow less conservative line-search steps.
Gradient scaling changes the geometry of MGDA and allows objective units to influence the resulting **common-descent direction**.

### Proximal and Accelerated Multiobjective Methods

When objectives have nonsmooth regularizers, write $f_m(x)=h_m(x)+r_m(x)$, where $h_m$ is smooth and $r_m$ is proper, convex, and possibly nonsmooth.
A multiobjective proximal-gradient step can minimize the worst composite local change:
$$
v_t\in\operatorname*{argmin}_{v\in\mathbb R^n}
\left\{\max_{m\in[M]}\Big[
\nabla h_m(x_t)^\top v+r_m(x_t+v)-r_m(x_t)
\Big]+\frac{1}{2\eta_t}\|v\|^2\right\}.
\tag{13}
$$
^eq-proximal-model

This formulation reduces to a scaled MGDA direction when each $r_m=0$.
[Tanabe, Fukuda, and Yamashita (2019)](https://doi.org/10.1007/s10589-018-0043-x) [\[9\]](#^ref-tanabe-prox-2019) study proximal methods with and without line search and establish Pareto-stationary accumulation points under their assumptions.

In **convex composite** problems, [Tanabe, Fukuda, and Yamashita (2023)](https://doi.org/10.1007/s10589-023-00497-w) [\[10\]](#^ref-tanabe-accelerated-2023) introduce an accelerated proximal variant with an $O(1/t^2)$ **merit-function** rate.
[Sonntag and Peitz (2024)](https://doi.org/10.1007/s10957-024-02389-3) [\[11\]](#^ref-sonntag-peitz-2024) derive inertial multiobjective gradient algorithms and analyze an accelerated $O(1/t^2)$ merit-value rate in the convex smooth setting.
These accelerated rates concern **convex optimality merit functions**; the squared residual $R(x_t)$ is the stationarity measure used for nonconvex MGDA above.

### Constrained and Projected Multiobjective Descent

For a feasible set $X$, Pareto stationarity depends on the **feasible directions** as well as on the objective gradients.
The constrained direction must therefore incorporate the geometry of $X$.
For a **closed convex** set $X$, one useful feasible-displacement model is
$$
v(x)\in\operatorname*{argmin}_{v:\,x+v\in X}
\left\{\max_{m\in[M]}\nabla f_m(x)^\top v
+\frac12\|v\|^2\right\}.
\tag{14}
$$
^eq-constrained-model

Because $X$ is convex, a step $x+\alpha v(x)$ stays feasible for $0\leq\alpha\leq1$.
The direction vanishes when there is no feasible direction that strictly decreases every objective to first order; this is **constrained Pareto stationarity**, which can also be expressed using the feasible tangent/normal cone.
For manifolds or nonlinear nonconvex constraints, feasible directions are computed from **tangent-space or sequential quadratic models** with constraint correction.

[Fliege and Vaz (2016)](https://doi.org/10.1137/15M1016424) [\[12\]](#^ref-fliege-vaz-2016) develop an SQP-based constrained, front-oriented algorithm with local Pareto convergence analysis.
[Désidéri (2018)](https://mgda.inria.fr/constrained) [\[13\]](#^ref-desideri-constrained-2018) instead combines tangent-space MGDA directions with a feasibility-correction step.
Both illustrate that **feasibility restoration** is a separate requirement from finding a common-descent direction.

## Stochastic MGDA and Gradient Bias

### Stochastic Multi-Gradient Methods

In learning problems, $f_m(x)=\mathbb E_{\xi}[\ell_m(x;\xi)]$, and only stochastic estimates $\widehat g_m(x,\xi)$ may be available.
Even if $\mathbb E[\widehat g_m(x,\xi)\mid x]=\nabla f_m(x)$ for every task, choosing the weights **from the same noisy gradients** typically biases the resulting update:
$$
\widehat d(x,\xi)
=-\sum_{m\in[M]}\widehat\lambda_m(x,\xi)\widehat g_m(x,\xi),
\qquad
\mathbb E[\widehat d(x,\xi)\mid x]\neq d(x)
\quad\text{in general}.
\tag{15}
$$
^eq-stochastic-bias

The bias arises from the nonlinear **minimum-norm projection** and the statistical dependence between $\widehat\lambda$ and $\widehat g_m$.
In single-objective SGD, by contrast, the update is linear in its unbiased gradient estimate.

#### A Bias Example with Unbiased Task Gradients

Let $f_1(x)=f_2(x)=x^2/2$ in one dimension.
At $x=1$, both true gradients equal $1$, so [[#^eq-mgda-residual|(1)]] gives $d(1)=-1$.
Let $\xi$ equal $+1$ or $-1$ with probability $1/2$, and use the unbiased estimates
$$
\widehat g_1(1,\xi)=1+2\xi,
\qquad
\widehat g_2(1,\xi)=1-2\xi.
\tag{16}
$$
^eq-bias-example

Each realization gives the pair $(3,-1)$ or $(-1,3)$.
The origin lies in the convex hull of either pair, so **stochastic MGDA selects $\widehat d=0$ with probability one**, although the true point is not Pareto stationary.
The example shows how instantaneous noisy weights can produce a **spurious stationary update**.

Early stochastic MGDA variants include [Mercier, Poirion, and Désidéri (2018)](https://doi.org/10.1016/j.ejor.2018.05.064) [\[14\]](#^ref-mercier-2018), which prove mean-square and almost-sure convergence under their stochastic-algorithm assumptions.
[Liu and Vicente (2021/2024)](https://doi.org/10.1007/s10479-021-04033-z) [\[15\]](#^ref-liu-vicente-2024) make the bias of the stochastic multi-gradient direction explicit and develop rates in convex and strongly convex settings.
Their convergence results use distinct **sampling assumptions and weight-update schemes**.

### Bias Correction and Stochastic Convergence

The example suggests a design principle: **separate or stabilize the estimation of weights from the estimation of gradients used to update the model**.
Several papers implement this principle differently.

**Weight averaging (CR-MOGM).**
[Zhou et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract.html) [\[16\]](#^ref-zhou-2022) demonstrate nonconvergence of instantaneous stochastic gradient manipulation on a convex example.
Their **composite-weight averaging** stabilizes the adaptive weights and yields convergence under the paper's assumptions.

**Gradient tracking/correction (MoCo).**
[Fernando et al. (2023)](https://openreview.net/forum?id=dLAYGdKTi2) [\[17\]](#^ref-fernando-2023) introduce **Multi-objective gradient Correction (MoCo)** to control the discrepancy between noisy gradients and the desired multiobjective descent behavior.
Their analysis allows **fixed mini-batch sizes** in nonconvex optimization, with carefully chosen update schedules.

**Double sampling (MoDo).**
To see the key statistical idea, let $G(x)=[\nabla f_1(x),\ldots,\nabla f_M(x)]$ and let $\widehat G^{(1)},\widehat G^{(2)}$ be independent unbiased estimates conditional on $x$.
Then
$$
\mathbb E\!\left[(\widehat G^{(1)})^\top
\widehat G^{(2)}\mid x\right]
=G(x)^\top G(x),
\tag{17}
$$
^eq-double-sampling

whereas the same-sample Gram matrix $\mathbb E[\widehat G^\top\widehat G\mid x]$ generally includes a **noise-covariance term**.
[Chen, Fernando, Ying, and Chen (2024)](https://www.jmlr.org/papers/v25/23-1287.html) [\[18\]](#^ref-chen-fernando-ying-2024) study **MoDo**, which uses double-sampling ideas in dynamic multiobjective weighting and establishes an important **optimization–generalization–conflict-avoidance trade-off**.
The resulting theory connects the **weight-estimation procedure** to population generalization and task conflict.

Under generalized smoothness, [Zhang et al. (2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html) [\[5\]](#^ref-zhang-2025) further analyze stochastic MGDA with double sampling, reporting an $O(\varepsilon^{-4})$ **sample** bound for their stated accuracy criteria.
The stochastic result counts **sample-gradient evaluations**, whereas Proposition 1 counts **full-gradient iterations**.

## Scalability and Computational Efficiency

### Efficient MGDA for Large Neural Networks

Exact MGDA must obtain all task gradients and solve a small quadratic problem in the task weights.
For a network with $n$ parameters and $M$ tasks, **materializing every full gradient** can require $O(Mn)$ storage before any improvement to the weight solver is considered.
The $M\times M$ Gram matrix $G^\top G$ is small only when $M$ is small; constructing it can still require work proportional to $Mn$.
These are separate **parameter-dimensional** and **task-dimensional** bottlenecks.

**Representation-space MGDA (MGDA-UB).**
[Sener and Koltun (2018)](https://mlanthology.org/neurips/2018/sener2018neurips-multitask/) [\[19\]](#^ref-sener-koltun-2018) optimize an upper-bound construction using gradients at a shared representation, reducing the cost of forming task-gradient combinations in multi-task networks.
The **representation-space upper bound** determines the weights and associated guarantees, while full-parameter MGDA uses the parameter gradients directly.

**Fast approximation under generalized smoothness (MGDA-FA).**
[Zhang et al. (2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html) [\[5\]](#^ref-zhang-2025) also analyze **MGDA-FA**, reporting constant extra time/space in their algorithmic cost model while retaining their MGDA accuracy guarantees.
The cost statement refers to the **additional overhead** measured in their algorithmic model.

### Low-Cost Adaptive Task Weighting

[Liu, Feng, Stone, and Liu (2023)](https://papers.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html) [\[20\]](#^ref-famo-2023) propose **FAMO**, which balances task-loss progress using $O(1)$ additional time and space in its stated scaling model instead of storing each full task gradient.
FAMO learns task weights from **loss changes**, while MGDA chooses weights by minimizing the norm of an aggregate gradient.
Their computational costs and optimization objectives are correspondingly different.

## Preference-Guided Solutions and Pareto-Front Exploration

### Preference-Aware Multiobjective Descent

MGDA computes **adaptive conflict-avoidance weights** from the current gradients, whereas preference-guided methods seek solutions reflecting an **explicit target trade-off**.
Initialization and trajectory influence which stationary point MGDA approaches.

**Pareto MTL.**
[Lin et al. (2019)](https://papers.neurips.cc/paper_files/paper/2019/hash/685bfde03eb646c27ed565881917c71c-Abstract.html) [\[21\]](#^ref-lin-2019) decompose learning into **preference-region constrained subproblems**, seeking multiple solutions in different regions of the front.

**EPO Search.**
[Mahapatra and Rajan (2020)](https://proceedings.mlr.press/v119/mahapatra20a.html) [\[22\]](#^ref-mahapatra-rajan-2020) develop **Exact Pareto Optimal (EPO) Search**, combining descent with controlled ascent to target user preferences.
Its **controlled-ascent** steps allow temporary increases in selected objectives to steer the search toward a preferred trade-off, in contrast to MGDA's componentwise decrease [[#^eq-one-step-decrease|(4)]].

The three algorithmic goals are **Pareto stationarity**, **preference matching**, and **Pareto-front coverage**.
Each requires a corresponding progress measure.

### Gradient-Based Pareto-Front Exploration

A single MGDA trajectory produces **one** candidate point.
To explore the front, one can instead maintain a **set of candidates**, improve their stationarity, and add or redistribute candidates to reach less-covered regions.
[Lapucci and Mansueto (2023)](https://doi.org/10.1016/j.orl.2023.03.001) [\[23\]](#^ref-lapucci-mansueto-front-2023) improve **front steepest descent** to address the limited front coverage of earlier versions while retaining their asymptotic convergence properties.

This separates a **local optimization certificate** ($R(x)\approx0$ for each candidate) from a **global allocation criterion** (how solutions are spread over the front).
Front-coverage guarantees additionally quantify the **distribution of candidate solutions**.

## Related Gradient-Balancing Methods in Multi-Task Learning

### Projection and Conflict-Averse Updates

**PCGrad.**
[Yu et al. (2020)](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) [\[24\]](#^ref-yu-pcgrad-2020) modify task gradients by projecting away pairwise conflicting components.
Its **pairwise, order-dependent projections** differ from MGDA's unique minimum-norm gradient aggregate; componentwise descent is a property of MGDA's exact direction.

**CAGrad.**
[Liu et al. (2021)](https://papers.nips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) [\[25\]](#^ref-liu-cagrad-2021) optimize **average task loss** while regularizing the worst local task improvement.
The method interpolates between ordinary gradient descent and conflict-averse behavior, and its convergence analysis targets the **average task loss**.

The distinction among MGDA, PCGrad, and CAGrad lies in their **aggregation rule**, **progress measure**, and **convergence target**.

### Fair and Bargaining-Based Gradient Aggregation

[Navon et al. (2022)](https://proceedings.mlr.press/v162/navon22a.html) [\[26\]](#^ref-navon-2022) formulate gradient combination as a **Nash bargaining problem**.
The Nash-MTL direction favors a proportionally fair negotiation among tasks under the model's assumptions, instead of minimizing the norm of a convex gradient combination.
The **bargaining-based criterion** defines its direction and convergence target, distinct from the minimum-norm criterion in [[#^eq-mgda-residual|(1)]].

## Comparison of Convergence Guarantees

| Setting or method | Convergence result or objective | Assumptions and progress measure |
|---|---|---|
| **Smooth deterministic MGDA** | $\min_{t<T}\|d(x_t)\|=O(T^{-1/2})$ | $L$-smoothness; one objective bounded below; exact direction |
| **Hölder-gradient MGDA** | $\min_{t<T}\|d(x_t)\|=O(T^{-\nu/(1+\nu)})$ | Global $\nu$-Hölder gradient bound; adapted step [[#^eq-holder-step|(9)]] |
| **Generalized-smooth MGDA** [\[5\]](#^ref-zhang-2025) | $O(\varepsilon^{-2})$ deterministic / $O(\varepsilon^{-4})$ stochastic samples | The paper's accuracy criteria and oracle assumptions |
| **Multiobjective Newton** [\[6\]](#^ref-fliege-newton-2009) | Local superlinear or quadratic convergence | Local curvature and second-order regularity |
| **Accelerated convex methods** [\[10\]](#^ref-tanabe-accelerated-2023) [\[11\]](#^ref-sonntag-peitz-2024) | $O(T^{-2})$ merit-function rate | Convexity; not a generic nonconvex residual bound |
| **MoCo / MoDo / weight correction** [\[16\]](#^ref-zhou-2022)–[\[18\]](#^ref-chen-fernando-ying-2024) | Control stochastic bias and/or study its consequences | Distinct stochastic assumptions, algorithms, and criteria |
| **Pareto MTL / EPO / front descent** [\[21\]](#^ref-lin-2019)–[\[23\]](#^ref-lapucci-mansueto-front-2023) | Preference targeting or front exploration | Requires targets beyond stationarity of one iterate |

The common thread is the **progress measure**.
For deterministic MGDA, componentwise decrease makes the stationarity residual $R(x)$ summable.
Curvature-aware, stochastic, constrained, and preference-guided methods modify the direction model or convergence target, leading to the distinct guarantees summarized above.

## References

[1] Jörg Fliege and Benar Fux Svaiter (2000). [Steepest descent methods for multicriteria optimization](https://doi.org/10.1007/s001860000043). *Mathematical Methods of Operations Research*, **51**(3), 479–494. ^ref-fliege-svaiter-2000

[2] Jean-Antoine Désidéri (2012). [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://doi.org/10.1016/j.crma.2012.03.014). *Comptes Rendus Mathématique*, **350**(5–6), 313–318. ^ref-desideri-2012

[3] Jörg Fliege, A. Ismael F. Vaz, and Luís N. Vicente (2019). [Complexity of gradient descent for multiobjective optimization](https://doi.org/10.1080/10556788.2018.1510928). *Optimization Methods and Software*, **34**(5), 949–959. ^ref-fliege-vaz-vicente-2019

[4] Matteo Lapucci (2024). [Convergence and complexity guarantees for a wide class of descent algorithms in nonconvex multi-objective optimization](https://doi.org/10.1016/j.orl.2024.107115). *Operations Research Letters*, **54**, 107115. ^ref-lapucci-2024

[5] Qi Zhang, Peiyao Xiao, Shaofeng Zou, and Kaiyi Ji (2025). [MGDA Converges under Generalized Smoothness, Provably](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html). *ICLR*. ^ref-zhang-2025

[6] Jörg Fliege, Luciana M. Graña Drummond, and Benar Fux Svaiter (2009). [Newton's Method for Multiobjective Optimization](https://doi.org/10.1137/08071692X). *SIAM Journal on Optimization*. ^ref-fliege-newton-2009

[7] Matteo Lapucci and Pierluigi Mansueto (2023). [A limited memory Quasi-Newton approach for multi-objective optimization](https://doi.org/10.1007/s10589-023-00454-7). *Computational Optimization and Applications*, **85**, 33–73. ^ref-lapucci-mansueto-lbfgs-2023

[8] Jian Chen, Liping Tang, and Xinmin Yang (2023). [A Barzilai-Borwein descent method for multiobjective optimization problems](https://doi.org/10.1016/j.ejor.2023.04.022). *European Journal of Operational Research*, **311**(1), 196–209. [Preprint (2022)](https://arxiv.org/abs/2204.08616). ^ref-chen-tang-yang-2023

[9] Hiroki Tanabe, Ellen H. Fukuda, and Nobuo Yamashita (2019). [Proximal gradient methods for multiobjective optimization and their applications](https://doi.org/10.1007/s10589-018-0043-x). *Computational Optimization and Applications*, **72**. ^ref-tanabe-prox-2019

[10] Hiroki Tanabe, Ellen H. Fukuda, and Nobuo Yamashita (2023). [An accelerated proximal gradient method for multiobjective optimization](https://doi.org/10.1007/s10589-023-00497-w). *Computational Optimization and Applications*, **86**. [Preprint (2022)](https://arxiv.org/abs/2202.10994). ^ref-tanabe-accelerated-2023

[11] Konstantin Sonntag and Sebastian Peitz (2024). [Fast Multiobjective Gradient Methods with Nesterov Acceleration via Inertial Gradient-Like Systems](https://doi.org/10.1007/s10957-024-02389-3). *Journal of Optimization Theory and Applications*, **201**, 539–582. ^ref-sonntag-peitz-2024

[12] Jörg Fliege and A. Ismael F. Vaz (2016). [A Method for Constrained Multiobjective Optimization Based on SQP Techniques](https://doi.org/10.1137/15M1016424). *SIAM Journal on Optimization*, **26**(4), 2091–2119. ^ref-fliege-vaz-2016

[13] Jean-Antoine Désidéri (2018). [Quasi-Riemannian Multiple Gradient Descent Algorithm for constrained multiobjective differential optimization](https://mgda.inria.fr/constrained). *INRIA Research Report RR-9159*. ^ref-desideri-constrained-2018

[14] Quentin Mercier, Fabrice Poirion, and Jean-Antoine Désidéri (2018). [A stochastic multiple gradient descent algorithm](https://doi.org/10.1016/j.ejor.2018.05.064). *European Journal of Operational Research*, **271**(3), 808–817. ^ref-mercier-2018

[15] S. Liu and Luís N. Vicente (2021/2024). [The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning](https://doi.org/10.1007/s10479-021-04033-z). *Annals of Operations Research*, online 2021; **339** (2024). ^ref-liu-vicente-2024

[16] Shiji Zhou, Wenpeng Zhang, Jiyan Jiang, Wenliang Zhong, Jinjie Gu, and Wenwu Zhu (2022). [On the Convergence of Stochastic Multi-Objective Gradient Manipulation and Beyond](https://proceedings.neurips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract.html). *NeurIPS*. ^ref-zhou-2022

[17] Heshan Fernando, Han Shen, Miao Liu, Subhajit Chaudhury, Keerthiram Murugesan, and Tianyi Chen (2023). [Mitigating Gradient Bias in Multi-Objective Learning: A Provably Convergent Approach](https://openreview.net/forum?id=dLAYGdKTi2). *ICLR*. ^ref-fernando-2023

[18] Lisha Chen, Heshan Fernando, Yiming Ying, and Tianyi Chen (2024). [Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance](https://www.jmlr.org/papers/v25/23-1287.html). *Journal of Machine Learning Research*, **25**(193), 1–53. ^ref-chen-fernando-ying-2024

[19] Ozan Sener and Vladlen Koltun (2018). [Multi-Task Learning as Multi-Objective Optimization](https://mlanthology.org/neurips/2018/sener2018neurips-multitask/). *NeurIPS*. ^ref-sener-koltun-2018

[20] Bo Liu, Yihao Feng, Peter Stone, and Qiang Liu (2023). [FAMO: Fast Adaptive Multitask Optimization](https://papers.neurips.cc/paper_files/paper/2023/hash/b2fe1ee8d936ac08dd26f2ff58986c8f-Abstract-Conference.html). *NeurIPS*. ^ref-famo-2023

[21] Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang, and Sam Kwong (2019). [Pareto Multi-Task Learning](https://papers.neurips.cc/paper_files/paper/2019/hash/685bfde03eb646c27ed565881917c71c-Abstract.html). *NeurIPS*. ^ref-lin-2019

[22] Debabrata Mahapatra and Vaibhav Rajan (2020). [Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization](https://proceedings.mlr.press/v119/mahapatra20a.html). *ICML*, PMLR **119**, 6597–6607 (EPO Search). ^ref-mahapatra-rajan-2020

[23] Matteo Lapucci and Pierluigi Mansueto (2023). [Improved front steepest descent for multi-objective optimization](https://doi.org/10.1016/j.orl.2023.03.001). *Operations Research Letters*, **51**(3), 242–247. ^ref-lapucci-mansueto-front-2023

[24] Tianhe Yu, Saurabh Kumar, Abhishek Gupta, Sergey Levine, Karol Hausman, and Chelsea Finn (2020). [Gradient Surgery for Multi-Task Learning](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html). *NeurIPS*. ^ref-yu-pcgrad-2020

[25] Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu (2021). [Conflict-Averse Gradient Descent for Multi-task Learning](https://papers.nips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html). *NeurIPS*. ^ref-liu-cagrad-2021

[26] Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya (2022). [Multi-Task Learning as a Bargaining Game](https://proceedings.mlr.press/v162/navon22a.html). *ICML*, PMLR **162**, 16428–16446. ^ref-navon-2022
