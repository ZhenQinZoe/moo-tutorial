---
title: Stochastic MGDA with Exact and Inexact Updates
author: Lisha Chen
create date: 2026-10-12
update date: 2026-11-09
description: How exact and inexact MGDA updates interact with stochastic gradients, and how growing batches, momentum, double sampling, and periodic weight updates control stationarity error.
tags:
  - Multi-objective-optimization
  - MGDA
  - Stochastic-optimization
  - Multi-objective-optimization
  - Momentum
type: blog
status: draft
---

# Stochastic MGDA with Exact and Inexact Direction Updates

[Deterministic MGDA](Intro%20Deterministic%20MGDA.md) computes a common-descent direction from all objective gradients. In learning problems, those gradients are often estimated from mini-batches. We can still solve the MGDA subproblem **exactly** at every iteration, but the resulting direction is exact only for the gradient matrix supplied to that subproblem. Stochastic error remains, and the direction is generally biased even when every individual gradient estimate is unbiased.

:::danger

*Q: If we solve the MGDA direction subproblem exactly using stochastic gradients, when do the iterates approach Pareto stationarity for the empirical objectives?*

:::

This post follows [Chen (2026, v2; September 6)](https://arxiv.org/abs/2607.18174v2) [\[1\]](#^ref-chen-2026). The main result covers vanilla stochastic multi-gradient descent (SMG) with growing batches; the paper's Appendix B analyzes fixed-batch momentum variants with the same exact inner solve. Earlier SMG work used increasing batches [Liu and Vicente (2024)](https://doi.org/10.1007/s10479-021-04033-z) [\[2\]](#^ref-liu-vicente-2024), while [Chen et al. (2024)](https://www.jmlr.org/papers/v25/23-1287.html) [\[3\]](#^ref-chen-et-al-2024) analyzed the bias through regularity of the *direction*. The newer proof instead uses regularity of the *direction norm*, which is a simpler stationarity measure.

## Empirical objectives and Pareto stationarity

<div style="background-color:#f2f2f2; border-left:5px solid #9e9e9e; padding:12px 16px; border-radius:4px; margin-bottom:16px;">

**Notation.** Let $S=(z_1,\ldots,z_N)$ be a fixed dataset and let $f_{z,m}:\mathbb{R}^n\to\mathbb{R}$ be the loss of objective $m\in[M]=\{1,\ldots,M\}$ on sample $z$. The empirical objective and objective vector are
$$
f_{S,m}(x)=\frac1N\sum_{j=1}^N f_{z_j,m}(x),
\qquad
F_S(x)=\bigl(f_{S,1}(x),\ldots,f_{S,M}(x)\bigr)^\top.
$$
Write $G_S(x)=[\nabla f_{S,1}(x),\ldots,\nabla f_{S,M}(x)]\in\mathbb{R}^{n\times M}$ and $\Delta_M=\{\lambda\in\mathbb{R}^M:\lambda_m\geq0\text{ for }m\in[M],\ \sum_{m\in[M]}\lambda_m=1\}$.

</div>

As in the deterministic note, $x$ is *Pareto stationary* if no direction $v$ satisfies $\nabla f_{S,m}(x)^\top v<0$ for every $m\in[M]$. By the theorem of alternatives, this is equivalent to $0\in\operatorname{conv}\{\nabla f_{S,m}(x):m\in[M]\}$. We measure its violation by
$$
R_S(x):=\min_{\lambda\in\Delta_M}\|G_S(x)\lambda\|^2.
\tag{1}
$$
^eq-stochastic-ps

Thus $R_S(x)=0$ exactly at empirical Pareto-stationary points. This is a **squared** stationarity measure; rates below refer to $\mathbb{E}[R_S(x_\tau)]$, not to the distance from a Pareto optimum. In particular, a rate for $R_S$ should not be reported as the same rate for $\sqrt{R_S}$.

## Exact MGDA update on a mini-batch

At iteration $t$, sample a mini-batch $Z_t$ from $S$ and form $Q_t=[q_{t,1},\ldots,q_{t,M}]$, where $q_{t,m}=\nabla f_{Z_t,m}(x_t)$ is the mini-batch gradient of objective $m$.

:::info

Vanilla SMG solves the minimum-norm problem completely:
$$
\lambda_t\in\operatorname*{argmin}_{\lambda\in\Delta_M}\|Q_t\lambda\|^2,
\qquad
d_t=-Q_t\lambda_t,
\qquad
x_{t+1}=x_t+\alpha_t d_t.
\tag{2}
$$
^eq-smg-update

:::

The optimum weight vector need not be unique, but $d_t$ is unique: $-d_t$ is the projection of the origin onto the convex hull of the columns of $Q_t$. The word **exact** in [[#^eq-smg-update|(2)]] concerns this inner minimum-norm problem. It does not claim that $Q_t=G_S(x_t)$ or that $d_t$ equals the full-gradient MGDA direction. In general,
$$
\mathbb{E}[Q_t\mid x_t]=G_S(x_t)
\quad\not\Longrightarrow\quad
\mathbb{E}[d_t\mid x_t]=-G_S(x_t)\lambda^\star_{G_S(x_t)},
$$
because $\lambda_t$ depends nonlinearly on the sampled gradients.

:::success

<a id="prop-stochastic-mgda-geometry"></a>

**Proposition 1 (Sampled descent and stability of the stationarity norm).**

For any matrix $Q=[q_1,\ldots,q_M]\in\mathbb{R}^{n\times M}$, let $\lambda_Q\in\operatorname*{argmin}_{\lambda\in\Delta_M}\|Q\lambda\|^2$, $d_Q=-Q\lambda_Q$, and $r(Q)=\|d_Q\|$. Then:

**(i)** $q_m^\top d_Q\leq-\|d_Q\|^2$ for every $m\in[M]$.

**(ii)** For all $Q,Q'\in\mathbb{R}^{n\times M}$,
$$
|r(Q)-r(Q')|\leq\|Q-Q'\|_2\leq\|Q-Q'\|_F.
\tag{3}
$$
^eq-norm-stability

The first claim concerns the columns of the matrix used in the update. The second concerns the norm of the MGDA direction, not the direction vector or the optimal weights.

:::

<a id="proof-proposition-1"></a>

**Proof of Proposition 1.** Let $u=-d_Q=Q\lambda_Q$. Since $u$ is the projection of the origin onto $\operatorname{conv}\{q_1,\ldots,q_M\}$, projection optimality gives $u^\top(q_m-u)\geq0$ for each $m$. Rearrangement proves (i). For any $\lambda\in\Delta_M$, the triangle inequality and $\|\lambda\|_2\leq1$ give $\|Q\lambda\|\leq\|Q'\lambda\|+\|Q-Q'\|_2$. Choose a minimizer for $r(Q')$ as the comparator to obtain $r(Q)\leq r(Q')+\|Q-Q'\|_2$; interchange $Q$ and $Q'$ to obtain the reverse inequality. Finally, $\|Q-Q'\|_2\leq\|Q-Q'\|_F$. <span class="qed" style="float:right;">&#9633;</span>

:::warning

**Key idea.** Applying (ii) with $Q'=G_S(x_t)$ gives the key bridge from the stochastic update to empirical stationarity:
$$
\sqrt{R_S(x_t)}
=r(G_S(x_t))
\leq\|d_t\|+\|Q_t-G_S(x_t)\|_F.
\tag{4}
$$
^eq-stochastic-bridge

Notice why controlling $r(Q)=\|d_Q\|$ helps: the direction map itself need not be Lipschitz in $Q$, and the optimal weights can behave even less regularly [Chen (2026, Lemma 2)](https://arxiv.org/html/2607.18174v2) [\[1\]](#^ref-chen-2026).

:::

## Convergence with growing batches

**Assumptions.** Assume every sample loss $f_{z,m}$ is $L$-smooth and every empirical objective is bounded below by $f_{S,m}^{\inf}:=\inf_x f_{S,m}(x)>-\infty$. Conditional on the history before $Z_t$ is drawn, suppose the mini-batch gradients are unbiased and satisfy
$$
\mathbb{E}[q_{t,m}\mid\mathcal F_t]=\nabla f_{S,m}(x_t),
\qquad
\mathbb{E}\!\left[\|q_{t,m}-\nabla f_{S,m}(x_t)\|^2\mid\mathcal F_t\right]
\leq\frac{\sigma^2}{|Z_t|}.
\tag{5}
$$
^eq-oracle-assumption

Here $\mathcal F_t$ contains the randomness before the batch at iteration $t$. Independent uniform sampling *with replacement* from the fixed dataset gives the first condition; a uniform per-sample variance bound gives the second. No bounded-gradient assumption is used. Let $A_T=\sum_{t=0}^{T-1}\alpha_t$ and $V_T=\sum_{t=0}^{T-1}\alpha_t/|Z_t|$. Choose an output index $\tau$ independently of the batches with $\Pr(\tau=t)=\alpha_t/A_T$.

:::success

<a id="thm-growing-batches"></a>

**Theorem 1 (Exact SMG updates with growing batches; Chen 2026, Theorem 1).**

Under the assumptions above, let $x_0$ be deterministic and let [[#^eq-smg-update|(2)]] use deterministic step sizes $0<\alpha_t\leq1/(2L)$ and batch sizes $|Z_t|\geq1$. For every objective $m\in[M]$,
$$
\mathbb{E}[R_S(x_\tau)]
\leq
\frac{8\bigl(f_{S,m}(x_0)-f_{S,m}^{\inf}\bigr)
+(4+2M)\sigma^2 V_T}{A_T}.
\tag{6}
$$
^eq-smg-rate

The expectation includes both the sampled batches and the randomized output index.

:::

<a id="proof-theorem-1"></a>

**Proof of Theorem 1.** Set $\xi_{t,m}=q_{t,m}-\nabla f_{S,m}(x_t)$ and $E_t=Q_t-G_S(x_t)$. Proposition 1(i) and Young's inequality imply $\nabla f_{S,m}(x_t)^\top d_t\leq-\frac12\|d_t\|^2+\frac12\|\xi_{t,m}\|^2$. The smoothness bound, $x_{t+1}=x_t+\alpha_t d_t$, and $L\alpha_t\leq1/2$ therefore give
$$
f_{S,m}(x_{t+1})
\leq f_{S,m}(x_t)-\frac{\alpha_t}{4}\|d_t\|^2
+\frac{\alpha_t}{2}\|\xi_{t,m}\|^2.
$$
By [[#^eq-stochastic-bridge|(4)]] and $(a+b)^2\leq2a^2+2b^2$, we also have $R_S(x_t)\leq2\|d_t\|^2+2\|E_t\|_F^2$. Eliminating $\|d_t\|^2$ yields
$$
\alpha_tR_S(x_t)
\leq8\bigl(f_{S,m}(x_t)-f_{S,m}(x_{t+1})\bigr)
+2\alpha_t\|E_t\|_F^2+4\alpha_t\|\xi_{t,m}\|^2.
$$
Take expectations, use $\mathbb E\|E_t\|_F^2\leq M\sigma^2/|Z_t|$ and $\mathbb E\|\xi_{t,m}\|^2\leq\sigma^2/|Z_t|$, then sum over $t$. The objective values telescope and the lower bound $f_{S,m}^{\inf}$ gives the numerator of [[#^eq-smg-rate|(6)]]. Division by $A_T$ converts the weighted average into $\mathbb E[R_S(x_\tau)]$. <span class="qed" style="float:right;">&#9633;</span>

**Corollary (Linearly growing batches).** For a constant step size $\alpha\in(0,1/(2L)]$ and linearly growing batches $|Z_t|\geq b(t+1)$, the harmonic-sum bound $V_T\leq(\alpha/b)(1+\log T)$ gives
$$
\mathbb E[R_S(x_\tau)]
\leq
\frac{8\bigl(f_{S,m}(x_0)-f_{S,m}^{\inf}\bigr)}{\alpha T}
+\frac{(4+2M)\sigma^2(1+\log T)}{bT}
=\widetilde O(T^{-1})
\tag{7}
$$
^eq-linear-batch-rate

for fixed problem parameters. This is an **iteration** rate; when $|Z_t|=\Theta(t+1)$, the total number of individual sample-gradient evaluations through iteration $T$ is $\Theta(T^2)$. The theorem also covers other deterministic batch and step schedules through the ratio $V_T/A_T$.

## Regularity-aware updates: the revised MoRe rate

[Huang and Chen (2026, v2; October 23)](https://arxiv.org/abs/2607.15412v2) [\[7\]](#^ref-huang-chen-2026) analyze **MoRe**: solve MGDA when the sampled matrix passes a reduced-nondegeneracy test; otherwise use fixed scalarization weights $\bar\lambda\in\Delta_M$, skipping the MGDA solve.

:::success

Their revised Theorem 1 gives $\widetilde O(T^{-1})$ squared empirical PS convergence. Define $\Phi(x)=\bar\lambda^\top F_S(x)$ and $\Delta_0=\Phi(x_0)-\inf_x\Phi(x)$. For constant $0<\alpha\leq1/(2L)$ and any positive thresholds $\mu_t$,
$$
\mathbb E\!\left[\min_{0\leq t<T}R_S(x_t)\right]
\leq\frac1T\sum_{t=0}^{T-1}\mathbb E[R_S(x_t)]
\leq
\frac{8\Delta_0}{\alpha T}
+\frac{6M\sigma^2}{T}\sum_{t=0}^{T-1}\frac1{|Z_t|}
=\widetilde O(T^{-1})
\quad\text{when }|Z_t|=\Theta(t+1).
$$
The average bound appears in Appendix B.C and also applies to a uniformly sampled iterate. For general $M$, the PS result requires smoothness, finite lower bounds, and unbiased gradients with bounded variance; bounded gradients and uniform nondegeneracy are unnecessary.

:::

The separate conflict-avoidance theorem assumes $M=2$ and bounded gradients. With linear batches and $\mu_t=\Theta((t+1)^{-1/2})$, the expected **squared direction error** is $O((t+1)^{-1/2})$.

## Exact updates with fixed batches: MoCo and MoCo+

Growing batches are not the only way to control the discrepancy in [[#^eq-stochastic-bridge|(4)]]. Appendix B of [Chen (2026)](https://arxiv.org/html/2607.18174v2) [\[1\]](#^ref-chen-2026) uses a fixed batch size and a tracked gradient matrix $Y_t=[y_{t,1},\ldots,y_{t,M}]$. Both variants initialize $Y_0=\nabla F_{Z_0}(x_0)$, then make the **same exact MGDA update**
$$
\lambda_t\in\operatorname*{argmin}_{\lambda\in\Delta_M}\|Y_t\lambda\|^2,
\qquad
d_t=-Y_t\lambda_t,
\qquad
x_{t+1}=x_t+\alpha_t d_t.
\tag{8}
$$
^eq-tracked-update

Here $\nabla F_Z(x)$ denotes the matrix of all mini-batch objective gradients at $x$. After computing $x_{t+1}$, draw a fresh batch $Z_{t+1}$. The two methods differ in how they update $Y_t$:

- **MoCo** uses exponential gradient tracking [Fernando et al. (2023)](https://openreview.net/forum?id=dLAYGdKTi2) [\[4\]](#^ref-fernando-2023):
  $$
  Y_{t+1}=(1-\beta_{t+1})Y_t+\beta_{t+1}\nabla F_{Z_{t+1}}(x_{t+1}).
  $$
- **MoCo+** uses a corrected momentum estimator [Fernando et al. (2024)](https://doi.org/10.1109/ICASSP48485.2024.10446038) [\[5\]](#^ref-fernando-2024):
  $$
  Y_{t+1}
  =\nabla F_{Z_{t+1}}(x_{t+1})
  +(1-\gamma_{t+1})
  \bigl(Y_t-\nabla F_{Z_{t+1}}(x_t)\bigr).
  $$

For MoCo+, both gradient evaluations use the **same fresh batch**, drawn after $x_{t+1}$ has been fixed. This coupling controls the noise in the gradient difference. These are the exact-inner-solve variants analyzed in the paper; the originally proposed momentum algorithms may use an inexact inner step or regularization.

The proof keeps the same two ingredients as Theorem 1: descent up to the matrix error $E_t=Y_t-G_S(x_t)$ and the stability bound $R_S(x_t)\leq2\|d_t\|^2+2\|E_t\|_F^2$. Momentum supplies a recursion for $\mathbb E\|E_t\|_F^2$. The comparison below adapts [Chen (2026, v2, Table 2)](https://arxiv.org/html/2607.18174v2) [\[1\]](#^ref-chen-2026), including the original momentum methods alongside the exact-solve variants. The MoRe row uses [Huang and Chen's October v2](https://arxiv.org/abs/2607.15412v2) [\[7\]](#^ref-huang-chen-2026), which supersedes the older MoRe rate in that table. Rates concern the squared PS measure in each paper's setting; $T$ counts outer iterations and $\widetilde O$ hides logarithmic factors.

| Method | Batch | Used continuity | Variance/bias control | Squared PS measure convergence rate |
| --- | --- | --- | --- | --- |
| SMG (prior) ([Liu and Vicente, 2024](https://doi.org/10.1007/s10479-021-04033-z)) | $\Theta(t+1)$ | Assumed Lipschitz $\lambda_Q^\star$ | Growing batches | convex rates only, no nonconvex PS rate |
| CR-MOGM ([Zhou et al., 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract-Conference.html)) | $\Theta(1)$ | — | Exponential averaging of aggregation weights | $O(T^{-1/4})$ |
| MoCo$^\dagger$ ([Fernando et al., 2023](https://openreview.net/forum?id=dLAYGdKTi2)) | $\Theta(1)$ | — | Momentum on gradients, regularization, and one inner SGD step | $O(T^{-1/2})$ |
| SMG (reanalysis) ([Chen et al., 2024](https://www.jmlr.org/papers/v25/23-1287.html)) | $\Theta(t+1)$ | Proved $\frac{1}{2}$-Hölder direction $d_Q$ | Growing batches | $\widetilde{O}(T^{-1/4})$ |
| ASSMG$^\ddagger$ ([Zhao et al., 2024](https://doi.org/10.1007/s10957-023-02334-w)) | Adaptive | Assumed Lipschitz $\lambda_Q^\star$ | Adaptive batches with a relative-error batch test | $O(T^{-1})$ |
| MoCo+ ([Fernando et al., 2024](https://doi.org/10.1109/ICASSP48485.2024.10446038)) | $\Theta(1)$ | — | STORM-type momentum on gradients and one inner SGD step | $O(T^{-2/3})$ |
| MoRe ([Huang and Chen, 2026, v2, Theorem 1](https://arxiv.org/html/2607.15412v2)) | $\Theta(t+1)$ | Proved Lipschitz norm $\|d_Q\|$ for PS analysis | Regularity test and fixed-weight scalarization fallback | $\widetilde{O}(T^{-1})$ |
| [SMG (reanalysis, Chen 2026) Theorem 1](https://arxiv.org/html/2607.18174v2) | $\Theta(t+1)$ | Proved Lipschitz norm $\|d_Q\|$ | Growing batches | $\widetilde{O}(T^{-1})$ |
| [MoCo (exact MGDA, reanalysis, Chen 2026) Theorem 4](https://arxiv.org/html/2607.18174v2) | $\Theta(1)$ | Proved Lipschitz norm $\|d_Q\|$ | Exponential-type momentum on gradients | $O(T^{-1/2})$ |
| [MoCo+ (exact MGDA, Chen 2026) Theorem 5](https://arxiv.org/html/2607.18174v2) | $\Theta(1)$ | Proved Lipschitz norm $\|d_Q\|$ | STORM-type momentum on gradients | $\widetilde{O}(T^{-2/3})$ |

In the “Used continuity” column, “Assumed” means an extra hypothesis, “Proved” means an established property, and “—” means no continuity of the unregularized MGDA map is invoked directly. MoRe v2 uses norm stability for PS convergence; direction regularity motivates its branch test.

The MoCo rate marked $\dagger$ is its original Theorem 3: it requires bounded gradients and objective values along the trajectory, plus a bias condition satisfied by unbiased oracles. Its Theorem 1 gives $O(T^{-1/12})$ under weaker conditions. CR-MOGM's nonconvex result also assumes bounded gradients and trajectory objective values. The ASSMG rate marked $\ddagger$ uses adaptive batches without a total stochastic-gradient evaluation bound. The 2024 SMG reanalysis uses bounded gradients; Chen (2026)'s exact-solve results and MoRe v2's PS theorem do not. These iteration rates have different sampling costs and guarantee settings.

For the exact-solve MoCo rate in Theorem 4, $\alpha_t,\beta_{t+1}=\Theta(T^{-1/2})$ are constant over a fixed horizon, with sufficiently small constants; diminishing schedules proportional to $(t+1)^{-1/2}$ instead give $\widetilde O(T^{-1/2})$. For exact-solve MoCo+, Theorem 5 uses $\alpha_t=\Theta((t+1)^{-1/3})$ and $\gamma_{t+1}=c_\gamma\alpha_t^2$ with admissible constants. Both use fixed batches; MoCo+ evaluates each fresh batch at two consecutive iterates.

The point of the analysis is that exact solution of the inner MGDA problem preserves a useful projection inequality, while the stochastic estimator error determines whether that sampled direction says anything about empirical Pareto stationarity. The norm stability in Proposition 1 connects those two pieces without imposing Lipschitz continuity on the weights.

## Inexact MGDA updates: single-step and periodic weight updates

An exact MGDA solve minimizes a quadratic over $\Delta_M$ before each model update. An inexact method instead carries its weights across iterations and takes one or a few projected steps. This saves aggregation work but introduces an additional error: the chosen combination need not satisfy the projection inequality in Proposition 1(i). The original MoCo and MoCo+ belong to this class; the exact-solve variants above replace their weight recursion by a complete minimum-norm solve.

### Descent with inexact updates

For a matrix estimate $Q_t$ and any feasible weights $\lambda_t\in\Delta_M$, set $u_t=Q_t\lambda_t$, $d_t=-u_t$, and $x_{t+1}=x_t-\alpha_tu_t$. Define the inner optimality residual
$$
\eta_t:=\max_{m\in[M]}\bigl\{\|u_t\|^2-q_{t,m}^\top u_t\bigr\}
=\max_{\lambda\in\Delta_M}
\left\langle Q_t^\top Q_t\lambda_t,\lambda_t-\lambda\right\rangle
\geq0.
\tag{9}
$$
^eq-inexact-residual

This is the Frank–Wolfe gap of $h_t(\lambda)=\frac12\|Q_t\lambda\|^2$ at $\lambda_t$. It vanishes at an exact minimizer and gives $q_{t,m}^\top d_t\leq-\|d_t\|^2+\eta_t$ for every objective. Norm stability still gives
$$
R_S(x_t)\leq2\|u_t\|^2+2\|E_t\|_F^2,
\qquad E_t:=Q_t-G_S(x_t),
$$
because $r(Q_t)\leq\|Q_t\lambda_t\|$ for every feasible $\lambda_t$. Thus, approximation affects the descent inequality even though the stationarity comparison survives.

:::success

<a id="prop-inexact-mgda-descent"></a>

**Proposition 2 (An inexact-update extension of Theorem 1; derived here).**

Suppose the empirical objectives are $L$-smooth and bounded below, and $x_0$ is deterministic. Let $Q_t$ be any gradient-matrix estimates, let $\lambda_t\in\Delta_M$, and assume the expectations below are finite. With deterministic $0<\alpha_t\leq1/(2L)$ and an independent output index satisfying $\Pr(\tau=t)=\alpha_t/A_T$, for every $m\in[M]$,
$$
\mathbb E[R_S(x_\tau)]
\leq\frac{1}{A_T}\left[
8\bigl(f_{S,m}(x_0)-f_{S,m}^{\inf}\bigr)
+\sum_{t=0}^{T-1}\alpha_t\left(
8\mathbb E[\eta_t]
+2\mathbb E\|E_t\|_F^2
+4\mathbb E\|E_{t,m}\|^2
\right)\right],
\tag{10}
$$
^eq-inexact-rate

where $E_{t,m}$ is the $m$-th column of $E_t$.

:::

<a id="proof-proposition-2"></a>

**Proof of Proposition 2.** Since $\nabla f_{S,m}(x_t)=q_{t,m}-E_{t,m}$, the residual inequality and Young's inequality give
$$
\nabla f_{S,m}(x_t)^\top d_t
\leq-\frac12\|d_t\|^2+\eta_t+\frac12\|E_{t,m}\|^2.
$$
Smoothness and $L\alpha_t\leq1/2$ imply
$$
f_{S,m}(x_{t+1})
\leq f_{S,m}(x_t)-\frac{\alpha_t}{4}\|d_t\|^2
+\alpha_t\eta_t+\frac{\alpha_t}{2}\|E_{t,m}\|^2.
$$
Use $R_S(x_t)\leq2\|d_t\|^2+2\|E_t\|_F^2$, rearrange, take expectations, and telescope as in Theorem 1. <span class="qed" style="float:right;">&#9633;</span>

**Consequence.** For raw mini-batch gradients satisfying [[#^eq-oracle-assumption|(5)]], [[#^eq-inexact-rate|(10)]] is exactly [[#^eq-smg-rate|(6)]] plus $8A_T^{-1}\sum_t\alpha_t\mathbb E[\eta_t]$. Hence a fixed nonzero inner tolerance leaves an error floor in this bound. With constant steps and linear batches, $\mathbb E[\eta_t]=O((t+1)^{-1})$ suffices to preserve $\widetilde O(T^{-1})$ convergence. These are sufficient conditions derived here, not the convergence theorems of the methods below. Their fixed-batch proofs exploit estimator recursions or telescoping weight errors rather than requiring this nonnegative residual to be small at every iteration.

### Momentum and double sampling control different errors

**MoCo and MoCo+.** [Fernando et al. (2023)](https://openreview.net/forum?id=dLAYGdKTi2) [\[4\]](#^ref-fernando-2023) use a tracked matrix $Y_t$ and a projected weight step of the form
$$
\lambda_{t+1}
=\Pi_{\Delta_M}\!\left[\lambda_t-\gamma_t
\bigl(Y_t^\top Y_t+\rho_t I\bigr)\lambda_t\right].
$$

Here $\gamma_t$ is a weight stepsize and $\rho_t\geq0$ regularizes the weight problem; this notation is local to this subsection.

The model step uses $Y_t\lambda_t$ in the original MoCo ordering. Gradient tracking controls sampling error, while the projected recursion controls weight error. Positive regularization makes the weight problem strongly convex; it also changes its minimizer. The best displayed $O(T^{-1/2})$ rate is Theorem 3 under the boundedness and oracle conditions noted above.

[MoCo+ (Fernando et al., 2024)](https://doi.org/10.1109/ICASSP48485.2024.10446038) [\[5\]](#^ref-fernando-2024) combines a STORM-type recursive gradient estimator with an inexact weight step. Its $O(T^{-2/3})$ squared PS rate improves the sampling exponent, while avoiding a complete MGDA solve at each iteration. The exact-solve MoCo+ analysis above has the same exponent up to logarithms, with a different inner update and assumptions.

**MoDo.** [Chen et al. (2024)](https://www.jmlr.org/papers/v25/23-1287.html) [\[3\]](#^ref-chen-et-al-2024) estimate the Gram-matrix action using two independent gradient samples. Conditional on the history $\mathcal F_t$, for history-measurable $\lambda_t$ and independent unbiased $Q_t^{(1)},Q_t^{(2)}$,
$$
\mathbb E\!\left[(Q_t^{(1)})^\top Q_t^{(2)}\lambda_t\mid\mathcal F_t\right]
=G_S(x_t)^\top G_S(x_t)\lambda_t.
$$
A single projected step uses this estimator plus $\rho\lambda_t$. Double sampling removes the covariance term that appears in $\mathbb E[Q_t^\top Q_t]$; it does not make the nonlinear MGDA direction unbiased. Remark 3 gives the squared empirical PS bound $O((\alpha T)^{-1}+\alpha+\gamma+\rho)$, yielding $O(T^{-1/2})$ for $\alpha,\gamma=\Theta(T^{-1/2})$ and $\rho=O(T^{-1/2})$. Its CA-distance bound can remain constant under this choice. The reference is the expanded JMLR version of the NeurIPS 2023 paper listed in Chen's table.

**SDMGrad.** [Xiao et al. (2023)](https://arxiv.org/abs/2305.18409) [\[9\]](#^ref-xiao-2023) use independent double sampling for a direction-oriented weight problem. With the gradient $g_0(x)=G_S(x)\bar\lambda$ of a fixed reference scalarization and parameter $\kappa\geq0$, its target and sampled model step are (with $q_{0,t}=Q_t\bar\lambda$)
$$
\min_{\lambda\in\Delta_M}\frac12\|G_S(x)\lambda+\kappa g_0(x)\|^2,
\qquad
x_{t+1}=x_t-\alpha_t\bigl(Q_t\lambda_t+\kappa q_{0,t}\bigr).
$$
The case $\kappa=0$ recovers the MGDA target. Theorem 3 gives $O(T^{-1/2})$ squared PS convergence with one inner projected step, constant $\kappa$, and fixed batches. This setting permits constant-level CA error. Requiring vanishing CA error uses a different inner-work budget; the displayed rate does not claim both properties at the same cost.

### Comparison of inexact, modified, and reused updates

The table selects the corresponding rows of [Chen (2026, v2, Table 2)](https://arxiv.org/html/2607.18174v2) [\[1\]](#^ref-chen-2026). It retains the five-column format above. “—” means that the analysis does not directly assume continuity of the unregularized MGDA weights or direction; regularized weight maps can still have useful regularity. The rates suppress objective-count dependence and use each source's objective and probability setting.

| Method | Batch | Used continuity | Variance/bias control | Squared PS measure convergence rate |
| --- | --- | --- | --- | --- |
| CR-MOGM ([Zhou et al., 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract-Conference.html)) [\[8\]](#^ref-zhou-2022) | $\Theta(1)$ | — | Weight momentum and one inner SGD step | $O(T^{-1/4})$ |
| MoCo$^\dagger$ ([Fernando et al., 2023](https://openreview.net/forum?id=dLAYGdKTi2)) [\[4\]](#^ref-fernando-2023) | $\Theta(1)$ | — | Gradient momentum, regularized weights, one projected weight step | $O(T^{-1/2})$ |
| MoDo ([Chen et al., 2024; NeurIPS 2023 precursor](https://www.jmlr.org/papers/v25/23-1287.html)) [\[3\]](#^ref-chen-et-al-2024) | $\Theta(1)$ | — | Independent double sampling and one projected weight step | $O(T^{-1/2})$ |
| SDMGrad$^\S$ ([Xiao et al., 2023](https://arxiv.org/abs/2305.18409)) [\[9\]](#^ref-xiao-2023) | $\Theta(1)$ | — | Double sampling; one weight step for a direction-oriented target | $O(T^{-1/2})$ |
| MoCo+ ([Fernando et al., 2024](https://doi.org/10.1109/ICASSP48485.2024.10446038)) [\[5\]](#^ref-fernando-2024) | $\Theta(1)$ | — | STORM-type gradient momentum and one projected weight step | $O(T^{-2/3})$ |
| SGSMGrad$^\P$ ([Zhang et al., 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html)) [\[10\]](#^ref-zhang-2025) | $\Theta(1)$ | — | Double sampling, one weight step, high-probability control | $O(T^{-1/2})$ |
| PSMGD$^\ast$ ([Xu et al., 2025](https://ojs.aaai.org/index.php/AAAI/article/view/35482)) [\[11\]](#^ref-xu-2025) | $\Theta(1)$ | — | Periodic subproblem solves and blended, reused weights | $O(T^{-1/2})$ |

$\dagger$ MoCo's rate uses its stronger Theorem 3 assumptions; $O(T^{-1/12})$ is the weaker-condition guarantee. $\S$ SDMGrad's one-step rate allows constant-level CA error. $\P$ SGSMGrad's rate is a high-probability guarantee under generalized smoothness, not an expectation bound under the globally $L$-smooth assumptions of Proposition 2. $\ast$ PSMGD solves a subproblem periodically and reuses weights, so it is included as a related approximation to the current MGDA update rather than a one-step inner solver.

**Related mechanisms.** CR-MOGM averages aggregation weights to reduce their correlation with current gradient noise [\[8\]](#^ref-zhou-2022). Its framework also permits base aggregation rules beyond MGDA, so a weight-averaging step should not be identified with gradient tracking.

[SGSMGrad (Zhang et al., 2025)](https://arxiv.org/abs/2405.19440) [\[10\]](#^ref-zhang-2025) combines double sampling with projected weight updates under generalized $\ell$-smoothness. Its analysis controls objective values and gradient sizes through a stopping-time argument, without assuming globally bounded gradients. The stochastic theorem gives an $O(\epsilon^2)$ average squared stationarity bound with high probability after $T=\Theta(\epsilon^{-4})$ iterations, explaining the $O(T^{-1/2})$ entry. Stronger iteration-wise CA guarantees require additional warm-start and batch costs.

[PSMGD (Xu et al., 2025)](https://arxiv.org/abs/2412.10961) [\[11\]](#^ref-xu-2025) computes a new sampled MGDA solution every prescribed number of iterations, blends it with earlier weights, and reuses the weights between refreshes. Its nonconvex theorem assumes smoothness, unbiased gradients with bounded variance, bounded weights and gradients, and an upper bound on objective values. The approximation error comes from blending and staleness even when each scheduled subproblem is solved exactly. The reported rate requires the paper's prescribed parameter choices; it does not cover arbitrary refresh intervals.

These methods reduce different costs and errors: momentum improves gradient estimates, double sampling estimates the weight gradient without a same-sample covariance bias, and periodic updates reduce weight-refresh frequency. Small PS error alone does not imply that every update closely tracks the exact MGDA direction. For fixed-size batches and a fixed number of sampling calls per iteration, squared PS rates of $T^{-1/2}$ and $T^{-2/3}$ correspond to $O(\epsilon^{-2})$ and $O(\epsilon^{-3/2})$ sampling costs for a squared PS tolerance $\epsilon$, respectively, with the stated qualifications. Growing-batch SMG's $\widetilde O(T^{-1})$ iteration rate instead uses $\Theta(T^2)$ samples. Comparing iteration exponents therefore also requires checking sampling, inner-solve work, and the desired CA guarantee.

## References

[1] L. Chen (2026). [Improved Convergence Rates for Stochastic Multi-Gradient Descent which Close the Gap: A Proof by AI](https://arxiv.org/abs/2607.18174v2). arXiv:2607.18174v2, revised September 6, 2026. ^ref-chen-2026

[2] S. Liu and L. N. Vicente (2024). [The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning](https://doi.org/10.1007/s10479-021-04033-z). *Annals of Operations Research*, **339**(3), 1119–1148. ^ref-liu-vicente-2024

[3] L. Chen, H. D. Fernando, Y. Ying, and T. Chen (2024). [Three-way trade-off in multi-objective learning: optimization, generalization and conflict-avoidance](https://www.jmlr.org/papers/v25/23-1287.html). *Journal of Machine Learning Research*, **25**(193), 1–53. ^ref-chen-et-al-2024

[4] H. D. Fernando, H. Shen, M. Liu, S. Chaudhury, K. Murugesan, and T. Chen (2023). [Mitigating gradient bias in multi-objective learning: a provably convergent approach](https://openreview.net/forum?id=dLAYGdKTi2). *International Conference on Learning Representations*. ^ref-fernando-2023

[5] H. D. Fernando, L. Chen, S. Lu, P. Chen, M. Liu, S. Chaudhury, K. Murugesan, G. Liu, M. Wang, and T. Chen (2024). [Variance reduction can improve trade-off in multi-objective learning](https://doi.org/10.1109/ICASSP48485.2024.10446038). *IEEE International Conference on Acoustics, Speech and Signal Processing*, 6975–6979. ^ref-fernando-2024

[6] Y. Zhao, W. Chen, and X. Yang (2024). [Adaptive sampling stochastic multigradient algorithm for stochastic multiobjective optimization](https://doi.org/10.1007/s10957-023-02334-w). *Journal of Optimization Theory and Applications*, **200**(1), 215–241. ^ref-zhao-2024

[7] C. Huang and L. Chen (2026). [Regularity-Aware Stochastic MGDA with Adaptive Per-Iterate Conflict-Avoidant Update](https://arxiv.org/abs/2607.15412v2). arXiv:2607.15412v2, revised October 23, 2026. ^ref-huang-chen-2026

[8] S. Zhou, W. Zhang, J. Jiang, W. Zhong, J. Gu, and W. Zhu (2022). [On the convergence of stochastic multi-objective gradient manipulation and beyond](https://papers.neurips.cc/paper_files/paper/2022/hash/f91bd64a3620aad8e70a27ad9cb3ca57-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, **35**, 38103–38115. ^ref-zhou-2022

[9] P. Xiao, H. Ban, and K. Ji (2023). [Direction-oriented multi-objective learning: simple and provable stochastic algorithms](https://proceedings.neurips.cc/paper_files/paper/2023/file/0e5b96f97c1813bb75f6c28532c2ecc7-Paper-Conference.pdf). *Advances in Neural Information Processing Systems*, **36**, 4509–4533. ^ref-xiao-2023

[10] Q. Zhang, P. Xiao, S. Zou, and K. Ji (2025). [MGDA converges under generalized smoothness, provably](https://proceedings.iclr.cc/paper_files/paper/2025/hash/1c3d419b754cb4de0a67a453cb28d959-Abstract-Conference.html). *International Conference on Learning Representations*. ^ref-zhang-2025

[11] M. Xu, P. Ju, J. Liu, and H. Yang (2025). [PSMGD: Periodic stochastic multi-gradient descent for fast multi-objective optimization](https://ojs.aaai.org/index.php/AAAI/article/view/35482). *Proceedings of the AAAI Conference on Artificial Intelligence*, **39**(20), 21770–21778. ^ref-xu-2025
