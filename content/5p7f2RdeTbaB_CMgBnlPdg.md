---
title: "Scalarization Methods for Multi-objective Optimization: From Weighted Sums to Preference-Guided Solutions"
author: Liuyuan Jiang
create date: 2026-11-02
update date: 2026-11-02
description: Planned tutorial on classical and modern scalarization methods, their Pareto guarantees, geometric interpretations, and preference-based Pareto-front exploration.
tags:
  - Multi-objective-optimization
  - Pareto-optimality
  - Scalarization
  - Preference-optimization
type: blog
status: outline
---

# Scalarization Methods for Multi-objective Optimization: From Weighted Sums to Preference-Guided Solutions

In multi-objective optimization, several objectives must be balanced without a single ordering of their values. **Scalarization** converts the vector-valued problem into a family of scalar optimization problems. The choice of scalarization determines which trade-offs are accessible, how preferences are represented, and which Pareto-optimality guarantees follow from solving a subproblem.

This tutorial will develop scalarization methods from their geometry and optimality properties, beginning with weighted sums and progressing to nonlinear, constrained, and reference-point formulations. It builds on the companion *Introduction to Multi-objective Optimization* and *Introduction to MGDA* notes. The focus is **which Pareto solutions each scalarization guarantees and can recover**, followed by the geometry and algorithms used to obtain them. Throughout, the tutorial distinguishes global optimality, local optimality, and first-order stationarity.

> **Status:** Writing backbone. Each section identifies its core statements, assumptions, proofs, and examples. The guarantee tables below are verified results for the specified formulations; full proofs and figures will be developed in the tutorial.

## 1. Problem setup and the role of scalarization

**Planned development.** Reuse the notation and definitions from the MOO introduction, without repeating its proofs. Let $X\subseteq\mathbb R^n$ be nonempty, let $[M]=\{1,\ldots,M\}$, and consider

$$
\min_{x\in X} F(x),\qquad F(x)=\bigl(f_1(x),\ldots,f_M(x)\bigr)^\top.
$$

Recall the simplex $\Delta_M=\{\lambda\in\mathbb R^M:\lambda_m\geq0,\ \sum_{m\in[M]}\lambda_m=1\}$. Introduce a *scalarization family* $\phi(x;\theta)$, where $\theta$ specifies weights, objective thresholds, or a reference point:

$$
\min_{x\in X}\ \phi(x;\theta).
$$

- **Central distinction:** A scalarized *global minimizer* can carry a Pareto-optimality guarantee. A numerical method that reaches only a stationary point of $\phi$ generally provides a local first-order guarantee.
- **Three guarantees:** (i) whether a global or local scalarized minimizer is weakly Pareto or Pareto optimal, (ii) whether scalar first-order stationarity implies Pareto stationarity, and (iii) whether varying $\theta$ recovers every (weakly) Pareto-optimal solution.
- **Terminology:** Briefly connect fixed preferences (*a priori*), parameter sweeps (*a posteriori*), and sequential preference updates (*interactive*). Keep the mathematical taxonomy central [1, 2].

## 2. Pareto guarantees of scalarization methods

**Core objective.** Establish the solution quality guaranteed by each scalarization *before* comparing algorithms. Reuse **Pareto optimality**, **weak Pareto optimality**, and **Pareto stationarity** from the MOO note, and use the same stationarity measure in the unconstrained differentiable setting:

$$
R(x):=\min_{\lambda\in\Delta_M}\left\|\sum_{m\in[M]}\lambda_m\nabla f_m(x)\right\|^2,
\qquad R(x)=0\iff x\text{ is Pareto stationary}.
$$

A scalarization has **three distinct guarantees**:

1. **Forward optimality:** the Pareto status of *every global minimizer* (or local minimizer) of a fixed scalarized problem.
2. **Recoverability:** whether *every target Pareto point* is a global minimizer for **some** parameter choice. This is an inverse statement and requires its own proof.
3. **First-order implication:** whether stationarity of the scalar objective implies $R(x)=0$. This is the typical result of a gradient-based optimizer on a nonconvex problem.

### 2.1 Monotonicity gives the forward guarantee

**Planned proposition (monotone scalarization).** Let $\phi(x;\theta)=h(F(x);\theta)$, with $\theta$ fixed. Suppose, whenever $u_m<v_m$ for every $m$, we have $h(u;\theta)<h(v;\theta)$. Then every **global minimizer** of $\phi$ is **weakly Pareto optimal**, and every **local minimizer** is **locally weakly Pareto optimal**. If, more strongly,

$$
u_m\leq v_m\ \text{for all }m,\quad u_j<v_j\ \text{for some }j
\quad\Longrightarrow\quad h(u;\theta)<h(v;\theta),
$$

then global and local minimizers are, respectively, **Pareto optimal** and **locally Pareto optimal**. Give a short contradiction proof. The $\varepsilon$-constraint method, which modifies the feasible set, is handled separately in Section 5. See [1, 2, 8].

### 2.2 From scalar stationarity to Pareto stationarity

**Planned proposition (gradient certificate).** Assume $X=\mathbb R^n$, each $f_m$ is differentiable, and $h$ is differentiable at $F(x)$. If

$$
\frac{\partial h}{\partial u_m}(F(x);\theta)\geq0\quad(m\in[M]),
\qquad\sum_{m\in[M]}\frac{\partial h}{\partial u_m}(F(x);\theta)>0,
$$

then $\nabla_x\phi(x;\theta)=0$ implies $R(x)=0$: normalize the nonnegative coefficients in

$$
0=\nabla_x\phi(x;\theta)
 =\sum_{m\in[M]}\frac{\partial h}{\partial u_m}(F(x);\theta)\nabla f_m(x).
$$

For the nonsmooth weighted maximum, the analogous statement uses **Clarke stationarity** $0\in\partial_C\phi(x)$ and strictly positive weights; its subgradients are convex combinations of active weighted objective gradients. With constraints, use feasible-direction or normal-cone/KKT stationarity under the corresponding constraint regularity, rather than the unconstrained identity $R(x)=0$.

### 2.3 Guarantee map for fixed parameters

The table uses **positive weights** to mean $\lambda_m>0$ for every $m$. Statements in the first three columns concern the original multi-objective problem, not just the scalar objective.

| Scalarization | Global minimizer | Local minimizer | Scalar stationarity, $X=\mathbb R^n$ |
|---|---|---|---|
| Weighted sum, $\lambda\in\Delta_M$ | **Weak Pareto**; **Pareto** if all weights are positive | Locally weak Pareto; locally Pareto if positive | $\nabla\phi=0\Rightarrow R(x)=0$ |
| Weighted Tchebycheff, positive weights | **Weak Pareto**; **Pareto** if the minimizer is unique | Locally weak Pareto | $0\in\partial_C\phi(x)\Rightarrow R(x)=0$ |
| Augmented Tchebycheff, positive weights and $\rho>0$ | **Pareto** | Locally Pareto | $0\in\partial_C\phi(x)\Rightarrow R(x)=0$ |
| $\varepsilon$-constraint | **Weak Pareto**; **Pareto** after lexicographic tie-breaking | Locally weak Pareto | A local minimizer satisfies $R(x)=0$ for differentiable objectives; KKT gives a certificate under constraint qualification |
| Augmented achievement/reference-point, positive weights and $\rho>0$ | **Pareto** | Locally Pareto | $0\in\partial_C\phi(x)\Rightarrow R(x)=0$ |
| Smooth Tchebycheff, positive weights and $\tau>0$ | **Pareto** | Locally Pareto | $\nabla\phi_\tau=0\Rightarrow R(x)=0$ |

**Technical distinctions.** For weighted Tchebycheff, the maximum is strictly improved when *all* objective values improve, but can remain unchanged under domination. The augmentation makes the scalar objective strictly increasing in every objective value. Smooth Tchebycheff is also strictly increasing for positive weights. For smooth Tchebycheff with nonnegative, nonzero weights, global and local minimizers retain **weak Pareto** guarantees, and scalar stationarity still implies $R(x)=0$ [2, 6].

### 2.4 Recoverability when preferences vary

| Scalarization family | Pareto-point coverage | Conditions and qualification |
|---|---|---|
| Weighted sum | **Supported points** | Under convexity and suitable closedness of $F(X)+\mathbb R_+^M$, every **weakly Pareto-optimal** point is obtainable with some nonnegative weight vector. Positive weights yield Pareto solutions but need not recover every Pareto point, even for convex problems. |
| Weighted Tchebycheff | **All weakly Pareto points**, hence all Pareto points | Given a utopian $z$ with $z_m<f_m(x^\star)$ for each $m$, choose $\lambda_m\propto[f_m(x^\star)-z_m]^{-1}$. Global scalarized optimality follows from weak Pareto optimality, with no convexity requirement. |
| Augmented Tchebycheff | **Pareto points with bounded trade-offs** | Positive $\rho$ ensures all global minimizers are Pareto optimal. For a target $x^\star$, inverse-gap weights plus the explicit bound in Section 4 recover it when $0<\rho\leq 1/B$; this includes suitable properly efficient solutions. |
| $\varepsilon$-constraint | **Every Pareto point** | For any Pareto-optimal $x^\star$, fix $\varepsilon_m=f_m(x^\star)$ for $m\ne j$. Then $x^\star$ globally minimizes objective $j$ over the restricted feasible set. |
| Augmented achievement/reference point | **Controlled-trade-off/properly efficient points** under suitable parameter choices | Reference-point variation and positive augmentation can recover properly efficient solutions with bounded trade-offs. State precise assumptions for the chosen achievement function. |
| Smooth Tchebycheff | **All Pareto points under front regularity and suitable smoothing** | Lin et al. [6, Theorem 3.8] assume a smooth Pareto-front representation with **positive normal components** and **bounded second derivatives**, and prove recovery for sufficiently small $\tau$. Their Corollary 3.9 gives recovery for any $\tau$ in its convex-front setting. |

**Terminology.** A **properly efficient** solution is Pareto optimal with suitably bounded objective trade-off ratios. This stronger notion is relevant to positive-weight supporting scalarizations and augmentation-based recovery; it is distinct from ordinary Pareto optimality.

**Planned proof sequence.** Prove the monotonicity proposition, the weighted Tchebycheff inverse-gap representation, and the $\varepsilon$-constraint representation. Use the finite three-point example to show why **forward guarantees and completeness are different statements**. For smooth Tchebycheff, state the cited theorem with its actual front-regularity assumptions instead of treating the $\tau\downarrow0$ approximation bound as an exact-recovery proof.

## 3. Linear scalarization: weighted sums

**Core formulation.** For a fixed $\lambda\in\Delta_M$,

$$
\min_{x\in X}\ \sum_{m\in[M]}\lambda_m f_m(x).
$$

**Planned components.**

1. **Global and local guarantees.** Reuse the MOO proposition: nonnegative weights give **weak Pareto** global solutions, and strictly positive weights give **Pareto** global solutions. The local analogues hold for local minimizers. If a global minimizer is unique, it is Pareto optimal even with zero weights. Refer to the companion proof.
2. **Geometric interpretation.** Explain how the level sets of a linear functional support the attainable objective set $F(X)$; relate supported solutions to supporting hyperplanes of $F(X)+\mathbb R_+^M$.
3. **Completeness under convexity.** Under convexity and suitable closedness of the upper image $F(X)+\mathbb R_+^M$, support every weakly Pareto point with a **nonnegative** weight vector. Explain that **strictly positive** weights recover supported Pareto points, including properly efficient points under the relevant conditions, but may not represent every Pareto point.
4. **Nonconvex fronts.** Reuse the three-point example from the MOO introduction briefly: $(0,2)^\top$, $(3/2,3/2)^\top$, and $(2,0)^\top$. The middle point is Pareto optimal but unsupported.

**Planned figure:** Weighted-sum level lines meeting convex and nonconvex objective fronts.

**Key reference:** Ehrgott [1], Miettinen [2].

## 4. Weighted Tchebycheff scalarization

**Motivation.** A nonlinear scalarization can recover trade-offs that weighted sums miss. Introduce a reference vector $z\in\mathbb R^M$, with each $z_m$ strictly below the attainable values of $f_m$, and strictly positive $\lambda\in\Delta_M$:

$$
\min_{x\in X}\ \max_{m\in[M]}\lambda_m\bigl(f_m(x)-z_m\bigr).
$$

**Planned components.**

1. **Ideal and utopian reference points.** Define $z_m^{\mathrm{ideal}}=\inf_{x\in X}f_m(x)$ and explain why a strictly lower (*utopian*) reference point simplifies the representation of arbitrary Pareto solutions.
2. **Geometric interpretation.** Compare the axis-aligned contours of a weighted maximum with the hyperplanes of weighted sums.
3. **Forward and stationary guarantees.** Prove that every global minimizer with positive weights is **weakly Pareto optimal**, and that a **unique** minimizer is Pareto optimal. If the objectives are differentiable and $X=\mathbb R^n$, a Clarke-stationary point of this weighted maximum is Pareto stationary.
4. **Complete recovery of weakly efficient points.** For any **weakly Pareto-optimal** $x^\star$, take $z_m<f_m(x^\star)$ and derive $\lambda_m\propto 1/(f_m(x^\star)-z_m)$. Show that $x^\star$ globally solves the resulting Tchebycheff subproblem. No convexity assumption is needed.
5. **Efficient solutions via augmentation.** Introduce the augmented formulation

$$
\min_{x\in X}\left\{\max_{m\in[M]}\lambda_m(f_m(x)-z_m)
+\rho\sum_{m\in[M]}\lambda_m(f_m(x)-z_m)\right\},\qquad \rho>0.
$$

Explain why positive weights and $\rho>0$ give **Pareto-optimal** global minimizers and **locally Pareto-optimal** local minimizers. For recovery, start with a Pareto-optimal $x^\star$ and the inverse-gap weights from item 4. A concrete sufficient **bounded-trade-off condition** is the existence of $B>0$ such that

$$
\sum_{m\in[M]}\lambda_m\bigl(f_m(x^\star)-f_m(y)\bigr)
\leq B\max_{m\in[M]}\lambda_m\bigl(f_m(y)-f_m(x^\star)\bigr)
\qquad\text{for every }y\in X.
$$

Then $0<\rho\leq 1/B$ makes $x^\star$ a global minimizer of the augmented formulation. The proof uses the fact that all weighted gaps are equal at $x^\star$: the increase in the maximum is at least the weighted-sum decrease multiplied by $1/B$. Relate this assumption to **proper efficiency** and introduce the epigraph formulation.

**Planned counterexample (weak versus Pareto).** Let $X=\{a,b\}$, $F(a)=(0,1)^\top$, $F(b)=(0,0)^\top$, $z=(-1,-1)^\top$, and $\lambda=(2/3,1/3)^\top$. Both points minimize the weighted maximum with value $2/3$, yet $b$ dominates $a$. The positive augmentation term selects $b$ for this example.

**Planned figure:** The same nonconvex objective front under weighted-sum and Tchebycheff contours.

**Key references:** Miettinen [2], Wierzbicki [3], Jahn [4].

## 5. The $\varepsilon$-constraint method

**Core formulation.** Choose one objective, indexed by $j\in[M]$, to minimize, and place upper bounds $\varepsilon_m$ on the others:

$$
\begin{aligned}
\min_{x\in X}\quad & f_j(x)\\
\text{subject to}\quad & f_m(x)\leq\varepsilon_m,\qquad m\in[M]\setminus\{j\}.
\end{aligned}
$$

**Planned components.**

1. **Forward guarantee and ties.** Show that every global optimizer is **weakly Pareto optimal** and every local optimizer is **locally weakly Pareto optimal**. Explain how lexicographic tie-breaking among globally optimal solutions selects a **Pareto-optimal** solution.
2. **Complete Pareto recovery.** For **any Pareto-optimal** $x^\star$, choose $\varepsilon_m=f_m(x^\star)$ for $m\ne j$ and prove $x^\star$ is a **global minimizer** for every choice of primary objective $j$. No convexity assumption is needed.
3. **Choice of thresholds.** Explain feasibility, the effect of loose or tight bounds, and how threshold sweeps generate front samples.
4. **First-order interpretation.** For differentiable objectives, a local optimizer in the unconstrained base problem is Pareto stationary: any common strict descent direction also decreases the primary objective while respecting all active threshold constraints for a short step. Use KKT multipliers for an explicit certificate when constraint qualifications hold.

**Planned example:** Apply the method to the same unsupported three-point front.

**Key references:** Ehrgott [1], Miettinen [2], Laumanns, Thiele, and Zitzler [5].

## 6. Reference-point and achievement scalarizing functions

**Motivation.** A weight vector specifies relative trade-off importance, whereas a reference vector describes desired objective levels. Extend Tchebycheff scalarization to an aspiration point $a\in\mathbb R^M$:

$$
\min_{x\in X}\left\{\max_{m\in[M]}\lambda_m\bigl(f_m(x)-a_m\bigr)
+\rho\sum_{m\in[M]}\lambda_m\bigl(f_m(x)-a_m\bigr)\right\}.
$$

**Planned components.**

1. **Aspirations and achievement.** Explain feasible and infeasible aspiration levels and how the method finds a trade-off relative to the supplied reference.
2. **Pareto guarantees.** Show that positive scaling coefficients and $\rho>0$ make the achievement scalarization strictly increasing in every objective value, so **global solutions are Pareto optimal** and **local solutions locally Pareto optimal**.
3. **Preference steering.** Show how changing $a$ moves the target region without fixing the solution in advance.
4. **Recoverability and trade-off bounds.** Relate the aspiration point to controlled-trade-off and properly efficient solutions. Explain how the augmentation size affects which Pareto solutions can be recovered, and separate its role from positive scaling coefficients.

**Planned example:** Two aspiration points on opposite sides of the same bi-objective front.

**Key references:** Wierzbicki [3], Miettinen [2].

## 7. Smooth and differentiable scalarizations

**Motivation.** The maximum in Tchebycheff scalarization is nonsmooth where several weighted deviations are active. Introduce log-sum-exp smoothing, with temperature $\tau>0$ and $u_m(x)=\lambda_m(f_m(x)-z_m)$:

$$
\phi_\tau(x)=\tau\log\sum_{m\in[M]}\exp\bigl(u_m(x)/\tau\bigr).
$$

**Planned components.**

1. **Approximation bound.** Derive $\max_m u_m(x)\leq\phi_\tau(x)\leq\max_m u_m(x)+\tau\log M$.
2. **Stationarity certificate.** Derive $\nabla\phi_\tau(x)=\sum_m p_m(x)\lambda_m\nabla f_m(x)$, where $p_m(x)=e^{u_m(x)/\tau}/\sum_j e^{u_j(x)/\tau}>0$. With positive weights, scalar stationarity implies **Pareto stationarity**.
3. **Forward Pareto guarantee.** The log-sum-exp is strictly increasing in each component for $\lambda_m>0$. Consequently **every global minimizer is Pareto optimal** and every local minimizer is locally Pareto optimal. Then explain the smoothing–conditioning trade-off and distinguish the method from MGDA.
4. **Exact recovery conditions.** State Lin et al. [6, Theorem 3.8] with its positive-normal and bounded-curvature front assumptions and sufficiently small $\tau$. Distinguish this result from the $\tau\log M$ **function approximation** bound, which alone does not establish exact recovery. Cover their convex-front Corollary 3.9.

**Planned example:** Plot hard-max and smooth-max contours for two objectives.

## 8. Gradient-based optimization of scalarized problems

**Central connection to the MGDA tutorial.** For a differentiable weighted sum,

$$
\nabla_x\!\left(\sum_{m\in[M]}\lambda_m f_m(x)\right)
=\sum_{m\in[M]}\lambda_m\nabla f_m(x).
$$

The same algebraic form appears in MGDA, but its **weights serve a different purpose**: scalarization holds $\lambda$ fixed to represent a chosen trade-off, while deterministic MGDA chooses $\lambda^\star(x)$ from the current gradients to obtain a minimum-norm common-descent direction.

**Planned components.**

1. Weighted-sum gradient descent and smooth-scalarization gradient descent.
2. Nonsmooth Tchebycheff: active objective indices, subgradients, and epigraph optimization.
3. **First-order stationarity.** Derive when $\nabla_x\phi(x;\theta)=0$ or $0\in\partial_C\phi(x;\theta)$ gives the same certificate $R(x)=0$ as in the MOO note. Include the threshold-constraint KKT case.
4. **Convexity and global recovery.** If $X$ is convex and $f_m$ are convex, the weighted sum, positive-weight Tchebycheff, augmented Tchebycheff, and log-sum-exp scalarizations are convex. First-order optimality then certifies **global scalarized optimality** under the relevant constrained or unconstrained optimality conditions. Translate that result through the forward guarantee table.

**Planned example:** Compare a fixed-weight descent direction with $d(x)$ from the MGDA note at the same point.

## 9. Pareto-front exploration and geometric alternatives

**Planned components.**

1. **Parameter sweeps.** Explain how repeated scalarized solves construct a finite approximation of $F(\mathcal P)$, where $\mathcal P$ is the Pareto set defined in the MOO note.
2. **Weight-to-front distortion.** Illustrate why uniformly spaced parameters generally do not yield uniformly spaced Pareto-front points.
3. **Normal-boundary intersection (NBI).** Introduce NBI as a related geometric front-generation method, distinct from scalar objective aggregation, and describe the construction and its limitations [7].
4. **Multiobjective learning.** Position preference-conditioned methods as extensions that generate several trade-offs rather than a single stationary point.

**Planned figure:** Nonuniform front coverage from uniform scalarization weights, alongside a reference-point or geometric construction.

## 10. Practical considerations and synthesis

**Planned components.**

- **Objective scaling:** Units, normalization, ideal and nadir estimates, and sensitivity of selected solutions to rescaling.
- **Preference interpretation:** Positive weights, relative trade-offs, aspiration levels, and explicit objective bounds.
- **Subproblem structure:** Smooth versus nonsmooth objectives, added constraints, and feasibility requirements.
- **Coverage:** The difference between finding one preferred solution and approximating an entire Pareto front.

**Planned synthesis.** Revisit the two guarantee maps in Section 2 and add **parameter meaning**, **solver requirements**, **objective scaling**, and **front coverage**. Retain three distinct columns for **global Pareto status**, **Pareto stationarity of local algorithms**, and **recoverability** rather than merging these claims.

## Planned propositions, examples, and figures

| Type | Item | Purpose |
|---|---|---|
| Proposition | General monotonicity criterion | Distinguish global Pareto from weak Pareto guarantees |
| Proposition | Scalar stationarity certificate | Normalize nonnegative objective-gradient coefficients to obtain $R(x)=0$ |
| Proposition | Weighted-sum Pareto guarantee | Reuse the proof from the MOO note |
| Proposition | Weighted-sum representation under convexity | Explain supported efficiency |
| Proposition | Tchebycheff weak efficiency and recovery | Prove recovery of **every weakly Pareto point** using inverse gaps |
| Example | Weak but non-Pareto Tchebycheff optimizer | Construct a dominated tied minimizer of the weighted maximum |
| Theorem summary | Smooth Tchebycheff completeness | List the front regularity and smoothing conditions in [6] |
| Proposition | Augmented Tchebycheff efficiency | Explain strict improvement under dominance |
| Proposition | $\varepsilon$-constraint recovery | Show representation of arbitrary Pareto-optimal points |
| Lemma | Log-sum-exp approximation bound | Prepare the smooth-scalarization discussion |
| Example | Bi-objective quadratics from the MOO note | Compare preference-to-solution mappings |
| Example | Three unsupported/supported objective vectors | Contrast weighted sums, Tchebycheff, and $\varepsilon$-constraints |
| Figure | Objective-space level sets | Show the geometry of major scalarizations |
| Figure | Weight-to-front mapping | Explain nonuniform coverage |

**Writing order:** Sections 1–5 establish the precise global, local, and first-order guarantees. Sections 6–8 cover reference-based and differentiable formulations. Sections 9–10 connect scalarization to Pareto-front exploration and practical selection. Proofs will state their hypotheses with the result and use the established notation $F(x)$, $[M]$, $\Delta_M$, and $R(x)$.

## References

[1] Matthias Ehrgott (2005). [Multicriteria Optimization, 2nd ed.](https://doi.org/10.1007/3-540-27659-9). Springer. ^ref-ehrgott-2005

[2] Kaisa Miettinen (1999). [Nonlinear Multiobjective Optimization](https://doi.org/10.1007/978-1-4615-5563-6). Kluwer Academic Publishers. ^ref-miettinen-1999

[3] Andrzej P. Wierzbicki (1980). [The use of reference objectives in multiobjective optimization](https://doi.org/10.1007/978-3-642-48782-8_32). In *Multiple Criteria Decision Making Theory and Application*, 468–486. ^ref-wierzbicki-1980

[4] Johannes Jahn (1984). [Scalarization in vector optimization](https://doi.org/10.1007/BF02592221). *Mathematical Programming*, **29**, 203–218. ^ref-jahn-1984

[5] Marco Laumanns, Lothar Thiele, and Eckart Zitzler (2006). [An efficient, adaptive parameter variation scheme for metaheuristics based on the epsilon-constraint method](https://doi.org/10.1016/j.ejor.2004.08.029). *European Journal of Operational Research*, **169**(3), 932–942. ^ref-laumanns-2006

[6] Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, and Qingfu Zhang (2024). [Smooth Tchebycheff Scalarization for Multi-Objective Optimization](https://proceedings.mlr.press/v235/lin24y.html). *ICML*, PMLR **235**, 30479–30509. ^ref-lin-2024

[7] Indraneel Das and John E. Dennis (1998). [Normal-boundary intersection: A new method for generating the Pareto surface in nonlinear multicriteria optimization problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, **8**(3), 631–657. ^ref-das-dennis-1998

[8] Stephan Helfrich, Arne Herzel, Stefan Ruzika, and Clemens Thielen (2024). [Using scalarizations for the approximation of multiobjective optimization problems: towards a general theory](https://doi.org/10.1007/s00186-023-00823-2). *Mathematical Methods of Operations Research*, **100**, 27–63. ^ref-helfrich-2024
