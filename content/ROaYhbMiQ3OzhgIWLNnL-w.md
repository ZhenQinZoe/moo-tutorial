---
title: "Introduction to MGDA: Gradient Descent for Multi-objective Optimization"
author: Lisha Chen
create date: 2025-05-01
update date: 2026-10-11
description: A short introduction to deterministic MGDA, its geometry, and the basic properties of its update direction.
tags:
  - Multi-objective-optimization
  - Pareto-optimality
  - MGDA
type: blog
status: draft
---

# Introduction to MGDA: Gradient Descent for Multi-objective Optimization

For one differentiable objective, the gradient tells us how to move downhill, which leads to gradient descent. With several objectives, their gradients may point in different directions. 
We will consider the unconstrained problem with $M$ objectives, using $[M]=\{1,\ldots,M\}$ for the objective index set:
$$
\min_{x\in\mathbb{R}^n} F(x),
\qquad
F(x)=\bigl(f_1(x),\ldots,f_M(x)\bigr)^\top,
$$
The difficulty is **gradient conflict**. For two objectives $f_1$ and $f_2$, if $\nabla f_1(x)^\top\nabla f_2(x)<0$, a sufficiently small step along $-\nabla f_1(x)$ decreases $f_1$ but increases $f_2$. Improving one objective can therefore undo progress on another. This motivates **conflict avoidance**: finding a direction that decreases all objectives together whenever a common descent direction exists, i.e.,

::: {.question-highlight style="background:#f6f8fa; border-left:4px solid #64748b; padding:0.7em 1em;"}

*Q: Can we find a steepest common descent direction that improves all objectives, and maximizes the minimum improvement amount?*

:::

The **Multiple-Gradient Descent Algorithm (MGDA)** answers this question using a small optimization problem. We can build it from the Taylor expansion, then rewrite it as a weighted combination of the gradients.

Early multicriterion methods worked directly with multiple objectives [Mukai (1980)](https://doi.org/10.1109/TAC.1980.1102298) [\[1\]](#^ref-mukai-1980). [Fliege and Svaiter (2000)](https://doi.org/10.1007/s001860000043) [\[2\]](#^ref-fliege-svaiter-2000) developed the regularized minimax steepest-descent direction and a componentwise Armijo rule. [Désidéri (2012)](https://doi.org/10.1016/j.crma.2012.03.014) [\[3\]](#^ref-desideri-2012) presented the minimum-norm geometry of MGDA and its common-descent property.

Assume each $f_m$ is differentiable. For the smoothness-based bounds and step-size guarantees, additionally assume each $f_m$ is **$L$-smooth**, with a common $L>0$: $\|\nabla f_m(x)-\nabla f_m(y)\|\leq L\|x-y\|$ for every $x,y\in\mathbb{R}^n$ and every $m\in[M]$. All objectives are minimized, and all gradients are exact. Write $g_m=\nabla f_m(x)$ when discussing a given point $x$. Throughout, $\|\cdot\|$ is the Euclidean norm.




## Pareto Stationarity

At a point $x$, a vector $v\in\mathbb{R}^n$ is a *common first-order descent direction* if $g_m^\top v<0$ for every $m\in[M]$. We call $x$ *Pareto stationary* (or *Pareto critical*) if no such direction exists.

Let $\Delta_M=\{\lambda\in\mathbb{R}^M:\lambda_m\geq0\text{ for all }m\in[M],\ \sum_{m\in[M]}\lambda_m=1\}$. Gordan's theorem of alternatives, the strict-inequality case of Motzkin's transposition theorem, says that exactly one of the following holds [Bertsekas (2009, Propositions 5.6.1–5.6.2)](https://faculty.engineering.asu.edu/bertsekas/wp-content/uploads/sites/129/2019/10/Convex_Theory_Entire_Book.pdf) [\[6\]](#^ref-bertsekas-2009):
$$
\begin{aligned}
\text{(D)}\quad &\exists v\in\mathbb{R}^n:\ g_m^\top v<0\quad\text{for every }m\in[M],\\
\text{(C)}\quad &\exists\lambda\in\Delta_M:\ \sum_{m\in[M]}\lambda_m g_m=0.
\end{aligned}
$$
The nonzero, nonnegative multipliers in the theorem can be normalized to sum to one. Since $\Delta_M$ is compact, the certificate (C) is equivalent to a zero minimum-norm value:
$$
x\text{ is Pareto stationary}
\quad\Longleftrightarrow\quad
\exists\lambda\in\Delta_M:\ \sum_{m\in[M]}\lambda_m g_m=0
\quad\Longleftrightarrow\quad
\min_{\lambda\in\Delta_M}\left\|\sum_{m\in[M]}\lambda_m g_m\right\|=0.
$$
This condition is necessary for local Pareto optimality but generally insufficient. For example, $F(x)=(-x^2,-2x^2)^\top$ is Pareto stationary at $x=0$, although every nonzero $x$ improves both objectives. For broader treatments of Pareto optimality and first-order conditions, see [Ehrgott (2005)](https://doi.org/10.1007/3-540-27659-9) [\[4\]](#^ref-ehrgott-2005) and [Miettinen (1999)](https://users.jyu.fi/~miettine/book/index.html) [\[5\]](#^ref-miettinen-1999).

## Update direction of MGDA

Starting from $x_0$, deterministic MGDA evaluates all objective gradients at $x_t$ and takes the step
$$
x_{t+1}=x_t+\alpha_t d_t,
\qquad d_t=d(x_t).
\tag{1}
$$
^eq-mgda-update

where $\alpha_t>0$ is the step size. The key is to choose $d_t$ so that all objectives decrease for a sufficiently small step, whenever such a direction exists. We first derive the direction from a Taylor expansion, then show how to compute it through a weighted combination of the objective gradients.

### Start from the Taylor expansion

Smoothness lets us bound the remainder in the Taylor expansion. For any direction $v$ and step $\alpha>0$,
$$
f_m(x+\alpha v)
\leq f_m(x)+\alpha g_m^\top v
+\frac{L\alpha^2}{2}\|v\|^2.
$$
The linear term predicts the change; the quadratic term accounts for curvature. To improve all objectives, we can minimize the largest of these upper bounds on their changes.

Taking the reference step $\alpha=1/L$ gives
$$
\max_{m\in[M]}\{f_m(x+v/L)-f_m(x)\}
\leq \frac1L\left\{\max_{m\in[M]} g_m^\top v+\frac12\|v\|^2\right\}.
$$
Minimizing the right-hand side gives the MGDA direction:
$$
\boxed{
d(x)=\operatorname*{argmin}_{v\in\mathbb{R}^n}
\left\{\max_{m\in[M]} g_m^\top v+\frac12\|v\|^2\right\}.
}
\tag{2}
$$
^eq-mgda

This is the steepest-descent formulation of [Fliege and Svaiter (2000, Section 3)](https://doi.org/10.1007/s001860000043) [\[2\]](#^ref-fliege-svaiter-2000). The coefficient $1/2$ comes from taking $\alpha=1/L$ in the smoothness bound. We keep this normalization of $d(x)$ and choose the actual step length later. The objective in [[#^eq-mgda|(2)]] is strongly convex, so the direction is unique. With one objective, it gives the familiar $d(x)=-\nabla f_1(x)$.

### Update direction is a weighted combination of the gradients

We can replace the maximum in [[#^eq-mgda|(2)]] by a scalar $s$:
$$
\min_{v,s}\ s+\frac12\|v\|^2
\qquad\text{subject to}\qquad
g_m^\top v\leq s,\quad m\in[M].
$$
Give each inequality a multiplier $\lambda_m\geq0$. The Lagrangian is
$$
\mathcal{L}(v,s,\lambda)
=\frac12\|v\|^2
+\left(\sum_{m\in[M]}\lambda_m g_m\right)^\top v
+s\left(1-\sum_{m\in[M]}\lambda_m\right).
$$
Minimizing over $s$ forces $\sum_{m\in[M]}\lambda_m=1$; otherwise the Lagrangian is unbounded below. Minimizing over $v$ then gives $v=-\sum_{m\in[M]}\lambda_m g_m$ and the dual objective $-\frac12\|\sum_{m\in[M]}\lambda_m g_m\|^2$.

The constraints are strictly feasible, for example at $v=0,s=1$, so strong duality makes this reformulation exact. Maximizing the negative squared norm is equivalent to minimizing the squared norm. Using the simplex $\Delta_M$ defined above, we obtain
$$
\lambda^\star\in
\operatorname*{argmin}_{\lambda\in\Delta_M}
\frac12\left\|\sum_{m\in[M]}\lambda_m g_m\right\|^2,
\qquad
d(x)=-\sum_{m\in[M]}\lambda_m^\star g_m.
\tag{3}
$$
^eq-weights

The weights are computed from the current gradients and can change at every iteration. They are not prescribed preferences over the objectives. Note that $\lambda^\star$ is not necessarily unique, but $d(x)$ is unique, which will be discussed below.

This also gives a geometric interpretation: $-d(x)$ is the point closest to the origin in
$$
C(x)=\operatorname{conv}\{g_1,\ldots,g_M\},
$$
the convex hull of the gradients. MGDA moves in the opposite direction.

### A bi-objective example

Suppose $g_1=(1,0)^\top$ and $g_2=(-2,1)^\top$. Simply taking their negative average gives $(1/2,-1/2)^\top$, whose inner product with $g_1$ is positive. That direction locally increases the first objective.

Instead, [[#^eq-weights|(3)]] chooses weights $(7/10,3/10)$, giving
$$
d=(-1/10,-3/10)^\top.
$$
Now $g_1^\top d=g_2^\top d=-1/10$: both objectives decrease to first order.

![Two gradients, their convex hull, and the MGDA direction. The minimum-norm point is -d=(0.1,0.3), and the direction is d=(-0.1,-0.3).](assets/mgda-direction.png)

*The segment contains all convex combinations of the two gradients. MGDA projects the origin onto this segment and uses its opposite as the update direction.*

## Properties of the MGDA direction

The following proposition collects the basic properties of the MGDA direction. The common-descent inequality [[#^eq-common-descent|(4)]] is the central geometric inequality in [Désidéri (2012, Lemma 2.1 and Theorem 2.2)](https://www.numdam.org/item/CRMATH_2012__350_5-6_313_0.pdf) [\[3\]](#^ref-desideri-2012).

::: {.proposition #prop-mgda-direction}

**Proposition 1 (Properties of the MGDA direction).**

Fix $x\in\mathbb{R}^n$ and write $g_m=\nabla f_m(x)$ and $d=d(x)$. Let $\lambda^\star\in\Delta_M$ be any solution of [[#^eq-weights|(3)]]. Properties (i), (ii), (iv), and (v) require only differentiability; property (iii) additionally requires the common $L$-smoothness assumption above.

**(i) Uniqueness and norm bound.** The direction $d$ is unique, even if $\lambda^\star$ is not, and
$$
\|d\|=\min_{z\in C(x)}\|z\| \leq\min_{m\in[M]}\|g_m\|.
$$
**(ii) Common descent.** Every objective satisfies
$$
g_m^\top d\leq-\|d\|^2 \quad\text{for every }m\in[M]. \tag{4}
$$
^eq-common-descent

**(iii) Decrease with a suitable step.** For every $\alpha>0$ and $m\in[M]$,
$$
f_m(x+\alpha d) \leq f_m(x)-\alpha\left(1-\frac{L\alpha}{2}\right)\|d\|^2.
$$
Consequently, if $d\neq0$, every step $0<\alpha<2/L$ strictly decreases all objectives. Convexity of the objectives is unnecessary.

**(iv) Pareto stationarity.** The direction vanishes exactly at Pareto-stationary points: $d=0$ if and only if $x$ is Pareto stationary.

**(v) Active weights and optimal value.** Objectives with positive optimal weights have equal directional derivatives:
$$
\lambda_m^\star>0\ \Longrightarrow\ g_m^\top d=-\|d\|^2, \qquad \max_{m\in[M]}g_m^\top d=-\|d\|^2. \tag{5}
$$
^eq-active-weights

The minimum value of the direction subproblem [[#^eq-mgda|(2)]] is $-\frac12\|d\|^2$.

:::

::: {.proof #proof-proposition-1}

**Proof of Proposition 1.**

**(i)** By [[#^eq-weights|(3)]], $-d$ is the projection of the origin onto the nonempty, compact convex set $C(x)$. This projection is unique, so every optimal weight vector gives the same direction $d$. The weights can still be nonunique, for example when gradients are repeated. Since each $g_m\in C(x)$, the projection also gives $\|d\|=\min_{z\in C(x)}\|z\|\leq\min_{m\in[M]}\|g_m\|$.

**(ii)** For any $z\in C(x)$, the segment $-d+t(z+d)$ lies in $C(x)$ for $t\in[0,1]$. Since $-d$ minimizes the squared norm on $C(x)$, the right derivative of $\frac12\|-d+t(z+d)\|^2$ at $t=0$ is nonnegative:
$$
(-d)^\top(z+d)\geq0.
$$
Taking $z=g_m$ and rearranging gives [[#^eq-common-descent|(4)]].

**(iii)** By $L$-smoothness, $f_m(x+\alpha d)\leq f_m(x)+\alpha g_m^\top d+\frac{L\alpha^2}{2}\|d\|^2$. Substituting (ii) yields the stated bound. When $d\neq0$ and $0<\alpha<2/L$, the decrease is strictly positive.

**(iv)** By [[#^eq-weights|(3)]], $-d$ is the minimum-norm convex combination of the gradients. Thus $d=0$ if and only if the minimum-norm value in the stationarity characterization is zero, which is equivalent to Pareto stationarity.

**(v)** Averaging (ii) with the optimal weights gives
$$
\sum_{m\in[M]}\lambda_m^\star g_m^\top d =(-d)^\top d=-\|d\|^2.
$$
Each directional derivative is at most $-\|d\|^2$, so every term with positive weight must attain equality. Since the weights sum to one, at least one is positive, which also gives the maximum in [[#^eq-active-weights|(5)]]. Evaluating the objective in [[#^eq-mgda|(2)]] at $d$ then gives $-\|d\|^2+\frac12\|d\|^2=-\frac12\|d\|^2$.

<div class="qed" style="text-align: right;">&#9633;</div>

:::

By (i), the direction norm is a natural stationarity measure. A small value certifies an almost-vanishing convex combination of gradients; it does not by itself certify a small distance to a Pareto optimum. Zero weight does not mean that an objective is ignored: inequality [[#^eq-common-descent|(4)]] still applies to it.

## The deterministic algorithm

With the direction available, we still need to choose the step size in the MGDA update [[#^eq-mgda-update|(1)]].

A common choice is componentwise Armijo backtracking, following [Fliege and Svaiter (2000, Sections 4–5)](https://doi.org/10.1007/s001860000043) [\[2\]](#^ref-fliege-svaiter-2000):

1. At $x_t$, compute every $\nabla f_m(x_t)$ and solve [[#^eq-weights|(3)]].
2. Stop if $\|d(x_t)\|\leq\varepsilon$, for a chosen tolerance $\varepsilon\geq0$.
3. Start with $\alpha=1$. Repeatedly replace $\alpha$ by $\rho\alpha$, with $\rho\in(0,1)$, until
   $f_m(x_t+\alpha d_t)\leq f_m(x_t)+c\alpha\nabla f_m(x_t)^\top d_t\quad\text{for all }m\in[M],$
   where $d_t=d(x_t)$ and $c\in(0,1)$.
4. Set $x_{t+1}=x_t+\alpha d_t$ and repeat.

The smoothness bound and [[#^eq-common-descent|(4)]] ensure that the Armijo condition holds whenever $0<\alpha\leq2(1-c)/L$. Backtracking therefore terminates, and every accepted step strictly decreases all objectives when $d_t\neq0$.

This version uses the unnormalized gradients and exact direction subproblems. Désidéri [\[3\]](#^ref-desideri-2012) also discusses gradient scaling and uses a different step-selection rule. Scaling matters: changing the units of an objective can change the MGDA direction.

The useful distinction is between finding a direction and choosing how far to move. The minimum-norm problem supplies common descent whenever the current point is not Pareto stationary; the line search turns that local property into an actual decrease.

<span style="color:red">[Kevin: 1) We may need to start with the basics of MOO first, e.g., the notions of dominance, Pareto optimality, weak Pareto optimality, and then Pareto stationarity; 2) basic MOO solution philosophies, e.g., no-preference methods, a priori methods, a posteri methods, iterative methods, and iterative methods; 3) basic scalarization approaches (e.g., linear scalarization, weighted Chebyshev, $\epsilon$-constrained, and their pros and cons); 4) stochastic MGDA, other gradient-based MOO methods, e.g., PCGrad, CAGrad? ]</span>


## References

[1] H. Mukai (1980). [Algorithms for multicriterion optimization](https://doi.org/10.1109/TAC.1980.1102298). *IEEE Transactions on Automatic Control*, **25**(2), 177–186. ^ref-mukai-1980

[2] Jörg Fliege and Benar Fux Svaiter (2000). [Steepest descent methods for multicriteria optimization](https://doi.org/10.1007/s001860000043). *Mathematical Methods of Operations Research*, **51**(3), 479–494. ^ref-fliege-svaiter-2000

[3] Jean-Antoine Désidéri (2012). [Multiple-gradient descent algorithm (MGDA) for multiobjective optimization](https://doi.org/10.1016/j.crma.2012.03.014). *Comptes Rendus Mathématique*, **350**(5–6), 313–318. ^ref-desideri-2012

[4] Matthias Ehrgott (2005). [Multicriteria Optimization, 2nd ed.](https://doi.org/10.1007/3-540-27659-9). Springer. ^ref-ehrgott-2005

[5] Kaisa Miettinen (1999). [Nonlinear Multiobjective Optimization](https://users.jyu.fi/~miettine/book/index.html). Kluwer Academic Publishers. ^ref-miettinen-1999

[6] Dimitri P. Bertsekas (2009). [Convex Optimization Theory](https://faculty.engineering.asu.edu/bertsekas/wp-content/uploads/sites/129/2019/10/Convex_Theory_Entire_Book.pdf). Athena Scientific, Section 5.6. ^ref-bertsekas-2009
