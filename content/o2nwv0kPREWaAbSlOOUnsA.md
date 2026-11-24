# Chapter 2: MOO Efficiency and Scalability

[TOC]

Recall from Chapter 1 that we consider
$$\min_{\mathbf{x}\in\mathbb{R}^d}\mathbf{F}(\mathbf{x}) := [f_1(\mathbf{x}),\ldots,f_S(\mathbf{x})]^\top .$$
Most gradient-based MOO methods follow the same recipe: at each iteration, compute the gradient of every objective, combine the gradients into one direction that decreases all objectives, and take a step along it. The prototype is MGDA (Fliege & Svaiter, 2000; Désidéri, 2012), which solves
$$\mathbf{w}^* = \arg\min_{\mathbf{w}\in\Delta_S}\Big\|\sum_{s=1}^S w_s \nabla f_s(\mathbf{x})\Big\|^2$$
over the probability simplex $\Delta_S$ and updates along $\mathbf{d} = -\sum_{s} w_s^* \nabla f_s(\mathbf{x})$. Later methods keep this template and change only the combination rule: they project away conflicting components (Yu et al., 2020), maximize the worst-case improvement near the average gradient (Liu et al., 2021), solve a bargaining game (Navon et al., 2022), steer toward a preferred direction (Xiao et al., 2023), or enforce fairness across objectives (Ban & Ji, 2024).

The recipe handles conflicts and converges to Pareto-stationary points, but it is expensive in two ways.

- **Per-iteration cost.** Each iteration needs $S$ backward passes instead of one, $O(Sd)$ memory to store the gradients, and $O(S^2 d)$ work to form their pairwise inner products. The cost is therefore $O(S)$ per iteration, compared with $O(1)$ for single-objective training.
- **Sample cost.** In the stochastic setting, the weights are computed from the same noisy gradients to which they are applied, so plugging minibatch gradients into MGDA yields a *biased* update. Early analyses controlled this bias with growing batch sizes (Liu & Vicente, 2024). Bias-corrected methods based on momentum or slowly moving weights (Zhou et al., 2022; Fernando et al., 2023; Xiao et al., 2023) remove this requirement but can still need more samples than their single-objective counterparts.

These overheads are negligible for a few objectives and a small model, but not for modern workloads such as multi-task models with many tasks, multi-objective alignment of large language models (Ramé et al., 2023; Zhou et al., 2024b), and industrial-scale ranking and recommendation (Zhou et al., 2023; Dong et al., 2026). With billions of parameters, $S$ backward passes or $S$ stored gradient copies per step are impractical, and every extra sample is costly. The cheapest fallback is a weighted sum $\sum_s w_s f_s(\mathbf{x})$ with fixed, heuristic (Kendall et al., 2018; Chen et al., 2018; Liu et al., 2019), or random (Lin et al., 2022) weights. It needs a single backward pass but, as discussed in Chapter 0, gives up the principled treatment of conflicts and any guarantee that all objectives improve.

:::danger
**Central question.** Can we keep the benefits of multi-objective methods (handling conflicts, balancing objectives, and converging to Pareto-stationary points) while matching the sample efficiency of single-objective optimization and reducing the per-iteration cost from $O(S)$ to $O(1)$?
:::

<!-- TODO: The intro refers to both "Chapter 1" (problem setup) and "Chapter 0" (weighted-sum limitations). Please make sure both pointers are correct. -->

### Roadmap

**Section 2.1** introduces a simple cost model and identifies where the $O(S)$ cost comes from. **Sections 2.2–2.4** reduce this cost in three complementary ways, and **Section 2.5** compares them and discusses open directions.


![Efficiency Strategies Tree Diagram](https://hackmd.io/_uploads/S1Iu7tUTfl.png)

<p align="center">
  <strong>Figure 1:</strong> Taxonomy of methods for improving MOO efficiency and scalability.
</p>

| Section | Strategy | Key idea | Representative methods |
|---|---|---|---|
| 2.2 | **Cheaper** gradients | Compute objective-wise gradients only at a shared representation, so the backbone is traversed once | MGDA-UB (Sener & Koltun, 2018) |
| 2.3 | **Fewer** gradients | Evaluate a random subset of objectives per step, or recompute the weights only periodically | SDMGrad-OS (Xiao et al., 2023), PSMGD (Xu et al., 2025) |
| 2.4 | **No** per-objective gradients | Set or update the weights from loss values; take one backward pass on the weighted loss | STCH (Lin et al., 2024), FAMO (Liu et al., 2023), MGDA-FA (Zhang et al., 2025), LDC-MTL (Xiao et al., 2025) |

For each strategy, we explain the key idea, present representative algorithms, and weigh the efficiency gained against the guarantees given up.

<!-- TODO: The original outline also promised variance-reduced estimators such as STIMULUS (Liu et al., 2025) in Section 2.3. Either add a short part on them or drop STIMULUS from the reference list. -->

<div style="background-color:#f2f2f2; border-left:5px solid #9e9e9e; padding:12px 16px; border-radius:4px; margin-bottom:16px;">

**Notation.** $S$ is the number of objectives and $d$ the number of model parameters.
$$G(\mathbf{x}) = [\nabla f_1(\mathbf{x}),\ldots,\nabla f_S(\mathbf{x})]\in\mathbb{R}^{d\times S}$$
is the gradient matrix, and $\mathbf{w}\in\Delta_S$ denotes aggregation weights, so that $G(\mathbf{x})\mathbf{w}=\sum_s w_s\nabla f_s(\mathbf{x})$. Stochastic gradients on a minibatch $\xi$ are written $\nabla f_s(\mathbf{x};\xi)$. The model step size is $\eta$; when the weights are updated by gradient steps, their step size is $\beta$.

</div>

---

## 2.1 Where Does the Cost Come From?

To compare methods on an equal footing, we use **linear scalarization** (LS), i.e., minimizing $\sum_{s} w_s f_s(\mathbf{x})$ with fixed weights, as the $O(1)$ baseline. Its per-iteration cost equals that of single-objective training.

### 2.1.1 A simple cost model

One iteration of a gradient-based MOO method has three cost components.

1. **Gradients.** Let $C_{\mathrm{bp}}$ be the cost of one backward pass, typically a small multiple of the forward-pass cost. Computing all objective-wise gradients takes $S$ backward passes, i.e., $S\cdot C_{\mathrm{bp}}$. This holds even in multi-task learning, where the objectives share one forward pass: each gradient with respect to the shared parameters is a separate vector–Jacobian product. LS needs a single backward pass on $\sum_s w_s f_s(\mathbf{x})$.
2. **Memory.** The $S$ gradients must be held simultaneously before they are combined, which takes $O(Sd)$ memory on top of the model. The stored activations usually must also stay alive across the $S$ backward passes. For large models, this alone can exceed GPU memory.
3. **Aggregation.** Most combination rules depend on the gradients only through the Gram matrix $G^\top G\in\mathbb{R}^{S\times S}$, which costs $O(S^2 d)$ to form. Solving the resulting $S$-dimensional subproblem costs $C_{\mathrm{sub}}(S)$, independent of $d$, and forming the direction $\mathbf{d}=-G\mathbf{w}$ costs $O(Sd)$.

:::info
**Per-iteration cost of standard gradient-based MOO**
$$\underbrace{S\cdot C_{\mathrm{bp}}}_{\text{gradients}} \;+\; \underbrace{O(S^2 d)}_{\text{Gram matrix}} \;+\; \underbrace{C_{\mathrm{sub}}(S)}_{\text{subproblem}}\quad\text{time},\qquad O(Sd)\ \text{extra memory}.$$
For comparison, LS costs $C_{\mathrm{bp}} + O(d)$ time and $O(d)$ extra memory.
:::

For deep networks, $C_{\mathrm{bp}}$ far exceeds the cost of $d$-dimensional vector operations, so the first term dominates and the cost grows **linearly in $S$**; this is what we mean by "$O(S)$ per iteration." With many objectives, such as the 40 attribute tasks of CelebA, the $O(S^2 d)$ Gram-matrix term also becomes noticeable.

### 2.1.2 Applying the model to representative methods

- **MGDA** (Désidéri, 2012; Sener & Koltun, 2018) solves the min-norm problem
  $$\mathbf{w}^* = \arg\min_{\mathbf{w}\in\Delta_S}\|G\mathbf{w}\|^2 = \arg\min_{\mathbf{w}\in\Delta_S}\mathbf{w}^\top (G^\top G)\mathbf{w},$$
  a quadratic program over the simplex, usually solved by a few Frank–Wolfe iterations on $G^\top G$.
- **PCGrad** (Yu et al., 2020) visits the tasks in random order and projects each gradient onto the normal plane of every gradient it conflicts with (negative inner product), then sums the results. This takes up to $O(S^2)$ projections of cost $O(d)$ each.
- **CAGrad** (Liu et al., 2021) seeks a direction near the average gradient $\mathbf{g}_0=\frac1S\sum_s\nabla f_s(\mathbf{x})$ that maximizes the worst-case local improvement. Its dual is an $S$-dimensional problem that depends on $G$ only through $G^\top G$ and $G^\top\mathbf{g}_0$.
- **Nash-MTL** (Navon et al., 2022) treats the objectives as bargaining players. Its weights $\boldsymbol{\alpha}\in\mathbb{R}^S_{>0}$ satisfy $G^\top G\,\boldsymbol{\alpha} = 1/\boldsymbol{\alpha}$ (elementwise), found by solving a sequence of $S$-dimensional convex problems; its $C_{\mathrm{sub}}(S)$ is typically the largest of the four.

### 2.1.3 Where the bottleneck lies

| Method | Backward passes | Extra memory | Aggregation time |
|---|---|---|---|
| Linear scalarization | $1$ | $O(d)$ | $O(d)$ |
| MGDA | $S$ | $O(Sd)$ | $O(S^2 d) + C_{\mathrm{sub}}(S)$ |
| PCGrad | $S$ | $O(Sd)$ | $O(S^2 d)$ |
| CAGrad | $S$ | $O(Sd)$ | $O(S^2 d) + C_{\mathrm{sub}}(S)$ |
| Nash-MTL | $S$ | $O(Sd)$ | $O(S^2 d) + C_{\mathrm{sub}}(S)$ |

The four methods differ mainly in the aggregation rule, which affects only lower-order terms. They share the same bottleneck: **all $S$ objective-wise gradients at every iteration**. Reaching $O(1)$ cost requires breaking the dominant term $S\cdot C_{\mathrm{bp}}$ in one of three ways:

- **Cheaper:** make each objective-wise gradient cost less than $C_{\mathrm{bp}}$, e.g., by computing it only at a shared representation (Section 2.2);
- **Fewer:** compute fewer than $S$ gradients per iteration on average, e.g., by sampling objectives or reusing weights (Section 2.3);
- **None:** compute no objective-wise gradients, e.g., by updating the weights from loss values (Section 2.4).

---

## 2.2 Cheaper Gradients: Shared-Representation Approximation

The cost model charges a full backward pass for every objective-wise gradient. In a multi-task network, however, most of that pass is spent in the backbone shared by all objectives. If the gradients are compared *before* they enter the backbone, each objective needs a backward pass only through its own (usually small) head, and the backbone is traversed once. This is the idea of MGDA-UB (Sener & Koltun, 2018), the first method to make MGDA practical for deep multi-task networks.

:::warning
**Key idea.** Compare the objectives' gradients at the encoder's *output* $Z$, which requires backpropagating only through the task heads, then backpropagate **once** through the encoder with the combined gradient.
:::

### 2.2.1 Setup: hard parameter sharing

A shared encoder with parameters $\theta^{\mathrm{sh}}\in\mathbb{R}^{d_{\mathrm{sh}}}$ maps a minibatch of $N$ inputs to a representation $Z\in\mathbb{R}^{N\times q}$, and each objective has a head with parameters $\theta^s$, so that $f_s = \ell_s\big(h_s(Z;\theta^s)\big)$. Since each head is trained by its own objective only, conflicts arise only in $\theta^{\mathrm{sh}}$, and MGDA needs only the shared-parameter gradients. By the chain rule,
$$\nabla_{\theta^{\mathrm{sh}}} f_s = \mathbf{J}^\top \nabla_Z f_s, \qquad \mathbf{J} := \frac{\partial Z}{\partial \theta^{\mathrm{sh}}} \in \mathbb{R}^{m\times d_{\mathrm{sh}}},\quad m := Nq,$$
with $\nabla_Z f_s$ flattened into $\mathbb{R}^m$. The Jacobian $\mathbf{J}$ is common to all objectives; the $S$ expensive backward passes of MGDA multiply by the same $\mathbf{J}^\top$ once per objective.

### 2.2.2 MGDA-UB

MGDA-UB replaces the MGDA subproblem with a cheaper surrogate, obtained in two steps.

**Step 1: Bound the MGDA objective.** Every shared-parameter gradient factors through the same Jacobian, so
$$\Big\|\sum_{s} w_s \nabla_{\theta^{\mathrm{sh}}} f_s\Big\|^2 = \Big\|\mathbf{J}^\top \sum_{s} w_s \nabla_Z f_s\Big\|^2 \;\le\; \|\mathbf{J}\|^2\,\Big\|\sum_{s} w_s \nabla_Z f_s\Big\|^2 .$$

**Step 2: Minimize the bound.** The factor $\|\mathbf{J}\|^2$ does not depend on $\mathbf{w}$, so it can be dropped:
$$\mathbf{w}^{\mathrm{UB}} \in \arg\min_{\mathbf{w}\in\Delta_S}\Big\|\sum_{s=1}^S w_s \nabla_Z f_s\Big\|^2 .$$

This is the MGDA min-norm problem, moved from the parameter space $\mathbb{R}^{d_{\mathrm{sh}}}$ to the representation space $\mathbb{R}^m$. It needs only the representation gradients $\nabla_Z f_s$, each of which costs a backward pass through one head rather than through the whole network.

:::info
**Algorithm 1: MGDA-UB (one iteration)**

1. **Forward.** Run one forward pass through the encoder and all heads.
2. **Head gradients.** For each $s$, backpropagate $f_s$ through head $s$ only, obtaining $\nabla_Z f_s$ and $\nabla_{\theta^s} f_s$.
3. **Weights.** Solve the min-norm problem on $\{\nabla_Z f_s\}_{s=1}^S$ (e.g., by Frank–Wolfe) to obtain $\mathbf{w}^{\mathrm{UB}}$.
4. **Encoder update.** Backpropagate the weighted loss $\sum_s w^{\mathrm{UB}}_s f_s$ through the encoder **once**. This yields $\mathbf{J}^\top\mathbf{v}$ with $\mathbf{v} := \sum_s w^{\mathrm{UB}}_s\nabla_Z f_s$, and $\theta^{\mathrm{sh}}$ is updated along $-\mathbf{J}^\top\mathbf{v}$.
5. **Head updates.** Update each $\theta^s$ along its own gradient $-\nabla_{\theta^s} f_s$.
:::

**Two objectives.** For $S=2$, step 3 has a closed form. With $\mathbf{g}_s = \nabla_Z f_s$,
$$w_1 = \mathrm{clip}_{[0,1]}\!\left(\frac{\langle \mathbf{g}_2-\mathbf{g}_1,\,\mathbf{g}_2\rangle}{\|\mathbf{g}_1-\mathbf{g}_2\|^2}\right),\qquad w_2 = 1-w_1 .$$

### 2.2.3 The guarantee and its scope

:::success
**Theorem (Sener & Koltun, 2018, Theorem 1).** Assume $\mathbf{J}$ has full rank. Then either
- (a) $\sum_s w^{\mathrm{UB}}_s\nabla_{\theta^{\mathrm{sh}}} f_s = 0$ and the current parameters are Pareto stationary, or
- (b) $-\sum_s w^{\mathrm{UB}}_s\nabla_{\theta^{\mathrm{sh}}} f_s$ is a descent direction that decreases all objectives.
:::

Part (a) follows from the chain rule: when $\mathbf{J}^\top$ is injective, $\mathbf{J}^\top\mathbf{v}=0$ exactly when $\mathbf{v}=0$.


**A caveat on part (b).** The min-norm optimality conditions in the representation space give $\langle\nabla_Z f_s,\mathbf{v}\rangle \ge \|\mathbf{v}\|^2$ for every $s$, as in MGDA. The directional derivative along the actual parameter update, however, is
$$\big\langle \nabla_{\theta^{\mathrm{sh}}} f_s,\, \mathbf{J}^\top\mathbf{v}\big\rangle = \nabla_Z f_s^\top \big(\mathbf{J}\mathbf{J}^\top\big)\mathbf{v}.$$
Common descent therefore transfers exactly when $\mathbf{J}\mathbf{J}^\top$ is a multiple of the identity and approximately when $\mathbf{J}$ is well conditioned. In other words, MGDA-UB solves the min-norm problem in the Euclidean geometry of the representation, while MGDA effectively solves it in the geometry induced by $\mathbf{J}\mathbf{J}^\top$; the two can give different weights. The theorem is best read as a statement about the bound being minimized, not as an equivalence with MGDA.

### 2.2.4 Cost

Let $C_{\mathrm{enc}}$ and $C_{\mathrm{head}}$ be the backward costs of the encoder and of one head, so that $C_{\mathrm{bp}}\approx C_{\mathrm{enc}} + C_{\mathrm{head}}$.

| Method | Backward cost | Extra memory | Gram matrix |
|---|---|---|---|
| Linear scalarization | $C_{\mathrm{enc}} + S\,C_{\mathrm{head}}$ | $O(d)$ | — |
| MGDA | $S\,(C_{\mathrm{enc}} + C_{\mathrm{head}})$ | $O(S d_{\mathrm{sh}})$ | $O(S^2 d_{\mathrm{sh}})$ |
| MGDA-UB | $C_{\mathrm{enc}} + S\,C_{\mathrm{head}}$ | $O(S m)$ | $O(S^2 m)$ |

When the heads are light ($C_{\mathrm{head}}\ll C_{\mathrm{enc}}$) and the representation is small ($m\ll d_{\mathrm{sh}}$), MGDA-UB costs about as much as LS. Sener & Koltun (2018) report that the training time of MGDA grows roughly linearly with the number of tasks, whereas that of MGDA-UB stays nearly flat, making MGDA-style weighting usable on the 40 tasks of CelebA.

### 2.2.5 Other representation-level methods

Nothing above is specific to the min-norm rule: any aggregation rule can be applied to $\{\nabla_Z f_s\}$ instead of $\{\nabla_{\theta^{\mathrm{sh}}} f_s\}$ with the same cost profile. Several methods work this way or offer such a variant.

- **IMTL-G** (Liu et al., 2021b) computes closed-form "impartial" weights, which give the aggregated gradient equal projections onto every task's unit gradient, from gradients with respect to the shared feature.
- **GradDrop** (Chen et al., 2020) resolves sign conflicts by stochastically masking gradient components at a chosen activation layer, typically the last shared one.
- **RotoGrad** (Javaloy & Valera, 2022) homogenizes task gradients at the shared representation, in magnitude and direction, by learning task-specific rotations of the representation.
- **Aligned-MTL-UB** (Senushkin et al., 2023) applies the same upper-bound argument to Aligned-MTL, which aligns the principal components of the gradient matrix to remove conflicts and dominance.
- **Library support.** LibMTL (Lin & Zhang, 2023) exposes representation-level gradients as a generic option, so MGDA, PCGrad, GradVac, CAGrad, Nash-MTL, and other aggregators can all run on $\nabla_Z f_s$.

<!-- TODO (Peiyao): please double-check the IMTL-G and GradDrop descriptions against the papers, and add any 2024–2026 methods that use representation-level gradients. -->

### 2.2.6 Limitations

The approach needs no change to the aggregation rule and no new hyperparameters, but it has three limitations.

:::danger
**Limitations.**
- **Architecture dependence.** It requires a shared representation followed by task-specific heads. When objectives share the whole network, as in multi-objective LLM alignment where several rewards are computed from the same output, there is no cheap head and the savings vanish. With large heads (e.g., dense-prediction decoders), the $S\,C_{\mathrm{head}}$ term is no longer negligible.
- **Approximate weights.** The weights are computed in a different geometry, and conflicts that arise inside the encoder are invisible at $Z$. The approximation often works well in practice, but there is no general bound on the gap between $\mathbf{w}^{\mathrm{UB}}$ and $\mathbf{w}^*$.
- **Per-iteration savings only.** The Gram matrix still costs $O(S^2 m)$, with $m=Nq$ growing with the batch size. The weights are still computed from the same minibatch to which they are applied, so the bias of stochastic MGDA, and hence the sample cost, is unchanged.
:::

:::info
**Takeaway.** Section 2.2 keeps $S$ objective-wise gradients per iteration but makes each one cheap, and works best with a large shared backbone and light heads. Sections 2.3 and 2.4 instead reduce the *number* of objective-wise gradients and do not rely on a particular architecture.
:::

---

## 2.3 Fewer Gradients: Objective Sampling and Periodic Weight Updates

Instead of making each objective-wise gradient cheaper, we can compute fewer of them.

:::warning
**Key idea.** Not every objective's gradient is needed at every step.
- **Objective sampling** (Section 2.3.1) evaluates only $n<S$ objectives per step and rescales so that every estimate stays unbiased. It reduces *how many* gradients are computed per step.
- **Periodic weight updates** (Section 2.3.2) refresh the weights every $R$ steps and take cheap weighted-loss steps in between. It reduces *how often* all $S$ gradients are needed.
:::

### 2.3.1 Objective sampling: SDMGrad-OS

Minibatching removes the dependence of the per-iteration cost on the dataset size, but not on $S$. A natural next step is to sample objectives as well. The difficulty is that the weights come from a subproblem that is quadratic in the gradients, so naively dropping objectives biases them. SDMGrad-OS (Xiao et al., 2023), the objective-sampling variant of SDMGrad, avoids this bias.

#### Background: direction-oriented weighting

SDMGrad chooses the weights by solving
$$
\mathbf{w}_\lambda^*(\mathbf{x})\in\arg\min_{\mathbf{w}\in\Delta_S}
\frac12\big\|G(\mathbf{x})\mathbf{w}+\lambda\mathbf{g}_0(\mathbf{x})\big\|^2,
\qquad
\mathbf{g}_0(\mathbf{x})=G(\mathbf{x})\widetilde{\mathbf{w}},
$$
where $\widetilde{\mathbf{w}}\in\Delta_S$ defines a preferred direction (e.g., the average gradient) and the scalar $\lambda\ge 0$ sets how strongly the update is pulled toward it; $\lambda=0$ recovers MGDA. Instead of solving this exactly, SDMGrad runs $K$ projected stochastic gradient steps on $\mathbf{w}$ per iteration, and each step still needs the gradients of all $S$ objectives.

#### Sampling and rescaling

SDMGrad-OS draws independent masks $b_s\sim\mathrm{Bernoulli}(n/S)$, where $n\le S$ is the expected number of sampled objectives, and forms
$$
H(\mathbf{x};\xi,\mathcal{I})=
\big[b_1\nabla f_1(\mathbf{x};\xi),\ldots,b_S\nabla f_S(\mathbf{x};\xi)\big],
$$
where $\mathcal{I}$ denotes the sampling randomness. Only objectives with $b_s=1$ need a backward pass. Since $\mathbb{E}[H]=\frac{n}{S}G(\mathbf{x})$, the rescaled matrix $\frac{S}{n}H$ is unbiased for $G(\mathbf{x})$.

The subproblem gradient, $G^\top\big(G\mathbf{w}+\lambda\mathbf{g}_0\big)$, is a product of two gradient-dependent factors, so SDMGrad-OS estimates the factors with independent samples $\xi,\xi'$ and masks $\mathcal{I},\mathcal{I}'$:
$$
\frac{S^2}{n^2}\,H(\mathbf{x}_t;\xi,\mathcal{I})^\top
\Big(H(\mathbf{x}_t;\xi',\mathcal{I}')\mathbf{w}_{t,k}
+\lambda\,H(\mathbf{x}_t;\xi',\mathcal{I}')\widetilde{\mathbf{w}}\Big).
$$
By independence, the expectation of the product is the product of the expectations, so the estimate is unbiased at the inner iterate $\mathbf{w}_{t,k}$. After $K$ inner steps, the model moves along
$$
\mathbf{d}_t=\frac{S}{n}\,H(\mathbf{x}_t;\zeta,\widetilde{\mathcal{I}})
\big(\mathbf{w}_{t,K}+\lambda\widetilde{\mathbf{w}}\big),
$$
with a fresh sample $\zeta$ and mask $\widetilde{\mathcal{I}}$.

#### The cost–variance trade-off

Every sampled matrix (two per inner step and one for the model update) needs $n$ objective-wise gradients in expectation instead of $S$, so the savings are largest when $S$ is large and $n$ is small. The price is variance: rescaling by $\gamma := S/n$ inflates it, and $\gamma$ appears explicitly in the convergence bounds. Smaller $n$ means cheaper iterations but more of them.

#### Guarantees

Under Lipschitz-continuous gradients, unbiased stochastic gradients with bounded variance, and bounded gradients, Xiao et al. (2023) bound two quantities: the **Pareto-stationarity gap** $\frac1T\sum_t\mathbb{E}\|G(\mathbf{x}_t)\mathbf{w}_t^*\|^2$, with $\mathbf{w}_t^*\in\arg\min_{\mathbf{w}\in\Delta_S}\|G(\mathbf{x}_t)\mathbf{w}\|^2$, and the **conflict-avoidance (CA) distance** between the expected update direction and the ideal direction given by the exact subproblem.

| Regime | Inner steps | Stationarity rate | Samples per objective | CA distance |
|---|---|---|---|---|
| CA-accurate (Theorem 2) | $K$ grows with $T$ | $\widetilde O\big(\sqrt{S\gamma/T}\big)$ | $\widetilde O(\epsilon^{-6})$ | $O(\epsilon)$ |
| Fast (Theorem 4) | $K=1$ | $O\big(S\gamma/\sqrt{T}\big)$ | $O(\epsilon^{-2})$ | Constant |

Tracking the conflict-avoidant direction accurately requires many inner samples per iteration. If a constant CA distance is acceptable, one inner step suffices and the sample complexity matches standard stochastic optimization.

:::danger
**Scope.** The guarantees hold *in expectation*. They do not imply that every objective decreases at every iteration, since a sampled step can ignore some objectives entirely. They also do not identify the best $n$: the bounds grow with $\gamma$ while the savings grow as $n$ shrinks, and the right balance depends on how expensive each objective's gradient is relative to the rest of the iteration.
:::

### 2.3.2 Periodic weight updates: PSMGD

Objective sampling still recomputes the weights at every step. PSMGD (Xu et al., 2025) asks whether they need to be recomputed at all.

#### Motivation: the weights move slowly

MGDA recomputes its weights every iteration, paying $S$ backward passes, an $O(S^2 d)$ Gram matrix, and a subproblem solve each time; on QM-9 with 11 objectives, it takes 12.94 min/epoch versus 1.70 for LS. LS never updates its weights and ignores how conflicts evolve. Empirically, adaptive weights matter but change slowly. On the two-objective Fonseca problem, whose Pareto front is non-convex, LS with random fixed weights reaches only the two endpoints of the front, whereas MGDA finds trade-off points along it; yet the MGDA weights stabilize within about five iterations. On Multi-MNIST, the per-iteration MGDA weights mostly stay within $[0.4,0.6]$ (Xu et al., 2025).

<!-- Suggested figure: Fig. 1 of Xu et al. (2025) (Fonseca trajectories and MGDA weight curves). -->

This is an empirical observation, not a general property: the weights can move faster early in training or when task relationships shift. It nevertheless suggests that most per-step recomputations are redundant.

#### The algorithm

Write $\Phi(\mathbf{x},\mathbf{w})=\sum_s w_s f_s(\mathbf{x})$ for the scalarized objective, so $\nabla_{\mathbf{x}}\Phi = G(\mathbf{x})\mathbf{w}$, and $\Phi(\mathbf{x},\mathbf{w};\xi)$ for its minibatch version.

:::info
**Algorithm 2: PSMGD**

For $t=0,1,\ldots,T-1$:
1. Sample a minibatch $\xi_t$.
2. **If** $t \bmod R = 0$ (weight-update step): compute all $S$ stochastic gradients and set
   $$\hat{\mathbf{w}}_t\in\arg\min_{\mathbf{w}\in\Delta_S}\Big\|\sum_{s} w_s\nabla f_s(\mathbf{x}_t;\xi_t)\Big\|^2,\qquad \mathbf{w}_t=\alpha_t\mathbf{w}_{t-R}+(1-\alpha_t)\hat{\mathbf{w}}_t,$$
   with $\mathbf{w}_0=\hat{\mathbf{w}}_0$ and momentum $\alpha_t\in[0,1]$.
3. **Else** (reuse step): set $\mathbf{w}_t=\mathbf{w}_{t-1}$.
4. Update $\mathbf{x}_{t+1}=\mathbf{x}_t-\eta_t\nabla_{\mathbf{x}}\Phi(\mathbf{x}_t,\mathbf{w}_t;\xi_t)$. On a reuse step, this is one backward pass of a fixed weighted loss, exactly as in LS.
:::

Over a block of $R$ iterations, PSMGD needs $S+R-1$ backward passes and one subproblem solve, whereas stochastic MGDA needs $RS$ passes and $R$ solves. The period $R$ interpolates between the two extremes: $R=1$ with $\alpha_t\equiv0$ recovers stochastic MGDA, and $R\ge T$ recovers LS with weights fixed at initialization.

The momentum plays two roles. In practice, it smooths the single-minibatch estimate $\hat{\mathbf{w}}_t$. In the analysis, $1-\alpha_t$ bounds how far the weights move at each update, which controls both the bias from sample-dependent weights and the drift of the scalarized objective being optimized.

#### Guarantees

The analysis assumes $L$-smooth objectives, unbiased stochastic gradients with bounded variance, and bounded gradients and objective values, plus bounded iterates (convex case) or $\mu$-strong convexity (strongly convex case). Unlike the analysis of SMG (Liu & Vicente, 2024), it does **not** require the optimal-weight map $\mathbf{w}^*(\mathbf{x})$ to be Lipschitz, an assumption that is hard to verify in practice.

Convergence is measured by $\|\nabla_{\mathbf{x}}\Phi(\mathbf{x}_t,\mathbf{w}_t)\|^2$, which upper-bounds the Pareto-stationarity gap $\min_{\mathbf{w}\in\Delta_S}\|G(\mathbf{x}_t)\mathbf{w}\|^2$ because $\mathbf{w}_t\in\Delta_S$. With suitable schedules for $\eta_t$ and $\alpha_t$, PSMGD converges at rate $O(1/\sqrt T)$ in the non-convex and convex cases and $O(1/T)$ in the strongly convex case. The error terms from weight drift and sample-dependent weights are summed only over weight-update steps, so infrequent updates do not slow the rate.

Because an MGDA iteration costs $S$ backward passes and a PSMGD reuse step costs one, iteration counts are misleading. Xu et al. (2025) therefore measure **backpropagation (BP) complexity**, the total number of backward passes to reach accuracy $\epsilon$. If $T_\epsilon$ iterations are needed, PSMGD uses $\frac{T_\epsilon}{R}(S+R-1)$ backward passes.

:::success
**Key result.** With $R=\Omega(S)$, the factor $(S+R-1)/R$ is $O(1)$, so the BP complexity is $O(1/\epsilon^2)$ in the non-convex and convex cases and $O(1/\epsilon)$ in the strongly convex case, the same orders as SGD on a single objective.
:::

#### Scope and practice

:::danger
**Scope.** The bounds certify Pareto stationarity, not the quality of the trade-off. LS, the limit $R\to\infty$, also converges to a Pareto-stationary point. Periodic MGDA updates affect *which* stationary point is reached and how balanced the trajectory is, which the stationarity measure does not capture. The bounds should therefore not be read as recommending an arbitrarily large $R$.
:::

In practice, $R$ is tuned empirically. On QM-9, $R=4$ gives $\Delta m\%=88.2$ at 2.61 min/epoch and $R=32$ gives $116.2$ at 2.07 min/epoch, versus $177.6$ for LS and $120.5$ for MGDA (lower is better). All PSMGD variants are about $5$–$6\times$ faster per epoch than MGDA. Because it needs fewer epochs, PSMGD also reaches a target test loss sooner than LS in wall-clock time (109 versus 170 minutes; MGDA takes 755) (Xu et al., 2025).

#### Comparison with other stochastic MOO methods

| Method | Weight updates | Lipschitz $\mathbf{w}^*$ assumed | BP complexity (non-convex) |
|---|---|---|---|
| SMG (Liu & Vicente, 2024) | Every step | Yes | $O(S/\epsilon^2)$ |
| CR-MOGM (Zhou et al., 2022), MoCo (Fernando et al., 2023), MoDo (Chen et al., 2023), SDMGrad (Xiao et al., 2023) | Every step | No | $O(S/\epsilon^2)$ |
| PSMGD (https://hackmd.io/team/moosurveyXu et al., 2025) | Every $R$ steps | No | $O\big((S+R-1)/(R\epsilon^2)\big)$ |

The methods in the second row differ in how they correct the bias of stochastic weights, but all compute $S$ gradients per step. PSMGD reduces how often this happens, and the two ideas are in principle compatible.

:::info
**Takeaway.** Section 2.3 lowers the *average* number of objective-wise gradients per iteration: from $S$ to $n$ for SDMGrad-OS and to $(S+R-1)/R$ for PSMGD, while keeping convergence to Pareto stationarity. Both still compute objective-wise gradients some of the time; Section 2.4 removes them entirely.
:::

---

## 2.4 No Per-Objective Gradients: Scalarization-based Methods

The most scalable option is to avoid objective-wise gradients altogether: determine or update the weights from objective *values*, and update the model with a single backward pass on the weighted loss
$$
\mathcal{L}_t(\mathbf{x})=\sum_{s=1}^S w_s(t)f_s(\mathbf{x}).
$$
This removes the $S\cdot C_{\mathrm{bp}}$ term, so the per-iteration cost matches LS for any $S$ and any architecture. The question is how much multi-objective behavior can be recovered from loss values alone. We group methods by how much they aim to recover:

- **Scalarization-based methods** (Section 2.4.1) sets the weights by a rule on the losses, without imitating a multi-gradient method: UW (Kendall et al., 2018), DWA (Liu et al., 2019), RLW (Lin et al., 2022), and STCH (Lin et al., 2024).
- **Approximating multi-gradient updates** (Section 2.4.2) uses loss changes between iterations as a proxy for the gradient inner products that multi-gradient methods need: FAMO (Liu et al., 2023) and MGDA-FA (Zhang et al., 2025).
- **Loss-discrepancy control** (Section 2.4.3) balances the objectives directly through a bilevel formulation: LDC-MTL (Xiao et al., 2025).

### 2.4.1 Scalarization and loss-based weighting 

#### Uncertainty weighting (UW)

Kendall et al. (2018) derive the weights from each task's *homoscedastic uncertainty*: noise inherent in the task's observations that does not depend on the input but may differ across tasks, such as measurement noise in a regression target. With a learnable noise level $\sigma_s>0$ per task and independent Gaussian noise, maximizing the likelihood of $S$ regression tasks leads to
$$\min_{\mathbf{x},\boldsymbol{\sigma}}\;\sum_{s=1}^{S}
\left(
\frac{1}{2\sigma_s^2}f_s(\mathbf{x})
+\log\sigma_s
\right),
$$
where $f_s$ is the squared-error loss. The effective weight $w_s=1/(2\sigma_s^2)$ is smaller for noisier tasks, and the $\log\sigma_s$ term prevents $\sigma_s$ from growing without bound and switching the task off. $\mathbf{x}$ and $\boldsymbol{\sigma}$ are trained jointly with one backward pass; the paper also extends the formulation to classification.

#### Dynamic weight average (DWA)

Liu et al. (2019) weight each objective by its recent rate of loss reduction. With $\bar f_s(t)$ the average loss of objective $s$ in epoch $t$,
$$
w_s(t) = \frac{S\exp(r_s(t-1)/\tau)}
{\sum_{j=1}^{S}\exp(r_j(t-1)/\tau)},
\qquad
r_s(t-1)=\frac{\bar f_s(t-1)}{\bar f_s(t-2)},
$$
where $\tau>0$ is a temperature. Objectives that improve more slowly (larger $r_s$) get larger weights; a larger $\tau$ makes the weights more uniform, and $\sum_s w_s(t)=S$. The weights are fixed within each epoch, uniform in the first two epochs, and cost almost nothing to compute.

#### Random loss weighting (RLW)

Lin et al. (2022) draw the weights at random at every iteration:
$$
\widetilde{w}_s(t)\sim\mathcal{N}(0,1),
\qquad
w_s(t)=
\frac{\exp(\widetilde{w}_s(t))}
{\sum_{j=1}^{S}\exp(\widetilde{w}_j(t))},
$$
so $\mathbf{w}_t\in\Delta_S$ and is independent of the losses. The paper also studies uniform, Dirichlet, and Bernoulli sampling. RLW is a simple baseline for testing whether a carefully designed weighting rule is actually needed.

#### Smooth Tchebycheff scalarization (STCH)

Lin et al. (2024) start from Tchebycheff scalarization, which minimizes the largest preference-weighted gap to the ideal point,
$$
\min_{\mathbf{x}}\max_{s\in[S]}
p_s(f_s(\mathbf{x})-z_s^*),
$$
where $\mathbf{p}\in\Delta_S$ encodes preferences and $z_s^*$ is the ideal value of objective $s$. To remove the nonsmooth maximum, STCH uses the log-sum-exp approximation
$$\mathcal{L}_{\mathrm{STCH}}(\mathbf{x})=\mu\log
\sum_{s=1}^S
\exp\left(
\frac{p_s(f_s(\mathbf{x})-z_s^*)}{\mu}
\right),
$$
with smoothing parameter $\mu>0$. Its gradient is
$$
\nabla \mathcal{L}_{\mathrm{STCH}}(\mathbf{x})=\sum_{s=1}^S \underbrace{\pi_s(\mathbf{x})\,p_s}_{=:\,w_s(\mathbf{x})}\,\nabla f_s(\mathbf{x}),
\qquad
\pi_s(\mathbf{x}) = \frac{
\exp\left(p_s(f_s(\mathbf{x})-z_s^*)/\mu\right)
}{
\sum_{j}
\exp\left(p_j(f_j(\mathbf{x})-z_j^*)/\mu\right)
},
$$
so STCH is a weighted loss whose weights depend only on objective values, reference values, and preferences, and one backward pass suffices. Objectives with larger weighted gaps get more emphasis, increasingly so as $\mu$ decreases.

:::success
**Guarantees (Lin et al., 2024).** A global minimizer of $\mathcal{L}_{\mathrm{STCH}}$ is weakly Pareto optimal, and Pareto optimal if all preferences are positive or the minimizer is unique (Theorem 3.7). Under mild conditions and for small enough $\mu$, every Pareto-optimal solution is attained for some preference vector (Theorem 3.8). In the nonconvex case, every stationary point of $\mathcal{L}_{\mathrm{STCH}}$ is Pareto stationary (Theorem 3.10).
:::

#### Comparison

| Method | Weight determination | Backward passes | Extra weight memory | Weight computation |
|---|---|---|---|---|
| Linear scalarization | Fixed | $1$ | $O(S)$ | $O(1)$ |
| UW | Learned task uncertainty | $1$ | $O(S)$ | $O(S)$ |
| DWA | Historical loss ratios | $1$ | $O(S)$ | $O(S)$ |
| RLW | Random sampling | $1$ | $O(S)$ | $O(S)$ |
| STCH | Preference-weighted gaps to the ideal point | $1$ | $O(S)$ | $O(S)$ |

"Weight computation" counts the scalar operations needed to obtain the $S$ weights, excluding loss evaluation and backpropagation. All methods match LS in backward passes.

:::danger
**Limitations.** None of these methods reproduces the conflict-avoiding behavior of rules such as MGDA. UW, DWA, and RLW are heuristic or random and give no guarantee that all objectives improve. STCH has Pareto guarantees for its minimizers, but its weights target the worst-off objective rather than a common descent direction.
:::

### 2.4.2 Approximating multi-gradient updates from loss changes

:::warning
**Key observation.** Multi-gradient methods update their weights using the gradient inner products $G^\top G$, and these can be *observed* through the losses. After a small step $\mathbf{x}_{t+1}=\mathbf{x}_t-\eta\,\mathbf{d}_t$, a first-order Taylor expansion gives
$$
f_s(\mathbf{x}_t)-f_s(\mathbf{x}_{t+1})\approx\eta\,\nabla f_s(\mathbf{x}_t)^\top\mathbf{d}_t \quad\text{for every } s.
$$
If $\mathbf{d}_t = G(\mathbf{x}_t)\mathbf{w}_t$, the vector of loss decreases approximates $\eta\,G^\top G\,\mathbf{w}_t$, which is $\eta$ times the gradient of the min-norm objective $\frac12\|G\mathbf{w}\|^2$ at $\mathbf{w}_t$. A gradient step on the weights can therefore be taken from loss values alone.
:::

FAMO (Liu et al., 2023) applies this observation to log-losses to balance relative improvement, and MGDA-FA (Zhang et al., 2025) applies it to the raw losses to approximate MGDA.

#### FAMO

**Objective.** FAMO aims to decrease all losses at roughly equal *relative* rates. For a step $\mathbf{x}_{t+1}=\mathbf{x}_t-\eta\mathbf{d}$, the relative improvement of task $s$ is
$$r_s(\eta,\mathbf{d})=\frac{f_s(\mathbf{x}_t)-f_s(\mathbf{x}_{t+1})}
{f_s(\mathbf{x}_t)},$$
with losses assumed strictly positive. Measuring relative rather than absolute improvement makes the method insensitive to loss scales. FAMO maximizes the worst-case relative improvement with a quadratic penalty on the step:
$$
\max_{\mathbf{d}\in\mathbb{R}^d}
\min_{s\in[S]}
\frac{r_s(\eta,\mathbf{d})}{\eta}
-\frac{1}{2}\|\mathbf{d}\|^2.
$$

**Reduction to MGDA on log-losses.** For small steps, $r_s(\eta,\mathbf{d})/\eta \approx \nabla f_s^\top\mathbf{d}/f_s = \nabla\log f_s^\top\mathbf{d}$, so the problem becomes $\max_{\mathbf{d}}\min_s \nabla\log f_s(\mathbf{x})^\top\mathbf{d} - \frac12\|\mathbf{d}\|^2$. Its dual (Liu et al., 2023, Proposition 3.1) is
$$
\mathbf{w}^*(\mathbf{x})
\in
\arg\min_{\mathbf{w}\in\Delta_S}
\frac{1}{2}
\big\|
\widetilde G(\mathbf{x})\mathbf{w}
\big\|^2, \qquad \mathbf{d}^*(\mathbf{x})=\widetilde G(\mathbf{x})\mathbf{w}^*(\mathbf{x}),
$$
with $\widetilde G(\mathbf{x}) := [\nabla\log f_1(\mathbf{x}),\ldots,\nabla\log f_S(\mathbf{x})]$, i.e., MGDA applied to the log-losses. Given the weights, $\mathbf{d}^*$ is the gradient of $\sum_s w^*_s\log f_s(\mathbf{x})$ and costs one backward pass.

**Weight update from loss changes.** FAMO amortizes the subproblem over training by taking one gradient step on the weights per iteration,
$$\mathbf{w}_{t+1}=\mathbf{w}_t-\beta\,\boldsymbol{\delta}_t,
\qquad \boldsymbol{\delta}_t=\widetilde G_t^\top \widetilde G_t\mathbf{w}_t,\quad \widetilde G_t := \widetilde G(\mathbf{x}_t).$$
By the key observation applied to the log-losses,
$$
\boldsymbol{\delta}_t \approx
\frac{1}{\eta}
\begin{bmatrix}
\log f_1(\mathbf{x}_t)-\log f_1(\mathbf{x}_{t+1})\\
\vdots\\
\log f_S(\mathbf{x}_t)-\log f_S(\mathbf{x}_{t+1})
\end{bmatrix},
$$
so the weight update needs only the log-losses before and after the model step. In practice, FAMO parameterizes $\mathbf{w}$ as a softmax over logits, applies the update to the logits via the chain rule, and adds a small regularization term.

#### MGDA-FA

**Objective.** MGDA-FA (Zhang et al., 2025) targets the MGDA direction itself, through the regularized min-norm problem
$$\mathbf{w}^*(\mathbf{x})=\arg\min_{\mathbf{w}\in\Delta_S}
\Big\{
\frac{1}{2}\|G(\mathbf{x})\mathbf{w}\|^2+\frac{\rho}{2}\|\mathbf{w}\|^2
\Big\},$$
where $\rho>0$ makes the problem strongly convex with a unique solution.

**Weight update from loss changes.** MGDA-FA takes one projected gradient step on the weights per iteration,
$$\mathbf{w}_{t+1}=\Pi_{\Delta_S}\Big[
\mathbf{w}_t-\beta
\big(
G(\mathbf{x}_t)^\top G(\mathbf{x}_t)\mathbf{w}_t
+\rho\mathbf{w}_t
\big)
\Big].$$
For the model update $\mathbf{x}_{t+1}=\mathbf{x}_t-\eta\, G(\mathbf{x}_t)\mathbf{w}_t$, Taylor expansion gives
$$
\mathbf{F}(\mathbf{x}_t)-\mathbf{F}(\mathbf{x}_{t+1})
=\eta\, G(\mathbf{x}_t)^\top G(\mathbf{x}_t)\mathbf{w}_t-\mathbf{r}_t,
\qquad \|\mathbf{r}_t\| = O(\eta^2)
$$
for smooth objectives. Substituting yields the gradient-free weight update
$$\mathbf{w}_{t+1}=\Pi_{\Delta_S}\Big[
\mathbf{w}_t-\beta
\Big(
\frac{\mathbf{F}(\mathbf{x}_t)-\mathbf{F}(\mathbf{x}_{t+1})}{\eta}
+\rho\mathbf{w}_t
\Big)
\Big].$$
Each iteration is one backward pass on $\sum_s w_s(t)f_s(\mathbf{x})$ plus a weight update from the observed loss changes, at the price of an extra loss evaluation and an $O(\eta)$ error in the weight gradient.

:::success
**Guarantees (Zhang et al., 2025).** For deterministic MOO with lower-bounded, twice-differentiable objectives satisfying generalized $\ell$-smoothness (which includes standard $L$-smoothness), and with suitable step sizes, regularization, and a warm start:
- $\frac{1}{T}\sum_{t=0}^{T-1}\|G(\mathbf{x}_t)\mathbf{w}_t\|^2=O(\epsilon^2)$ (Theorem 4);
- the *iteration-wise* CA distance satisfies $\|G(\mathbf{x}_t)\mathbf{w}_t^*-G(\mathbf{x}_t)\mathbf{w}_t\|=O(\epsilon)$, i.e., the loss-based weights track the exact regularized MGDA weights at every iteration (Theorem 5).

Under the parameter choices needed for the second guarantee, an $\epsilon$-accurate Pareto-stationary point with $O(\epsilon)$ iteration-wise CA distance is reached in $O(\epsilon^{-11})$ iterations. No global bounded-gradient assumption is required.
:::

#### Comparison: FAMO and MGDA-FA

| Method | Target of the weights | Loss signal | Backward passes | Extra weight memory | Weight computation |
|---|---|---|---|---|---|
| FAMO | Equal relative improvement (MGDA on log-losses) | Log-loss differences | $1$ | $O(S)$ | $O(S)$ |
| MGDA-FA | Regularized MGDA direction | Loss differences | $1$ | $O(S)$ | $O(S\log S)$ |

Both need one backward pass plus one extra loss evaluation per iteration and never store the $d\times S$ gradient matrix. The $O(S\log S)$ entry comes from the Euclidean projection onto the simplex.

:::danger
**Limitation.** Both methods replace exact gradient information with loss differences, which introduces two sources of error.
- **Step-size error.** The loss difference matches $\eta\,G^\top G\,\mathbf{w}_t$ only to first order, so the larger the model step $\eta$, the less accurate the weight update. MGDA-FA's analysis accounts for this error, but its stronger CA guarantee also requires a warm start.
- **Sampling noise.** In minibatch training, $f_s(\mathbf{x}_t)$ and $f_s(\mathbf{x}_{t+1})$ are usually evaluated on different minibatches. Their difference then mixes two effects, the model update and the change of minibatch, and only the first carries information about $G^\top G\,\mathbf{w}_t$.
:::

### 2.4.3 Loss-discrepancy control: LDC-MTL

Xiao et al. (2025) proposed Loss Discrepancy Control for Multi-Task Learning (LDC-MTL), which aims to achieve balanced task performance by explicitly controlling discrepancies among objective losses. Unlike FAMO and MGDA-FA, which approximate multi-gradient updates using loss differences, LDC-MTL determines the aggregation weights through a bilevel optimization formulation. The main idea is to minimize the weighted sum of losses while adjusting the weights to reduce the gaps between different task losses. Specifically, LDC-MTL considers the following bilevel problem:

$$
\begin{align}
&\min_\mathbf{W}\sum_{i=1}^{S-1}|\tau_il_i(\mathbf{x})-\tau_{i+1}l_{i+1}(\mathbf{x})|:=f(\mathbf{W},\mathbf{x}),\\
&\text{s.t.  }\mathbf{x}^*\in\arg\min_\mathbf{x}\sum_{s=1}^{S}\sigma_s(\mathbf{W})f_s(\mathbf{x}):=g(\mathbf{W},\mathbf{x}),
\end{align}
$$

where $\boldsymbol{\sigma}(\mathbf{W})\in\Delta_S$ denotes the objective weights generated by a lightweight routing network parameterized by $\mathbf{W}$. The coefficients $\boldsymbol{\tau}$ control the relative importance of task-loss discrepancies, with the paper considering $\boldsymbol{\tau}=\boldsymbol{\sigma}(\mathbf{W})$ and $\boldsymbol{\tau}=\mathbf{1}$. The lower-level problem optimizes the model parameters through a weighted loss, while the upper-level problem adjusts the routing parameters to reduce the discrepancies between adjacent weighted task losses. Thus, the formulation encourages balanced objective values without requiring all losses to be exactly equal.

![flowchart_1](https://hackmd.io/_uploads/S1c0h286Mx.png)
<p align="center">
  <strong>Figure 2:</strong> Bilevel loss discrepancy control pipeline for multi-task learning.
</p>

However, directly solving this bilevel problem requires repeatedly optimizing the lower-level model and differentiating through its solution, introducing substantial computational overhead. To avoid this cost, LDC-MTL reformulates the lower-level optimality requirement through the value-function constraint,

$$\min_{\mathbf{W},\mathbf{x}}f(\mathbf{W},\mathbf{x})\;\;\text{s.t. }g(\mathbf{W},\mathbf{x})-g(\mathbf{W},\mathbf{x}^*)
\leq 0.
$$

Introducing a penalty coefficient $\lambda>0$ gives the penalized objective

$$
\min_{\mathbf{W},\mathbf{x}}f(\mathbf{W},\mathbf{x})+\lambda(g(\mathbf{W},\mathbf{x})-g(\mathbf{W},\mathbf{x}^*))
$$

The penalty controls the trade-off between minimizing loss discrepancies and approaching a minimizer of the weighted lower-level objective. A conventional first-order bilevel method would still require an inner optimization loop to approximate $\mathbf{x}^*(\mathbf{W})$. LDC-MTL instead observes empirically that the gradient of the lower-level objective with respect to the routing parameters, evaluated near the lower-level solution, is substantially smaller than the corresponding gradient at the current model parameters. 

![gradient_norm_combined](https://hackmd.io/_uploads/Syn_0n86Gg.png)
<p align="center">
  <strong>Figure 3:</strong> Gradient norm values during the training
process on the CelebA and Cityscapes datasets.
</p>

Motivated by this observation, it neglects the lower-level solution-dependent term and performs the simplified updates:

$$
\begin{aligned}
&\mathbf{x}_{t+1}=\mathbf{x}_{t}-\eta\left[\nabla_{\mathbf{x}}f(\mathbf{W}_t,\mathbf{x}_t)+\lambda\nabla_{\mathbf{x}}g(\mathbf{W}_t,\mathbf{x}_t)\right]\\
&\mathbf{W}_{t+1}=\mathbf{W}_{t}-\eta
\left[\nabla_{\mathbf{W}}f(\mathbf{W}_t,\mathbf{x}_t)+\lambda\nabla_{\mathbf{W}}g(\mathbf{W}_t,\mathbf{x}_t)\right].
\end{aligned}
$$

These updates correspond to gradient descent on the simplified objective $f(\mathbf{W},\mathbf{x})+\lambda g(\mathbf{W},\mathbf{x})$. Consequently, the model and routing parameters can be updated jointly through a single backward pass, without computing individual objective gradients, Hessian-vector products, or nested lower-level optimization loops. The method therefore achieves $O(1)$ backward-pass complexity with respect to the number of objectives, while requiring only $O(S)$ scalar loss and weight information. LDC-MTL also provides theoretical convergence guarantees.


:::success
**Guarantees.** Under Lipschitz continuity, smoothness, and a Polyak–Łojasiewicz (PL) condition on the lower-level objective, Xiao et al. (2025, Theorem 1) establish convergence to an $\epsilon$-accurate stationary point of the penalized bilevel problem. If this stationary point is a local/global solution to penalized problem, it is also a
local/global solution to the constraint problem, and the it also satisfies a Pareto-stationarity guarantee:

$$
\min_{\mathbf{w}\in\Delta_S}
\|G(\mathbf{x})\mathbf{w}\|^2=\mathcal{O}(\epsilon).
$$

The guarantee extends to the simplified single-loop LDC-MTL algorithm when the omitted lower-level routing-gradient term is $O(\epsilon)$. Thus, the simplified algorithm retains the theoretical guarantee under an additional approximation condition.
:::


<!-- TODO: LDC-MTL (Xiao et al., 2025). Suggested structure, to match the rest of the section: (i) objective: control the discrepancy between (normalized) task losses rather than target the worst-off task (contrast with STCH) or a min-norm direction (contrast with FAMO/MGDA-FA); (ii) the bilevel formulation and how the weights are updated without per-task gradients; (iii) cost: O(1) backward passes per iteration; (iv) convergence guarantee and what it certifies. -->

:::info
**Takeaway.** Section 2.4 matches the per-iteration cost of LS for any $S$ and any architecture. The methods differ in how much multi-objective behavior they recover: none (RLW), heuristic balance (UW, DWA), Pareto guarantees for the minimizer of a fixed scalarization (STCH), or an approximation of the multi-gradient direction with explicit error control (FAMO, MGDA-FA).
:::

---

## 2.5 Summary and Open Directions

The three strategies attack different parts of the cost model of Section 2.1 and trade efficiency against fidelity to the multi-gradient update in different ways.

| Strategy | Representative methods | Objective-wise gradients per iteration | What is kept | What is given up |
|---|---|---|---|---|
| Cheaper (2.2) | MGDA-UB, IMTL-G, RotoGrad, Aligned-MTL-UB | $S$, but through the heads only | Any aggregation rule; one encoder pass | Exact MGDA weights; needs a shared representation with light heads |
| Fewer (2.3) | SDMGrad-OS, PSMGD | $n$ in expectation (SDMGrad-OS); $(S+R-1)/R$ on average (PSMGD) | Convergence to Pareto stationarity; CA distance (SDMGrad-OS) | Per-step descent on all objectives; extra variance ($\gamma=S/n$) or stale weights |
| None (2.4) | UW, DWA, RLW, STCH, FAMO, MGDA-FA, LDC-MTL | $0$ | LS cost for any $S$ and architecture | Exact gradient information; guarantees range from none to approximate MGDA |

**Which strategy to use.** With a large shared backbone and light heads, Section 2.2 keeps the most multi-gradient fidelity at nearly LS cost. When $S$ is large, Sections 2.3 and 2.4 scale better. When all objectives share the whole network, as in multi-objective LLM alignment, only the methods of Sections 2.3 and 2.4 reduce the cost.

**How to evaluate.** Methods with the same convergence rate can differ by a factor of $S$ in computation, so iteration counts alone are misleading. Comparisons should report backward passes or wall-clock time, individual objective values (not only averages such as $\Delta m\%$), and sensitivity to efficiency parameters such as $n$ and $R$, since a cheaper method that converges to a less balanced solution does not offer the same benefit.

**Open directions.**
- **Combining strategies.** The strategies are complementary: for example, PSMGD's weight-update steps could compute their gradients at a shared representation, and objective sampling could be used on those steps. How the respective guarantees compose remains to be worked out.
- **Adaptive efficiency parameters.** $n$ and $R$ are currently fixed and tuned by hand. Choosing them adaptively, e.g., refreshing the weights only when the gradient geometry changes, could reduce cost without sacrificing balance.
- **Stochastic analysis of loss-based approximations.** The guarantees for MGDA-FA are stated for deterministic MOO. Extending them to minibatch loss differences, where sampling noise enters the weight update directly, is a natural next step.

---

## References

<!-- Our own MOO papers -->

- Ban, H., & Ji, K. (2024). Fair resource allocation in multi-task learning. *ICML 2024*.
- Dong, C., Xiao, P., Wang, Y., & Ji, K. (2026). DeepMTL2R: A library for deep multi-task learning to rank. arXiv:2602.14519.
- Liu, Z., Dong, C., Momma, M., Shao, S., Xu, S., Gao, Y., Yang, H., & Liu, J. (2025). STIMULUS: Achieving fast convergence and low sample complexity in stochastic multi-objective learning. *UAI 2025*. [arXiv:2506.19883](https://arxiv.org/abs/2506.19883)
- Xiao, P., Ban, H., & Ji, K. (2023). Direction-oriented multi-objective learning: Simple and provable stochastic algorithms. *NeurIPS 2023*.
- Xiao, P., Dong, C., Zou, S., & Ji, K. (2025). LDC-MTL: Balancing multi-task learning through scalable loss discrepancy control. arXiv preprint.
- Xu, M., Ju, P., Liu, J., & Yang, H. (2025). PSMGD: Periodic stochastic multi-gradient descent for fast multi-objective optimization. *AAAI 2025*. [arXiv:2412.10961](https://arxiv.org/abs/2412.10961)
- Zhou, T., Momma, M., Dong, C., Yang, F., Guo, C., Shang, J., & Liu, J. (2023). Multi-task learning on heterogeneous graph neural network for substitute recommendation. *KDD 2023 Workshop on Mining and Learning with Graphs (MLG)*.

<!-- Other references -->

- Chen, L., Fernando, H., Ying, Y., & Chen, T. (2023). Three-way trade-off in multi-objective learning: Optimization, generalization and conflict-avoidance. *NeurIPS 2023*.
- Chen, Z., Badrinarayanan, V., Lee, C.-Y., & Rabinovich, A. (2018). GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks. *ICML 2018*.
- Chen, Z., Ngiam, J., Huang, Y., Luong, T., Kretzschmar, H., Chai, Y., & Anguelov, D. (2020). Just pick a sign: Optimizing deep multitask models with gradient sign dropout. *NeurIPS 2020*.
- Désidéri, J.-A. (2012). Multiple-gradient descent algorithm (MGDA) for multiobjective optimization. *Comptes Rendus Mathématique*, 350(5–6), 313–318.
- Fernando, H., Shen, H., Liu, M., Chaudhury, S., Murugesan, K., & Chen, T. (2023). Mitigating gradient bias in multi-objective learning: A provably convergent approach. *ICLR 2023*.
- Fliege, J., & Svaiter, B. F. (2000). Steepest descent methods for multicriteria optimization. *Mathematical Methods of Operations Research*, 51(3), 479–494.
- Javaloy, A., & Valera, I. (2022). RotoGrad: Gradient homogenization in multitask learning. *ICLR 2022*.
- Kendall, A., Gal, Y., & Cipolla, R. (2018). Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. *CVPR 2018*.
- Lin, B., Ye, F., Zhang, Y., & Tsang, I. W. (2022). Reasonable effectiveness of random weighting: A litmus test for multi-task learning. *Transactions on Machine Learning Research*.
- Lin, B., & Zhang, Y. (2023). LibMTL: A Python library for deep multi-task learning. *Journal of Machine Learning Research*, 24(209), 1–7.
- Lin, X., Zhang, X., Yang, Z., Liu, F., Wang, Z., & Zhang, Q. (2024). Smooth Tchebycheff scalarization for multi-objective optimization. *ICML 2024*.
- Liu, B., Feng, Y., Stone, P., & Liu, Q. (2023). FAMO: Fast adaptive multitask optimization. *NeurIPS 2023*.
- Liu, B., Liu, X., Jin, X., Stone, P., & Liu, Q. (2021). Conflict-averse gradient descent for multi-task learning. *NeurIPS 2021*.
- Liu, L., Li, Y., Kuang, Z., Xue, J.-H., Chen, Y., Yang, W., Liao, Q., & Zhang, W. (2021b). Towards impartial multi-task learning. *ICLR 2021*.
- Liu, S., Johns, E., & Davison, A. J. (2019). End-to-end multi-task learning with attention. *CVPR 2019*.
- Liu, S., & Vicente, L. N. (2024). The stochastic multi-gradient algorithm for multi-objective optimization and its application to supervised machine learning. *Annals of Operations Research*.
- Navon, A., Shamsian, A., Achituve, I., Maron, H., Kawaguchi, K., Chechik, G., & Fetaya, E. (2022). Multi-task learning as a bargaining game. *ICML 2022*.
- Ramé, A., et al. (2023). Rewarded soups: Towards Pareto-optimal alignment by interpolating weights fine-tuned on diverse rewards. *NeurIPS 2023*.
- Sener, O., & Koltun, V. (2018). Multi-task learning as multi-objective optimization. *NeurIPS 2018*.
- Senushkin, D., Patakin, N., Kuznetsov, A., & Konushin, A. (2023). Independent component alignment for multi-task learning. *CVPR 2023*.
- Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). Gradient surgery for multi-task learning. *NeurIPS 2020*.
- Zhang, et al. (2025). MGDA-FA. <!-- TODO: full reference -->
- Zhou, S., Zhang, W., Jiang, J., Zhong, W., Gu, J., & Zhu, W. (2022). On the convergence of stochastic multi-objective gradient manipulation and beyond. *NeurIPS 2022*.
- Zhou, Z., et al. (2024b). Beyond one-preference-fits-all alignment: Multi-objective direct preference optimization. *Findings of ACL 2024*.