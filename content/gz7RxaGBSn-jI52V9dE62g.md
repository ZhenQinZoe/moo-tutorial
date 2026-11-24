# 4. Multi-Objective Bilevel Optimization

[TOC]

## 4.0 Overview

Modern machine learning increasingly involves complex model architectures and learning procedures with structured dependencies among different decision variables. Examples include selecting hyperparameters based on the performance of the resulting trained models, learning shared representations that support task-specific adaptation, and adjusting task weights according to the performance achieved after training. While conventional MOO provides a principled framework for characterizing trade-offs among competing objectives, treating all decision variables as jointly optimized quantities does *not* adequately capture hierarchical dependencies in which some variables must solve optimization problems parameterized by others. These challenges motivate **multi-objective bilevel optimization (MOBO)** and the development of theoretical and algorithmic foundations that explicitly account for both competing objectives and nested optimization structures.

In this section, we review recent work on MOBO problems. Combining multiple objectives with a bilevel structure gives rise to different formulations, depending on which level involves multiple objectives. **First**, when multiple objectives appear only at the upper level, they share a common single-objective lower-level subproblem. This formulation can arise when optimizing fine-tuning configurations for a pretrained model: the lower level adapts the model under a given configuration, while the upper level balances multiple evaluation criteria, such as accuracy, fairness, and robustness. **Second**, when multiple objectives appear only at the lower level, the formulation can capture the selection of a preferred solution from the lower-level Pareto set according to an additional upper-level criterion. For example, the lower level may characterize trade-offs among task-specific training losses, while the upper level guides the choice among these trade-offs based on validation performance or user preferences. We focus on these two formulations and trace their development in recent years. **Finally**, we briefly discuss the more general setting in which both levels involve multiple objectives.



## 4.1. MOBO with Multiple Objectives at Upper Level

The MOBO problem with multiple objectives at upper level **(MOBO-U)** can be formulated as follows:
\begin{equation}\tag{MOBO-U}
    \begin{aligned}
        & \min_{\mathbf{x},\mathbf{y}} \mathbf{F}(\mathbf{x},\mathbf{y}) = \big[f_{1}(\mathbf{x},\mathbf{y}), \dotsc, f_{S}(\mathbf{x},\mathbf{y})\big]^\top \\
    & \text{s.t. } \mathbf{y}\in \mathcal{M}(\mathbf{x}) := \arg\min_\mathbf{y} g(\mathbf{x},\mathbf{y}),
    \end{aligned}
\end{equation}
where $S$ denotes the number of tasks, $\mathbf{x}\in\mathbb{R}^p, \mathbf{y}\in\mathbb{R}^q$ are optimization variables with $d=p+q$, and $\mathbf{F}, g$ refer to upper-level (UL) and lower-level (LL) functions, respectively.

MOBO-U problem captures an important requirement in modern machine learning: balancing competing downstream objectives that depend on a shared training or adaptation process. Compared with conventional MOO, the key difficulty lies in accounting for how lower-level solutions respond to upper-level decisions. Inexact lower-level optimization can bias hypergradient estimates, compromising common descent directions and complicating preference-guided Pareto exploration. Without lower-level strong convexity, the solution mapping may also be set-valued, posing further challenges to differentiation and the characterization of Pareto stationarity. Representative applications include multi-task adaptation of foundation models, multi-objective actor–critic learning, and data weighting for RLHF reward modeling and LLM alignment. We are now ready to review several representative algorithms for the MOBO-U problem.



### 4.1.1 Foundations and Early Algorithmic Developments

A natural starting point is to extend MGDA to the bilevel setting. Under the lower-level strong convexity (LLSC) condition, the LL solution $\mathbf{y}^*(\mathbf{x})$ is uniquely determined, and each UL objective can be written as $\phi_s(\mathbf{x}) := f_s(\mathbf{x},\mathbf{y}^*(\mathbf{x}))$. MGDA combines the hypergradients through:
\begin{equation}
    \begin{aligned}
        & \boldsymbol{\lambda}^*(\mathbf{x})\in\arg\min_{\boldsymbol{\lambda}\in\Delta_S}\left\|\sum_{s=1}^{S}\lambda_s\nabla\phi_s(\mathbf{x})\right\|^2, \\
        & \mathbf{d}(\mathbf{x}) = \sum_{s=1}^{S}\lambda_s^*(\mathbf{x})\nabla\phi_s(\mathbf{x}),
    \end{aligned}
\end{equation}
where $\Delta_S := \{\boldsymbol{\lambda}\in\mathbb{R}^S:\lambda_s\geq 0,\sum_{s=1}^{S}\lambda_s=1\}$. If $\mathbf{d}(\mathbf{x})\neq 0$, then $-\mathbf{d}(\mathbf{x})$ is a common descent direction; otherwise, $\mathbf{x}$ is Pareto stationary. Compared to the conventional MGDA, however, both the LL solution and the hypergradients must be approximated.

[Ye et al. (2021)](https://arxiv.org/pdf/2102.07121) introduce **MOML**, an early gradient-based framework for the MOBO-U problem. MOML combines inner-loop LL updates with an MGDA-type UL step based on approximate hypergradients. Its analysis establishes asymptotic convergence of Pareto stationary solutions under the LLSC condition and upper-level P-convexity conditions. The cost of approximating the LL response and differentiating through it nevertheless remains a major concern.

Subsequent works address this computational difficulty from different perspectives. **FORUM**, proposed by [Ye et al. (2024)](https://arxiv.org/pdf/2401.09257), uses a value-function-based reformulation and first-order multi-gradient aggregation to avoid second-order derivative evaluations. Its finite-time analysis controls KKT stationarity and feasibility. **gMOBA**, proposed by [Yang et al. (2024)](https://arxiv.org/pdf/2406.05455), instead tracks the LL solution and the auxiliary variables needed for hypergradient approximation within a single loop, and establishes asymptotic Pareto-stationarity guarantees. These methods improve different aspects of the computation, while leaving further questions about stochastic scenario, preference guidance, and the dependence on LLSC.



### 4.1.2 Stochastic Gradient Bias and Correction: [MoCo (Fernando et al., 2023)](https://arxiv.org/pdf/2210.12624)

The extension from deterministic to stochastic MGDA is not straightforward. Even when each stochastic gradient is unbiased, the combination weights are computed from the same random gradients and depend on them nonlinearly. Consequently, the resulting combination need not be an unbiased estimate of the true MGDA direction. In MOBO-U, this difficulty is compounded by the additional bias from approximate LL solutions and hypergradient computations. Thus, controlling each objective's gradient estimate alone does not immediately guarantee a reliable common descent direction.

:::warning
**Key ideas.**
**MoCo [(Fernando et al., 2023)](https://arxiv.org/pdf/2210.12624)** addresses this issue by introducing a tracking variable for each objective's gradient. It uses the tracked estimates to update the combination weights through a projected gradient step for a regularized MGDA subproblem, and then updates the UL variable along the resulting direction. This avoids solving the weight subproblem exactly at every iteration. The central idea is to control the errors in gradient tracking and weight estimation together, so that the computed direction approaches the true MGDA direction.
:::

Although MoCo is initially developed for stochastic MOO, Section 3.2 extends the framework to nested problems with biased hypergradient estimates. This formulation allows each objective to have its own LL subproblem. When these LL variables are separate blocks, their objectives can be summed into a single scalar LL objective, bringing the problem within the MOBO-U formulation above. The paper considers both simultaneous UL/LL updates and multiple inner LL updates; the latter provide more accurate hypergradient estimates at an additional computational cost.

MoCo establishes finite-time bounds on the average expected squared norm of the true MGDA direction. Writing $T$ for the number of outer iterations, the general bound is $\mathcal{O}(S^{\frac{1}{2}}T^{-\frac{1}{12}})$. Bounded objective values improve this to $\mathcal{O}(ST^{-\frac{2}{5}})$, while the stronger gradient-estimation condition supported by multiple inner updates yields $\mathcal{O}(ST^{-\frac{1}{2}})$. These results show why the accuracy of the inner computation must be considered when interpreting the convergence rate.

MoCo provides a theoretical basis for stochastic MOBO-U, but its nested analysis still relies on LLSC and its update rule does not explicitly incorporate user preferences. This leaves a further question: how can the decision-maker guide which trade-off is obtained?



### 4.1.3 Preference-Guided Pareto Exploration: [WC-MHGD (Zhang et al., 2026a)](https://arxiv.org/pdf/2511.07824)

Pareto stationarity alone does not determine which trade-off among the UL objectives is desirable. For example, two stationary solutions may assign very different priorities to the tasks, even though both satisfy the Pareto stationarity criterion. This motivates **WC-MHGD**, which explicitly incorporates preference information into MOBO-U and supports the exploration of different trade-offs.

:::warning
**Key ideas.**
**[WC-MHGD (Zhang et al., 2026a)](https://arxiv.org/pdf/2511.07824)** combines multi-hypergradient descent with weighted-Chebyshev (WC) scalarization. For a positive preference vector $\mathbf{r}\in\Delta_S$, the WC problem takes the form
\begin{equation}
    \min_{\mathbf{x}}\max_{s\in[S]}\{r_s\phi_s(\mathbf{x})\},
\end{equation}
with a suitable translation of the objective values when needed. Compared with linear scalarization, WC scalarization can represent trade-offs on nonconvex Pareto fronts. Even better, the translated single-objective problem is equivalent to the original MOO problem. Specifically, when varying the preference vector $\mathbf{r}$ across the simplex, the solution to the WC-scalarized problem restores the entire Pareto front.
:::

WC-MHGD incorporates the WC preference term into an MGDA-type quadratic subproblem:
\begin{equation}\tag{WC-MHGD}
    \begin{aligned}
        & \min_{\boldsymbol{\lambda}\geq\mathbf{0}} \underbrace{\left\|\mathbf{K}(\mathbf{r}\odot\boldsymbol{\lambda})\right\|^2}_{\mathrm{MHGD}} - u\underbrace{\boldsymbol{\lambda}^\top(\mathbf{r}\odot\boldsymbol{\Phi}(\mathbf{x}_k))}_{\text{WC-Scalarization}} \\
        & \text{s.t. } \mathbf{1}^\top\boldsymbol{\lambda}=1.
    \end{aligned}
\end{equation}
At each outer iteration, it first approximates the LL solution and then all UL hypergradients $\boldsymbol{\Phi}(\mathbf{x}_k)$, using either conjugate-gradient or Neumann-series techniques. It then computes a direction by solving this WC-MHGD problem, a quadratic program that balances stationarity and WC scalarization through a tunable parameter $u$. The main difficulty is to preserve the relationship between the computed direction and Pareto stationarity despite the errors introduced by the LL approximation. The analysis therefore controls these errors together with the preference-dependent direction.

![WC](https://hackmd.io/_uploads/ryddyWJazl.png)


Under LLSC and the stated regularity conditions, WC-MHGD provides finite-time guarantees in both deterministic and stochastic settings. The deterministic bound contains an $\mathcal{O}(T^{-1})$ term together with LL and hypergradient approximation errors. The stochastic bound additionally depends on sampling and truncation errors, so the same outer-iteration dependence should not be interpreted as the same total computational cost. With suitable parameter choices, the reported oracle complexities scale as $\widetilde{\mathcal{O}}(\epsilon^{-1})$ and $\widetilde{\mathcal{O}}(\epsilon^{-2})$ in the deterministic and stochastic settings.

WC-MHGD thus adds preference guidance to the convergence analysis and enables systematic exploration by varying the preference vector. However, both its design and the analysis still depend on the LLSC condition.



### 4.1.4 Beyond Lower-Level Strong Convexity: [WC-Penalty (Zhang et al., 2026b)](https://arxiv.org/pdf/2605.09094)

The LLSC condition plays a central role in the preceding hypergradient-based methods: it ensures a unique LL solution and, together with smoothness, supports implicit differentiation. Under lower-level general convexity (LLGC), however, $\mathcal{M}(\mathbf{x})$ may contain multiple solutions, and the LL Hessian may be singular. The reduced objectives $\phi_s(\mathbf{x})$ can therefore no longer be defined through a unique differentiable LL response without additional assumptions. As a result, relaxing the LLSC condition not only prevents the hypergradient-based algorithm design, but also makes it unclear to understand the optimality of MOBO-U problems.

:::warning
**Key idea.**
**WC-Penalty [(Zhang et al., 2026b)](https://arxiv.org/pdf/2605.09094)** address this difficulty through an equality-constrained reformulation. For the unconstrained, differentiable, convex LL problem, the condition $\mathbf{y}\in\mathcal{M}(\mathbf{x})$ is equivalent to $\nabla_{\mathbf{y}}g(\mathbf{x},\mathbf{y})=0$. Hence, MOBO-U can be written as
\begin{equation}\tag{ECMO}
    \begin{aligned}
        & \min_{\mathbf{z}} \mathbf{F}(\mathbf{z}) \\
        & \text{s.t. } \mathbf{h}(\mathbf{z}) := \nabla_{\mathbf{y}}g(\mathbf{x},\mathbf{y}) = 0,
    \end{aligned}
\end{equation}
where $\mathbf{z}:=[\mathbf{x}^\top,\mathbf{y}^\top]^\top$ and $\mathbf{F}(\mathbf{z})$ denotes $\mathbf{F}(\mathbf{x},\mathbf{y})$. This representation retains LL optimality as an explicit constraint and avoids selecting a single LL response. The resulting problem is an equality constrained multi-objective optimization (ECMO) problem.
:::

![logic](https://hackmd.io/_uploads/Hy2HyWypMg.png)


The main challenge then becomes characterizing stationarity while accounting for feasibility. The paper uses WC scalarization and its KKT conditions to construct a preference-dependent stationarity residual, and proposes **WC-Penalty** to control this residual. Specifically, it introduces an epigraph variable and nonnegative slack variables for the WC inequalities, penalizes the resulting equalities together with the ECMO constraints, and applies projected gradient updates. The algorithm has a single-loop structure. Although it avoids Hessian inversion, its application to MOBO-U still requires derivatives of $\mathbf{h}=\nabla_{\mathbf{y}}g$, and thus second-order information about $g$.

Under the stated smoothness and constraint-qualification assumptions, WC-Penalty achieves an $\mathcal{O}(ST^{-\frac{1}{2}})$ bound on the average squared KKT residual. A stochastic counterpart provides a high-probability bound of the same order with a horizon-dependent batch size. These guarantees account for feasibility as well as stationarity, and should therefore be distinguished from bounds on the unconstrained MGDA direction.

Its limitations include the remaining convexity and regularity requirements, the conditioning introduced by penalty parameters, and the sampling cost of the stochastic guarantee. Extending this approach to nonconvex LL problems requires further care, since LL stationarity is then insufficient to guarantee LL optimality.



### 4.1.5 Alternative Formulations and Related MTL Methods

A different decision criterion is considered by **MORBiT** ([Gu et al., 2023](https://arxiv.org/pdf/2203.01924)), which minimizes the worst UL objective. The method uses a single-loop stochastic gradient descent-ascent scheme and establishes finite-time stationarity guarantees for the min-max formulation under LLSC. This formulation is useful for robust representation learning and hyperparameter optimization, where protecting the worst-performing objective is the main concern. Its guarantee concerns the selected worst-case criterion, rather than preference-guided exploration of multiple trade-offs.

Related MTL methods also provide useful perspectives on balancing objectives. **FairGrad** ([Ban and Ji, 2024](https://arxiv.org/pdf/2402.15638)) introduces fairness criteria for allocating task improvements and provides a convergence guarantee. **LDC-MTL** ([Xiao et al., 2025](https://arxiv.org/pdf/2502.08585)) develops a bilevel formulation for controlling loss discrepancies and connects its convergence to Pareto stationarity. **ConsMTL** ([Qin et al., 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Qin_Towards_Consistent_Multi-Task_Learning_Unlocking_the_Potential_of_Task-Specific_Parameters_CVPR_2025_paper.pdf)) uses a bilevel design to coordinate shared and task-specific parameters. These works illustrate different ways to manage task conflicts, but their formulations and guarantees need to be distinguished from those for the general MOBO-U problem.



### 4.1.6 Discussion and Comparison

:::info
**Takeaway.**
MOML, FORUM, gMOBA represent the prior MOBO-U algorithmic designs. The subsequent works address three remaining questions in MOBO-U problems: how to control stochastic estimation errors, how to incorporate preferences into the search for Pareto stationary solutions, and how to proceed when the LL solution is no longer uniquely determined. MoCo, WC-MHGD, and WC-Penalty provide representative approaches to these questions, respectively. The following table summarizes the main structural differences; each guarantee also requires standard assumptions such as smoothness, boundedness, and estimation conditions.
:::

| Algorithm (Reference) | Loop Type | Setting | Main Structural Assumptions | Preference | Convergence Criterion and Guarantee |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MOML [Ye et al., 2021] | Double-loop | Deterministic | LLSC, UL P-convexity | Preference-free | Pareto optimality; asymptotic |
| FORUM [Ye et al., 2024] | Double-loop | Deterministic | LLSC | Preference-free | Pareto stationarity; finite-time |
| gMOBA [Yang et al., 2024] | Single-loop | Deterministic | LLSC | Preference-free | Pareto stationarity; asymptotic |
| MoCo [Fernando et al., 2023] | Nested updates | Stochastic | LLSC | Preference-free| Pareto stationarity; finite-time |
| WC-MHGD [Zhang et al., 2026a] | Double-loop | Det. & Stoch. | LLSC | Preference-aware | Pareto stationarity; finite-time |
| WC-Penalty [Zhang et al., 2026b] | Single-loop | Det. & Stoch. | LLGC | Preference-aware | Pareto stationarity; finite-time |
| MORBiT [Gu et al., 2023] | Single-loop | Stochastic | LLSC | Worst-case criterion | Min-max stationarity; finite-time |

:::danger
**Limitations.**
Several questions remain open for developing more general and efficient MOBO-U methods. These include solving problems without assuming LL convexity, further reducing computational costs as the model size and number of objectives grow, and developing efficient strategies to explore the entire Pareto front with good coverage and diversity.
:::



### Reference

1. Ye, F., Lin, B., Yue, Z., Guo, P., Xiao, Q., & Zhang, Y. (2021). Multi-objective meta learning. Advances in Neural Information Processing Systems, 34, 21338-21351.
2. Fernando, H. D., Shen, H., Liu, M., Chaudhury, S., Murugesan, K., & Chen, T. (2023). Mitigating gradient bias in multi-objective learning: A provably convergent approach. In The Eleventh International Conference on Learning Representations.
3. Ye, F., Lin, B., Cao, X., Zhang, Y., & Tsang, I. W. (2024). A first-order multi-gradient algorithm for multi-objective bi-level optimization. In ECAI 2024 (pp. 2621-2628).
4. Gu, A., Lu, S., Ram, P., & Weng, T.-W. (2023). Min-max multi-objective bilevel optimization with applications in robust machine learning. In The Eleventh International Conference on Learning Representations.
5. Yang, X., Yao, W., Yin, H., Zeng, S., & Zhang, J. (2024). Gradient-based algorithms for multi-objective bi-level optimization. Science China Mathematics, 67(6), 1419-1438.
6. Zhang, Z., Liu, Z., Zhang, X., Chen, W. Y., Yang, J., & Liu, J. (2026a, March). Multi-Objective Bilevel Learning. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 40, No. 34, pp. 28653-28661).
7. Zhang, Z., Oh, M. S., Qin, Z., Li, J., Zhang, X., & Liu, J. (2026b). A Tale of Two Problems: Multi-Task Bilevel Learning Meets Equality Constrained Multi-Objective Optimization. arXiv preprint arXiv:2605.09094.
8. Ban, H., & Ji, K. (2024, July). Fair resource allocation in multi-task learning. In Proceedings of the 41st International Conference on Machine Learning (pp. 2715-2731).
9. Xiao, P., Dong, C., Zou, S., & Ji, K. (2025). LDC-MTL: Balancing multi-task learning through scalable loss discrepancy control. arXiv preprint arXiv:2502.08585.
10. Qin, X., Wang, X., & Yan, J. (2025). Towards consistent multi-task learning: Unlocking the potential of task-specific parameters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10067-10076).







## 4.2 MOBO with Multiple Objectives at Lower Level

The MOBO problem with multiple objectives at lower level (MOBO-L) can be formulated as follows:
\begin{equation}\tag{MOBO-L}
    \begin{aligned}
        & \min_{\mathbf{x}} f(\mathbf{x}) \\
    & \text{s.t. } \mathbf{x}\in \mathcal{M} := \arg\min_\mathbf{x} G(\mathbf{x}) = \arg\min_\mathbf{x} \big[g_{1}(\mathbf{x}), \dotsc, g_{S}(\mathbf{x})\big]^\top,
    \end{aligned}
\end{equation}
where $S$ denotes the number of tasks, $\mathbf{x}\in\mathbb{R}^p$ is optimization variable, and $f, G$ refer to upper-level (UL) and lower-level (LL) functions, respectively.




## 4.3 MOBO with Multiple Objectives at Both Levels

When both levels involve multiplease objectives, the MOBO problems can be expressed as:

\begin{equation}\tag{MOBO-UL}
    \begin{aligned}
        & \min_{\mathbf{x},\mathbf{y}} \mathbf{F}(\mathbf{x},\mathbf{y}) = \big[f_{1}(\mathbf{x},\mathbf{y}), \dotsc, f_{S}(\mathbf{x},\mathbf{y})\big]^\top \\
    & \text{s.t. } \mathbf{y}\in \mathcal{P}(\mathbf{x}),
    \end{aligned}
\end{equation}
where $S$ denotes the number of tasks, $\mathbf{x}\in\mathbb{R}^p, \mathbf{y}\in\mathbb{R}^q$ are optimization variables with $d=p+q$, and $\mathbf{F}$ refers to upper-level function, and $\mathcal{P}(x)$ is the set of eﬃcient solutions of a multi-objective lower-level problem.

This setting has received limited attention in the machine learning literature and poses substantial challenges, since the upper-level optimization must account for trade-offs among its own objectives while remaining constrained by a lower-level Pareto solution set that varies with $\mathbf{x}$. A relevant contribution is Yang et al. (2024), who discuss this setting in Remark 3.1 as an extension of gMOBA, a method originally proposed to solve the MOBO-U problem.

Let $g_1,\dotsc,g_m$ denote the lower-level objectives. Yang et al. (2024) Assumes that each $g_j(\mathbf{x},\cdot)$ is $\mu$-strongly convex, and the authors characterize $\mathcal{P}(\mathbf{x})$ through weighted-sum scalarization: $\mathbf{y}\in\mathcal{P}(\mathbf{x})$ if and only if $\mathbf{y}=\mathbf{y}^{\star}(\mathbf{x},\boldsymbol{\lambda}):=\arg\min_{\mathbf{y}\in\mathbb{R}^q}\sum_{j=1}^{T}\lambda_j g_j(\mathbf{x},\mathbf{y})$ for some $\boldsymbol{\lambda}\in\Delta_T:=\{\boldsymbol{\lambda}\in\mathbb{R}^T:\boldsymbol{\lambda}\geq 0,\ \sum_{j=1}^{T}\lambda_j=1\}$. Treating $\boldsymbol{\lambda}$ as an additional upper-level variable yields an equivalent bilevel problem with a scalar lower-level objective. The resulting problem can thus be addressed by gMOBA using approximate hypergradients and a multi-objective proximal update, as mentioned in Section 4.1.1.

:::danger
**Limitations.**
Nevertheless, this extension relies on lower-level strong convexity (LLSC), requiring every $g_j(\mathbf{x},\cdot)$ to be strongly convex. Beyond this assumption, the scalarized lower-level problem may admit multiple solutions, and its weighted-sum characterization need not coincide with $\mathcal{P}(\mathbf{x})$. Specifically, in the general convexity and nonconvex cases, weighted-sum scalarization may additionally fail to recover the Pareto solutions. These limitations highlight the need for a systematic study of MOBO problems with multiple objectives at both levels, particularly with a well-defined lower-level solution set and beyond the LLSC assumption.
:::

### Reference

Yang, X., Yao, W., Yin, H., Zeng, S., & Zhang, J. (2024). Gradient-based algorithms for multi-objective bi-level optimization. Science China Mathematics, 67(6), 1419-1438.