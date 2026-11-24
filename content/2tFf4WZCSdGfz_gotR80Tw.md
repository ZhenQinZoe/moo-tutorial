---
title: "Preference-Guided Multi-Objective Optimization"
author: Lisha Chen
create date: 2026-11-09
update date: 2026-11-09
description: Preference-guided multi-objective optimization through scalarization, objective constraints, target ratios, ordering cones, and Pareto-front approximation.
tags:
  - Multi-objective-optimization
  - Preference-guided-optimization
type: blog
status: draft
---

# Preference-Guided Multi-Objective Optimization

## Problem and notation

Let $X\subseteq\mathbb R^d$ be a nonempty feasible set, and let $f_s:X\to\mathbb R$ be the objective indexed by $s\in[S]=\{1,\ldots,S\}$. We consider minimization of the vector-valued function
$$
\min_{\mathbf{x}\in X}\mathbf{F}(\mathbf{x}),
\qquad
\mathbf{F}(\mathbf{x})=\bigl(f_1(\mathbf{x}),\ldots,f_S(\mathbf{x})\bigr)^\top.
\tag{1}
$$
^eq-moo-model

Let $\Delta_S=\{\mathbf{w}\in\mathbb R^S:w_s\geq0\text{ for all }s\in[S],\ \sum_{s\in[S]}w_s=1\}$ denote the probability simplex.

For Pareto optimality, scalarization, and Pareto stationarity, see [Introduction to Multi-objective Optimization](Intro%20Multi-Objective%20Optimization.md).

## Classical constrained formulations

Classical methods express trade-off preferences through objective bounds, aspiration levels, priorities, or geometric constraints in objective space. Each construction gives a scalar constrained subproblem over the original feasible set $X$. The symbols $f_s$ and $\mathbf{F}$ always denote the objectives in the chosen units, including after any fixed normalization. Fixing its parameters selects a trade-off; varying them can generate alternatives for later selection [Miettinen (1999)](https://doi.org/10.1007/978-1-4615-5563-6) [\[2\]](#^ref-miettinen-1999).

| Method | Preference or geometric control | Constrained construction and distinction |
| --- | --- | --- |
| $\varepsilon$-constraint | Acceptable upper bounds on the other objectives | Minimize one objective with $f_s(\mathbf{x})\leq\varepsilon_s$ for the others. Sweeping bounds generates trade-offs; dominated ties need care [Miettinen (1999)](https://doi.org/10.1007/978-1-4615-5563-6) [\[2\]](#^ref-miettinen-1999). A learning application is given by [Curtis et al. (2024)](https://doi.org/10.1007/s11590-023-02024-6) [\[11\]](#^ref-curtis-2024-fair). |
| Augmented $\varepsilon$-constraint (AUGMECON) | Objective bounds and a positive slack augmentation | Reward normalized constraint slacks in the scalar objective to avoid weakly efficient ties. The finite augmentation weight can affect the selected trade-off [Mavrotas (2009)](https://doi.org/10.1016/j.amc.2009.03.037) [\[71\]](#^ref-mavrotas-2009-augmecon). |
| Normal-boundary intersection (NBI) | Base points on the convex hull of individual-objective minima | Maximize displacement along an inward normal or quasi-normal, with an objective-space equality linking the displaced point to an attainable objective vector [Das and Dennis (1998)](https://doi.org/10.1137/S1052623496307510) [\[7\]](#^ref-das-dennis-1998). |
| Normal constraint / normalized normal constraint (NC / NNC) | Moving cuts relative to the anchor simplex | Minimize one objective subject to half-space constraints normal to the anchor directions. Normalize the objectives before constructing the cuts [Messac et al. (2003)](https://doi.org/10.1007/s00158-002-0276-1) [\[70\]](#^ref-messac-2003-nnc). |
| Goal attainment | An aspiration vector and positive deviation weights | Minimize a common attainment factor under $\mathbf{F}(\mathbf{x})\leq\mathbf{g}+\alpha\mathbf{w}$. This permits goals to be under- or overachieved [Gembicki and Haimes (1975)](https://doi.org/10.1109/TAC.1975.1101105) [\[72\]](#^ref-gembicki-haimes-1975). |
| Lexicographic / multi-level constraints | An ordered list of objective priorities | Optimize objectives sequentially while constraining previously optimized values. Zero tolerance preserves strict priority; positive tolerances permit specified degradation [Miettinen (1999)](https://doi.org/10.1007/978-1-4615-5563-6) [\[2\]](#^ref-miettinen-1999). |
| Equal-spacing continuation | A target distance between neighboring objective vectors | Solve successive constrained subproblems with an objective-space distance equality and a condition selecting the next branch [Pereyra et al. (2013)](https://doi.org/10.1016/j.mcm.2010.12.044) [\[73\]](#^ref-pereyra-2013-continuation). |

![Six objective-space panels comparing NBI, epsilon-constraint and AUGMECON, normalized normal constraints, goal attainment, lexicographic / multi-level constraints, and equal-spacing continuation.](assets/classical-constrained-moo.png)

**Figure 1. Classical constrained constructions for two objectives.** The dashed curve is the common illustrative front $\{(u^2,(1-u)^2)^\top:u\in[0,1]\}$ in the chosen objective units; cyan dots mark selected solutions. Blue arrows indicate geometric constructions rather than numerical solver trajectories. Panel (b) shows the bound geometry shared by $\varepsilon$-constraint and AUGMECON; augmentation changes the scalar objective. Panel (e) illustrates lexicographic / multi-level constraints and uses a positive tolerance for visibility. Panel (f) imposes equal **chord lengths** in those same objective units. Scalar objectives are $f_1,f_2$; bold symbols denote vectors, including $\mathbf{F},\mathbf{x},\mathbf{p},\mathbf{n},\mathbf{c},\mathbf{g},\mathbf{w}$. The constraint $\mathbf{x}\in X$ is implicit in every panel; $\boldsymbol{\beta}\in\Delta_2$ in (a), while $\beta\in[0,1]$ in (c). This original schematic follows the visual style of the supplied reference. [Editable SVG](assets/classical-constrained-moo.svg) · [Figure source](assets/generate-classical-constrained-moo.mjs).
^fig-classical-constrained-moo

### Bounds, goals, and priorities

For minimization, the $\varepsilon$-constraint subproblem is
$$
\min_{\mathbf{x}\in X} f_1(\mathbf{x})
\quad\text{subject to}\quad
f_s(\mathbf{x})\leq\varepsilon_s,\qquad s=2,\ldots,S.
$$
AUGMECON introduces slacks $\xi_s\geq0$ through $f_s(\mathbf{x})+\xi_s=\varepsilon_s$ and minimizes $f_1(\mathbf{x})-\rho\sum_{s=2}^S\xi_s/R_s$, where $R_s>0$ are scaling ranges and $\rho>0$. The augmentation favors improvement in constrained objectives; exact lexicographic priority requires additional care beyond choosing a small finite $\rho$ [Mavrotas (2009)](https://doi.org/10.1016/j.amc.2009.03.037) [\[71\]](#^ref-mavrotas-2009-augmecon).

Goal attainment instead solves $\min_{\mathbf{x}\in X,\,\alpha\in\mathbb R}\alpha$ subject to $\mathbf{F}(\mathbf{x})\leq\mathbf{g}+\alpha\mathbf{w}$, with an aspiration vector $\mathbf{g}$ and $w_s>0$. Positive $\alpha$ allows shortfall from the goals; negative $\alpha$ improves on every goal. This is a constrained epigraph form of $\min_{\mathbf{x}\in X}\max_s (f_s(\mathbf{x})-g_s)/w_s$ [Gembicki and Haimes (1975)](https://doi.org/10.1109/TAC.1975.1101105) [\[72\]](#^ref-gembicki-haimes-1975); see also the [goal-attainment formulation](https://www.mathworks.com/help/optim/ug/multiobjective-optimization-algorithms.html).

For two-objective lexicographic optimization, first find $v_1=\min_{\mathbf{x}\in X}f_1(\mathbf{x})$, then minimize $f_2$ subject to $f_1(\mathbf{x})\leq v_1+\delta$. Setting $\delta=0$ preserves the best primary value; $\delta>0$ specifies an acceptable loss in that value [Miettinen (1999)](https://doi.org/10.1007/978-1-4615-5563-6) [\[2\]](#^ref-miettinen-1999).

### Anchor geometry: NBI and normal constraints

Choose objective units before constructing the anchor geometry; fixed ideal-value shifts and positive scaling factors can be used for normalization. We retain the notation $f_s$ and $\mathbf{F}$ for the resulting objectives. Assume each individual minimum is attained. Let $\mathbf{x}^{(s)}$ minimize objective $s$, and form the anchor matrix $\Phi=[\mathbf{F}(\mathbf{x}^{(1)}),\ldots,\mathbf{F}(\mathbf{x}^{(S)})]$. For $\boldsymbol{\beta}\in\Delta_S$, the base point is $\mathbf{p}(\boldsymbol{\beta})=\Phi\boldsymbol{\beta}$. NBI solves
$$
\max_{\mathbf{x}\in X,\,t\in\mathbb R} t
\quad\text{subject to}\quad
\mathbf{F}(\mathbf{x})=\mathbf{p}(\boldsymbol{\beta})+t\mathbf{n},
$$
where $\mathbf{n}$ is an inward normal or quasi-normal to the anchor simplex. The parameter $\boldsymbol{\beta}$ locates a base point; it is not a weighted-sum coefficient in the objective [Das and Dennis (1998)](https://doi.org/10.1137/S1052623496307510) [\[7\]](#^ref-das-dennis-1998).

For two objectives in these chosen units, let $\mathbf{a}$ and $\mathbf{b}$ be the anchors minimizing $f_1$ and $f_2$, respectively, and let $\mathbf{c}=\mathbf{b}-\mathbf{a}$. A normal-constraint subproblem is
$$
\min_{\mathbf{x}\in X}f_2(\mathbf{x})
\quad\text{subject to}\quad
\mathbf{c}^{\top}\bigl(\mathbf{F}(\mathbf{x})-\mathbf{p}(\beta)\bigr)\leq0,
\qquad
\mathbf{p}(\beta)=(1-\beta)\mathbf{a}+\beta\mathbf{b}.
$$
The cut retains a half-space, allowing the scalar objective to select a boundary point. Boundary constructions can return dominated candidates, so Pareto filtering can be needed. Equal spacing of anchor parameters also need not yield equal arc-length spacing on the front [Messac et al. (2003)](https://doi.org/10.1007/s00158-002-0276-1) [\[70\]](#^ref-messac-2003-nnc).

### Spacing constraints and continuation

Equal-spacing continuation adds a condition such as
$$
\left\|\mathbf{F}(\mathbf{x}_{k+1})-\mathbf{F}(\mathbf{x}_k)\right\|_2=\Delta,
\qquad \Delta>0,
$$
to a continuation formulation, together with a condition selecting the forward branch. The distance equality controls **objective-space chord length**; Pareto optimality comes from the accompanying optimality or optimization conditions. It does not follow from the distance equality alone [Pereyra et al. (2013)](https://doi.org/10.1016/j.mcm.2010.12.044) [\[73\]](#^ref-pereyra-2013-continuation). These subproblems can use SQP, interior-point, or active-set solvers; the numerical solver and the preference construction are separate choices.

## Preference-guided methods: related work

Preference guidance specifies which trade-offs are desirable, through scalarization weights, objective bounds, target ratios, priorities, or an ordering of improvement directions. The discussion below draws on the preference-guided methods and Pareto-front approximation literature reviewed in [FERERO, Section 4 and Appendix B](https://arxiv.org/html/2412.01773) [\[9\]](#^ref-chen-2024-ferero) and more recent work on front sampling, with references to the original works.

The comparison follows the finite-set and learned-family branches of [Chen et al. (2026)](https://openreview.net/forum?id=eCUcXXH3PS) [\[6\]](#^ref-chen-2026-survey). Section and figure numbering here follows the [arXiv version, Sections 4–5 and Figure 2](https://arxiv.org/pdf/2501.10945v3#page=4), which distinguishes preference-vector methods, set-generation methods, hypernetworks, preference-conditioned networks, and model combinations. A learned continuous family is an approximation; its parameterization alone does not certify recovery of the full Pareto set.

### Prescribed preferences and adaptive reference vectors

| Method | Preference input or selection role | Main idea and distinction |
| --- | --- | --- |
| Weighted sum and Tchebycheff scalarization | Weights and, for Tchebycheff, a reference point | Solve a parameterized scalar problem. Weighted sums can miss unsupported trade-offs; the choice of scalarization and objective scaling affects which solutions are reachable. See the [scalarization overview](Intro%20Multi-Objective%20Optimization.md#scalarization-choosing-a-trade-off). |
| Smooth Tchebycheff scalarization (STCH) | Weights, a reference point, and a smoothing parameter | Smooth the Tchebycheff maximum to obtain a differentiable scalarization for gradient methods while retaining preference-dependent trade-off control [Lin et al. (2024)](https://proceedings.mlr.press/v235/lin24y.html) [\[10\]](#^ref-lin-2024-stch). |
| Pareto Multi-Task Learning (PMTL) | Reference vectors defining objective-space regions | Decompose the problem into constrained subproblems and use gradient-based updates within the corresponding regions. A region preference does not require exact alignment with its reference ray [Lin et al. (2019)](https://papers.nips.cc/paper_files/paper/2019/hash/685bfde03eb646c27ed565881917c71c-Abstract.html) [\[12\]](#^ref-lin-2019-pmtl). |
| Exact Pareto Optimal (EPO) search | Desired ratios between task losses | Seek a Pareto solution whose weighted losses are balanced, corresponding to a ray in objective space. Controlled ascent can worsen some losses to move toward the desired trade-off [Mahapatra and Rajan (2020)](https://proceedings.mlr.press/v119/mahapatra20a.html) [\[13\]](#^ref-mahapatra-2020-epo). |
| WC-MGDA and XWC-MGDA | Weighted Chebyshev preferences and their extension relative to a reference model | Combine preference alignment with Pareto-stationarity conditions. The extended formulation accommodates shifted reference geometry rather than requiring a ray from the origin [Momma et al. (2022)](https://proceedings.mlr.press/v162/momma22a.html) [\[14\]](#^ref-momma-2022-wc-mgda). |
| Preference-based MGDA (PMGDA) | A preference-violation function for a target ray or region of interest | Alternate prediction steps that improve objectives with correction steps that reduce preference violation. Correction can permit objective increases. [Zhang et al. (2026)](https://arxiv.org/abs/2402.09492) [\[49\]](#^ref-zhang-2026-pmgda). |
| Dynamic barrier gradient descent | Objective constraints or lexicographic priorities | Optimize a secondary loss while respecting a primary-loss requirement; lexicographic optimization treats improvement of the primary objective as having priority [Gong et al. (2021)](https://papers.nips.cc/paper_files/paper/2021/hash/f7b027d45fd7484f6d0833823b98907e-Abstract.html) [\[15\]](#^ref-gong-2021-dynamic-barrier). |
| FERERO | An ordering cone together with affine constraints on objective values | Combine preferences over improvement directions with bounds, ratios, or shifted objective-space constraints in a constrained vector optimization framework [Chen et al. (2024)](https://arxiv.org/abs/2412.01773) [\[9\]](#^ref-chen-2024-ferero). |
| GMOOAR | Reference vectors adapted during optimization | Use bilevel optimization to adapt reference vectors to the underlying front while optimizing the corresponding trade-off solutions. [Chen and Kwok (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d313b4a8c88eba7f0542c489899cec77-Abstract-Conference.html) [\[50\]](#^ref-chen-2022-gmooar). |

### Finite-set coverage and sampling

| Method | Preference input or selection role | Main idea and distinction |
| --- | --- | --- |
| Hypervolume gradient methods (GradHV / HIGA-MO) | A hypervolume reference point; alternatives for later selection | Optimize the hypervolume of a finite solution set using objective gradients. The reference point affects the set-quality criterion; it is not a target ray. [Wang et al. (2017)](https://doi.org/10.1007/978-3-319-54157-0_44) [\[52\]](#^ref-wang-2017-gradhv). |
| Multi-objective Stein variational gradient descent (MOO-SVGD) | No explicit target preference required; generate diverse alternatives for later selection | Combine multi-gradient improvement with particle repulsion to encourage diversity when approximating the Pareto front [Liu et al. (2021)](https://papers.nips.cc/paper_files/paper/2021/hash/7bb16972da003e87724f048d76b7e0e1-Abstract.html) [\[20\]](#^ref-liu-2021-moo-svgd). |
| Uniform multi-objective designs (UMOD) | A desired representative distribution of trade-offs | Optimize a max-packing design on the front as a surrogate for minimizing fill distance. This targets set coverage rather than one prescribed ray. [Zhang et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/040d3b6af368bf71f952c18da5713b48-Abstract-Conference.html) [\[51\]](#^ref-zhang-2024-umod). |
| Few for Many (F4M) | A small solution budget for many objectives | Use Tchebycheff set scalarization so each objective is served by at least one solution. Complementary objective coverage differs from densely approximating the front. [Lin et al. (2025)](https://arxiv.org/abs/2405.19650) [\[53\]](#^ref-lin-2025-f4m). |
| Sum-of-minimum optimization (SoM) | Set-level objective coverage without a prescribed preference vector | Minimize the sum of best-achieved objective values across a small set of decisions, allowing different decisions to specialize in different objectives. [Ding et al. (2024)](https://proceedings.mlr.press/v235/ding24a.html) [\[54\]](#^ref-ding-2024-som). |
| Many-objective multi-solution transport (MosT) | Objective-to-solution assignments learned by optimal transport | Alternate transport-based assignments with multi-objective updates to learn a few complementary solutions, each emphasizing a subset of objectives. [Li et al. (2025)](https://arxiv.org/abs/2403.04099) [\[55\]](#^ref-li-2025-most). |
| SURF | Objective-space arc-length positions, or a menu for later preference selection | Reconstruct the map from scalarization weights to normalized cumulative arc length and sample its inverse for approximately uniform coverage of the reachable front. Uniform weights generally do not give uniform coverage [Jiang et al. (2026, v2)](https://arxiv.org/abs/2605.20619v2) [\[48\]](#^ref-jiang-2026-surf). |
| Multiple-target sampling gradient descent (MT-SGD) | Multiple target distributions; no user-specified objective-space ray required | Evolve intermediate distributions toward multiple targets. Its asymptotic connection to MGDA links the sampling formulation to multi-objective optimization [Phan et al. (2022)](https://papers.nips.cc/paper_files/paper/2022/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html) [\[21\]](#^ref-phan-2022-mt-sgd). |

### Learned families through hypernetworks

| Method | Preference input or selection role | Main idea and distinction |
| --- | --- | --- |
| Pareto HyperNetworks (PHN) | Preference vectors supplied to a hypernetwork; trade-offs can be selected after training | Learn a preference-to-model map to generate models for different operating points without independently training one model per preference. The learned family approximates the front [Navon et al. (2021)](https://openreview.net/forum?id=NjF772F4ZZR) [\[16\]](#^ref-navon-2021-phn). |
| Controllable Pareto Multi-Task Learning (CPMTL) | Preference vectors conditioning a hypernetwork | Learn a preference-to-parameter map using preference-conditioned multi-objective optimization, enabling trade-off control at inference time. [Lin et al. (2020)](https://arxiv.org/abs/2010.06313) [\[56\]](#^ref-lin-2020-cpmtl). |
| PHN-HVI | Multiple trade-off vectors sampled together | Generate several models with a hypernetwork and jointly maximize their hypervolume, coupling the quality of the generated solutions. [Hoang et al. (2023)](https://ojs.aaai.org/index.php/AAAI/article/view/25953) [\[57\]](#^ref-hoang-2023-phn-hvi). |
| SUHNPF | A learned constrained Pareto manifold for later trade-off selection | Use Fritz–John conditions and guided double gradient descent to learn a weak Pareto manifold; apply the learned manifold to generate target-model parameters. [Gupta et al. (2022)](https://proceedings.mlr.press/v180/gupta22a.html) [\[58\]](#^ref-gupta-2022-suhnpf). |
| Hyper-Trans | Reference vectors and split feasibility constraints | Replace an MLP hypernetwork with a transformer-based hypernetwork for controllable Pareto-front learning within the specified constraint region. [Tuan et al. (2024)](https://arxiv.org/abs/2402.05955) [\[59\]](#^ref-tuan-2024-hyper-trans). |
| Pareto set learning for expensive MOO (PSL) | Preference vectors indexing a learned set of candidate decisions | Learn a continuous Pareto-set approximation within Bayesian optimization, supporting exploration of trade-off regions under a limited objective-evaluation budget [Lin et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7a583691ccfcf8945ab714b677ccbf0b-Abstract.html) [\[17\]](#^ref-lin-2022-psl). |

### Learned families through preference-conditioned networks

| Method | Preference input or selection role | Main idea and distinction |
| --- | --- | --- |
| COSMOS | Preference vectors supplied directly to the predictive network | Train a preference-conditioned model with scalarized losses and an angular alignment penalty, avoiding a hypernetwork that generates every model parameter. [Ruchte and Grabocka (2021)](https://arxiv.org/abs/2103.13392) [\[60\]](#^ref-ruchte-2021-cosmos). |
| You Only Train Once (YOTO) | Coefficients parameterizing the loss function | Train one model over a distribution of losses and condition its outputs on the loss coefficients, allowing trade-off changes at inference time. [Dosovitskiy and Djolonga (2020)](https://openreview.net/forum?id=HyxY6JHKwr) [\[61\]](#^ref-dosovitskiy-2020-yoto). |
| Controllable dynamic multi-task architectures (DYMU) | Task-importance preferences and a computational budget | Condition both network weights and a tree-structured architecture on user inputs, allowing inference-time control over task trade-offs and computation. [Raychaudhuri et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/html/Raychaudhuri_Controllable_Dynamic_Multi-Task_Architectures_CVPR_2022_paper.html) [\[62\]](#^ref-raychaudhuri-2022-dymu). |

### Learned families through model combinations

| Method | Preference input or selection role | Main idea and distinction |
| --- | --- | --- |
| Pareto Manifold Learning (PaMaL) | Preference coefficients for weight-space interpolation | Jointly train base models and interpolate their parameters to obtain a continuous family of task trade-offs. [Dimitriadis et al. (2023)](https://proceedings.mlr.press/v202/dimitriadis23a.html) [\[63\]](#^ref-dimitriadis-2023-pamal). |
| LORPMAN | Preference coefficients combining low-rank model components | Combine a shared main network with low-rank components to reduce the parameter cost of learning a preference-dependent model family. [Chen and Kwok (2024)](https://proceedings.mlr.press/v235/chen24x.html) [\[64\]](#^ref-chen-2024-lorpman). |
| Preference-aware Pareto merging (YOMO / Pareto Merging) | Preferences over the tasks of pretrained models | Use a preference-aware low-rank tensor parameterization to learn a family of merged models in one merging process. [Chen and Kwok (2024)](https://arxiv.org/abs/2408.12105v1) [\[65\]](#^ref-chen-2024-yomo). |
| Panacea | A vector of preferences over alignment objectives | Inject preferences into singular values of SVD-based low-rank adapters, allowing a single LLM to adapt to different trade-offs. [Zhong et al. (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89f39d0b3d49a47606a165eefba2778c-Abstract-Conference.html) [\[66\]](#^ref-zhong-2024-panacea). |
| Mixture-of-experts model fusion | Preferences controlling the fusion of pretrained task models | Learn preference-dependent routers that combine specialized model weights into different trade-off models. [Tang et al. (2024)](https://arxiv.org/abs/2406.09770v1) [\[67\]](#^ref-tang-2024-moe-fusion). |
| Pareto low-rank adapters (PaLoRA) | Preference coefficients interpolating task-specific adapters | Learn a shared base network and task-specific low-rank adapters, with a preference-sampling schedule supporting a continuous family of trade-offs. [Dimitriadis et al. (2025)](https://arxiv.org/abs/2407.08056) [\[68\]](#^ref-dimitriadis-2025-palora). |

### Related preference-guided policy methods

| Method | Preference input or selection role | Main idea and distinction |
| --- | --- | --- |
| Pareto Policy Adaptation (PPA) | Distributions over linear preferences in multi-objective reinforcement learning | Adapt a policy to a preference distribution through a loss based on a Pareto-stationarity solver and implicit differentiation [Kyriakis et al. (2022)](https://openreview.net/pdf?id=wfZGut6e09) [\[18\]](#^ref-kyriakis-2022-ppa). |
| Pareto Policy Pool (P3) | Reference vectors defining regions of the return–uncertainty front | Learn a pool of policies through constrained bi-objective problems in model-based offline reinforcement learning, allowing later selection among trade-offs [Yang et al. (2022)](https://openreview.net/pdf/811177d23b2117fa0be0cc22952e7c1e3325bf59.pdf) [\[19\]](#^ref-yang-2022-policy-pool). |

For EPO, a positive preference vector $\mathbf{r}$ encodes the target relation $r_1f_1(\mathbf{x})=\cdots=r_Sf_S(\mathbf{x})$, so the target ray is proportional to $(1/r_1,\ldots,1/r_S)$. This illustrates why scalarization weights and desired loss ratios should not be treated as interchangeable. Exact ray alignment also requires the ray to intersect attainable objective values; an algorithm cannot satisfy an infeasible preference [Mahapatra and Rajan (2020)](https://proceedings.mlr.press/v119/mahapatra20a.html) [\[13\]](#^ref-mahapatra-2020-epo).

**PMGDA: prediction and preference correction.** Let $h(\mathbf{F}(\mathbf{x}))\geq0$ measure violation of a desired ray or region, with zero indicating satisfaction. PMGDA uses a prediction step to improve the objectives along a direction tangent to the preference constraint, and a correction step to reduce violation when the preference is not satisfied. The correction step can allow some objective values to increase. Prediction manages objective improvement, while correction explicitly steers toward preference satisfaction [Zhang et al. (2026)](https://arxiv.org/abs/2402.09492) [\[49\]](#^ref-zhang-2026-pmgda).

**FERERO: preferences over values and improvement directions.** FERERO distinguishes *absolute preferences*, which restrict preferred objective values, from *relative preferences*, which define an ordering cone. Absolute preferences take the form
$$
B_{\mathrm{g}}\mathbf{F}(\mathbf{x})+\mathbf{b}_{\mathrm{g}}\leq0,
\qquad
B_{\mathrm{h}}\mathbf{F}(\mathbf{x})+\mathbf{b}_{\mathrm{h}}=0,
$$
where the matrices and offsets specify the desired region. Relative preferences use a polyhedral cone $C_A=\{\mathbf{y}:A\mathbf{y}\geq0\}$ to define improvement through $A\mathbf{F}(\mathbf{x})$. The identity $A=I_S$ recovers the usual componentwise ordering. The framework combines these ingredients with gradient-based updates and includes single-loop deterministic and stochastic variants with convergence guarantees under their stated assumptions [Chen et al. (2024)](https://arxiv.org/abs/2412.01773) [\[9\]](#^ref-chen-2024-ferero). These two kinds of preference describe *what* is specified; a priori, a posteriori, and interactive describe *when* it is specified.

**Learning or generating many trade-offs.** Rather than solving a new problem independently for every preference, **Pareto HyperNetworks** learn a preference-to-model map [Navon et al. (2021)](https://openreview.net/forum?id=NjF772F4ZZR) [\[16\]](#^ref-navon-2021-phn). **Pareto set learning for expensive MOO** learns a continuous approximation of the Pareto set within a Bayesian optimization framework, allowing exploration of different trade-off regions under a limited evaluation budget [Lin et al. (2022)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7a583691ccfcf8945ab714b677ccbf0b-Abstract.html) [\[17\]](#^ref-lin-2022-psl). In reinforcement learning, **Pareto Policy Adaptation** adapts policies to distributions over preferences [Kyriakis et al. (2022)](https://openreview.net/pdf?id=wfZGut6e09) [\[18\]](#^ref-kyriakis-2022-ppa), while **Pareto Policy Pool** constructs policies in different reference-vector regions to trade off model return and uncertainty in model-based offline learning [Yang et al. (2022)](https://openreview.net/pdf/811177d23b2117fa0be0cc22952e7c1e3325bf59.pdf) [\[19\]](#^ref-yang-2022-policy-pool). These approaches can support selection after training even though preference vectors are used during training.

**Parameterization and training are separate choices.** Hypernetworks, directly conditioned networks, and model combinations describe how preferences enter the model. Their training can use scalarized losses, preference-aware updates, or a set-level criterion. For example, PHN considers scalarization and EPO-based training [Navon et al. (2021)](https://openreview.net/forum?id=NjF772F4ZZR) [\[16\]](#^ref-navon-2021-phn), whereas PHN-HVI trains jointly generated models through hypervolume maximization [Hoang et al. (2023)](https://ojs.aaai.org/index.php/AAAI/article/view/25953) [\[57\]](#^ref-hoang-2023-phn-hvi). These choices affect preference alignment and coverage even when the architecture is unchanged.

**Local continuous exploration.** A related route constructs a local continuous Pareto-set approximation from an existing solution using gradient and Hessian information, rather than learning a global preference-conditioned generator [Ma et al. (2020)](https://proceedings.mlr.press/v119/ma20a.html) [\[69\]](#^ref-ma-2020-exploration). Local expansion does not establish coverage of the entire front.

**SURF: geometry-aware sampling of trade-offs.** SURF (Sampling Uniformly along the Pareto Front) adjusts scalarization weights to obtain approximately uniform objective-space coverage [Jiang et al. (2026, v2)](https://arxiv.org/abs/2605.20619v2) [\[48\]](#^ref-jiang-2026-surf). For two objectives, let $\mathbf{x}_w^\star$ minimize $wf_1(\mathbf{x})+(1-w)f_2(\mathbf{x})$ over $X$, and suppose $\mathbf{z}(w)=\mathbf{F}(\mathbf{x}_w^\star)$ smoothly traces the front with positive speed. Define the normalized cumulative arc length
$$
\Phi(w)=\frac{\int_0^w\|\mathbf{z}'(\tau)\|\,\mathrm{d}\tau}{\int_0^1\|\mathbf{z}'(\tau)\|\,\mathrm{d}\tau}.
$$
Equal arc-length positions correspond to weights $w_j=\Phi^{-1}(j/K)$ for $j=0,\ldots,K$, where $K$ is the number of intervals. SURF alternates scalarized solves with reconstruction of $\Phi$. This supports a posteriori selection from an evenly spaced menu; a prescribed normalized arc-length position $q$ can also be targeted using $w=\Phi^{-1}(q)$. Its bi-objective linear-scalarization analysis gives linear convergence of the CDF reconstruction error to an $O(K^{-2})$ discretization floor under the stated assumptions. Coverage depends on the chosen scalarization's reachable front, and Euclidean arc length depends on objective scaling.

**Front approximation without an explicit target preference.** The related probabilistic approaches include multi-objective Stein variational gradient descent, which combines gradient-based improvement with particle repulsion to encourage diverse solutions [Liu et al. (2021)](https://papers.nips.cc/paper_files/paper/2021/hash/7bb16972da003e87724f048d76b7e0e1-Abstract.html) [\[20\]](#^ref-liu-2021-moo-svgd), and multiple-target sampling gradient descent, which evolves distributions toward multiple targets and connects asymptotically to MGDA [Phan et al. (2022)](https://papers.nips.cc/paper_files/paper/2022/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html) [\[21\]](#^ref-phan-2022-mt-sgd). These methods are relevant to preference-based decision making through the alternatives they generate, even when their updates do not target a user-specified ray.

**Connections to vector and constrained optimization.** Cone-based preferences build on classical vector optimization [Jahn (2011)](https://doi.org/10.1007/978-3-642-17005-8) [\[22\]](#^ref-jahn-2011) and steepest descent for general cone-induced orders [Graña Drummond and Svaiter (2005)](https://doi.org/10.1016/j.cam.2004.06.018) [\[23\]](#^ref-drummond-svaiter-2005). Related constrained MOO algorithms include SQP-based methods [Fliege and Vaz (2016)](https://doi.org/10.1137/15M1016424) [\[24\]](#^ref-fliege-vaz-2016) and descent methods using active sets for equality and inequality constraints [Gebken et al. (2019)](https://doi.org/10.1007/978-3-319-96104-0_2) [\[25\]](#^ref-gebken-2019). These provide optimization machinery that can be used with preference constraints; the preference model and the numerical solver remain separate choices. Convergence to a stationary or KKT point under specified assumptions should be distinguished from global Pareto optimality, exact preference satisfaction, and coverage of the full front.

## References

[2] Kaisa Miettinen (1999). [Nonlinear Multiobjective Optimization](https://doi.org/10.1007/978-1-4615-5563-6). Kluwer Academic Publishers. ^ref-miettinen-1999

[6] Weiyu Chen, Baijiong Lin, Xiaoyuan Zhang, Xi Lin, Han Zhao, Qingfu Zhang, and James Kwok (2026). [Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond](https://openreview.net/forum?id=eCUcXXH3PS). *Transactions on Machine Learning Research*, August 2026. [Published PDF](https://openreview.net/pdf?id=eCUcXXH3PS). [arXiv version used for section and figure numbering](https://arxiv.org/abs/2501.10945v3). ^ref-chen-2026-survey

[7] Indraneel Das and J. E. Dennis (1998). [Normal-Boundary Intersection: A New Method for Generating the Pareto Surface in Nonlinear Multicriteria Optimization Problems](https://doi.org/10.1137/S1052623496307510). *SIAM Journal on Optimization*, **8**(3), 631–657. ^ref-das-dennis-1998

[9] Lisha Chen, AFM Saif, Yanning Shen, and Tianyi Chen (2024). [FERERO: A Flexible Framework for Preference-Guided Multi-Objective Learning](https://papers.nips.cc/paper_files/paper/2024/hash/21a33eba893ca3890e395651b38810df-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, **37**. [arXiv:2412.01773](https://arxiv.org/abs/2412.01773). ^ref-chen-2024-ferero

[10] Xi Lin, Xiaoyuan Zhang, Zhiyuan Yang, Fei Liu, Zhenkun Wang, and Qingfu Zhang (2024). [Smooth Tchebycheff Scalarization for Multi-Objective Optimization](https://proceedings.mlr.press/v235/lin24y.html). *Proceedings of the 41st International Conference on Machine Learning*, PMLR **235**, 30479–30509. ^ref-lin-2024-stch

[11] Frank E. Curtis, Suyun Liu, and Daniel P. Robinson (2024). [Fair machine learning through constrained stochastic optimization and an $\varepsilon$-constraint method](https://doi.org/10.1007/s11590-023-02024-6). *Optimization Letters*, **18**, 1975–1991. First published online in 2023. ^ref-curtis-2024-fair

[12] Xi Lin, Hui-Ling Zhen, Zhenhua Li, Qing-Fu Zhang, and Sam Kwong (2019). [Pareto Multi-Task Learning](https://papers.nips.cc/paper_files/paper/2019/hash/685bfde03eb646c27ed565881917c71c-Abstract.html). *Advances in Neural Information Processing Systems*, **32**. ^ref-lin-2019-pmtl

[13] Debabrata Mahapatra and Vaibhav Rajan (2020). [Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization](https://proceedings.mlr.press/v119/mahapatra20a.html). *Proceedings of the 37th International Conference on Machine Learning*, PMLR **119**, 6597–6607. ^ref-mahapatra-2020-epo

[14] Michinari Momma, Chaosheng Dong, and Jia Liu (2022). [A Multi-objective / Multi-task Learning Framework Induced by Pareto Stationarity](https://proceedings.mlr.press/v162/momma22a.html). *Proceedings of the 39th International Conference on Machine Learning*, PMLR **162**, 15895–15907. ^ref-momma-2022-wc-mgda

[15] Chengyue Gong, Xingchao Liu, and Qiang Liu (2021). [Automatic and Harmless Regularization with Constrained and Lexicographic Optimization: A Dynamic Barrier Approach](https://papers.nips.cc/paper_files/paper/2021/hash/f7b027d45fd7484f6d0833823b98907e-Abstract.html). *Advances in Neural Information Processing Systems*, **34**, 29630–29642. ^ref-gong-2021-dynamic-barrier

[16] Aviv Navon, Aviv Shamsian, Ethan Fetaya, and Gal Chechik (2021). [Learning the Pareto Front with Hypernetworks](https://openreview.net/forum?id=NjF772F4ZZR). *International Conference on Learning Representations*. ^ref-navon-2021-phn

[17] Xi Lin, Zhiyuan Yang, Xiaoyuan Zhang, and Qingfu Zhang (2022). [Pareto Set Learning for Expensive Multi-Objective Optimization](https://proceedings.neurips.cc/paper_files/paper/2022/hash/7a583691ccfcf8945ab714b677ccbf0b-Abstract.html). *Advances in Neural Information Processing Systems*, **35**. ^ref-lin-2022-psl

[18] Panagiotis Kyriakis, Jyotirmoy V. Deshmukh, and Paul Bogdan (2022). [Pareto Policy Adaptation](https://openreview.net/pdf?id=wfZGut6e09). *International Conference on Learning Representations*. ^ref-kyriakis-2022-ppa

[19] Yijun Yang, Jing Jiang, Tianyi Zhou, Jie Ma, and Yuhui Shi (2022). [Pareto Policy Pool for Model-based Offline Reinforcement Learning](https://openreview.net/pdf/811177d23b2117fa0be0cc22952e7c1e3325bf59.pdf). *International Conference on Learning Representations*. ^ref-yang-2022-policy-pool

[20] Xingchao Liu, Xin Tong, and Qiang Liu (2021). [Profiling Pareto Front With Multi-Objective Stein Variational Gradient Descent](https://papers.nips.cc/paper_files/paper/2021/hash/7bb16972da003e87724f048d76b7e0e1-Abstract.html). *Advances in Neural Information Processing Systems*, **34**. ^ref-liu-2021-moo-svgd

[21] Hoang Phan, Ngoc Tran, Trung Le, Toan Tran, Nhat Ho, and Dinh Phung (2022). [Stochastic Multiple Target Sampling Gradient Descent](https://papers.nips.cc/paper_files/paper/2022/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, **35**. ^ref-phan-2022-mt-sgd

[22] Johannes Jahn (2011). [Vector Optimization: Theory, Applications, and Extensions, 2nd ed.](https://doi.org/10.1007/978-3-642-17005-8). Springer. ^ref-jahn-2011

[23] L. M. Graña Drummond and Benar Fux Svaiter (2005). [A steepest descent method for vector optimization](https://doi.org/10.1016/j.cam.2004.06.018). *Journal of Computational and Applied Mathematics*, **175**(2), 395–414. ^ref-drummond-svaiter-2005

[24] Jörg Fliege and A. Ismael F. Vaz (2016). [A Method for Constrained Multiobjective Optimization Based on SQP Techniques](https://doi.org/10.1137/15M1016424). *SIAM Journal on Optimization*, **26**(4), 2091–2119. ^ref-fliege-vaz-2016

[25] Bennet Gebken, Sebastian Peitz, and Michael Dellnitz (2019). [A Descent Method for Equality and Inequality Constrained Multiobjective Optimization Problems](https://doi.org/10.1007/978-3-319-96104-0_2). In *Numerical and Evolutionary Optimization – NEO 2017*, Studies in Computational Intelligence **785**, 29–61. Springer. ^ref-gebken-2019

[48] Liuyuan Jiang, Chentong Huang, and Lisha Chen (2026). [SURF: Steering the Scalarization Weight to Uniformly Traverse the Pareto Front](https://arxiv.org/abs/2605.20619v2). arXiv:2605.20619v2, revised October 9, 2026. ^ref-jiang-2026-surf

[49] Xiaoyuan Zhang, Xi Lin, and Qingfu Zhang (2026; preprint 2024). [PMGDA: A Preference-based Multiple Gradient Descent Algorithm](https://doi.org/10.1109/TETCI.2025.3526459). *IEEE Transactions on Emerging Topics in Computational Intelligence*, **10**(2), 1098–1110. [Preprint](https://arxiv.org/abs/2402.09492). ^ref-zhang-2026-pmgda

[50] Weiyu Chen and James T. Kwok (2022). [Multi-Objective Deep Learning with Adaptive Reference Vectors](https://proceedings.neurips.cc/paper_files/paper/2022/hash/d313b4a8c88eba7f0542c489899cec77-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, **35**. ^ref-chen-2022-gmooar

[51] Xiaoyuan Zhang, Genghui Li, Xi Lin, Yichi Zhang, Yifan Chen, and Qingfu Zhang (2024). [Gliding over the Pareto Front with Uniform Designs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/040d3b6af368bf71f952c18da5713b48-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, **37**. ^ref-zhang-2024-umod

[52] Hao Wang, André Deutz, Thomas Bäck, and Michael Emmerich (2017). [Hypervolume Indicator Gradient Ascent Multi-objective Optimization](https://doi.org/10.1007/978-3-319-54157-0_44). In *Evolutionary Multi-Criterion Optimization*, LNCS **10173**, 654–669. ^ref-wang-2017-gradhv

[53] Xi Lin, Yilu Liu, Xiaoyuan Zhang, Fei Liu, Zhenkun Wang, and Qingfu Zhang (2025). [Few for Many: Tchebycheff Set Scalarization for Many-Objective Optimization](https://arxiv.org/abs/2405.19650). *International Conference on Learning Representations*. ^ref-lin-2025-f4m

[54] Lisang Ding, Ziang Chen, Xinshang Wang, and Wotao Yin (2024). [Efficient Algorithms for Sum-Of-Minimum Optimization](https://proceedings.mlr.press/v235/ding24a.html). *Proceedings of the 41st International Conference on Machine Learning*, PMLR **235**, 10927–10959. ^ref-ding-2024-som

[55] Ziyue Li, Tian Li, Virginia Smith, Jeff Bilmes, and Tianyi Zhou (2025). [Many-Objective Multi-Solution Transport](https://arxiv.org/abs/2403.04099). *International Conference on Learning Representations*. ^ref-li-2025-most

[56] Xi Lin, Zhiyuan Yang, Qingfu Zhang, and Sam Kwong (2020). [Controllable Pareto Multi-Task Learning](https://arxiv.org/abs/2010.06313). arXiv:2010.06313. ^ref-lin-2020-cpmtl

[57] Long P. Hoang, Dung D. Le, Tran Anh Tuan, and Tran Ngoc Thang (2023). [Improving Pareto Front Learning via Multi-Sample Hypernetworks](https://ojs.aaai.org/index.php/AAAI/article/view/25953). *Proceedings of the AAAI Conference on Artificial Intelligence*, **37**(7), 7875–7883. ^ref-hoang-2023-phn-hvi

[58] Soumyajit Gupta, Gurpreet Singh, Raghu Bollapragada, and Matthew Lease (2022). [Learning a Neural Pareto Manifold Extractor with Constraints](https://proceedings.mlr.press/v180/gupta22a.html). *Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence*, PMLR **180**, 749–758. ^ref-gupta-2022-suhnpf

[59] Tran Anh Tuan, Nguyen Viet Dung, and Tran Ngoc Thang (2024). [A Hyper-Transformer Model for Controllable Pareto Front Learning with Split Feasibility Constraints](https://arxiv.org/abs/2402.05955). arXiv:2402.05955. ^ref-tuan-2024-hyper-trans

[60] Michael Ruchte and Josif Grabocka (2021). [Scalable Pareto Front Approximation for Deep Multi-Objective Learning](https://arxiv.org/abs/2103.13392). *IEEE International Conference on Data Mining*. ^ref-ruchte-2021-cosmos

[61] Alexey Dosovitskiy and Josip Djolonga (2020). [You Only Train Once: Loss-Conditional Training of Deep Networks](https://openreview.net/forum?id=HyxY6JHKwr). *International Conference on Learning Representations*. ^ref-dosovitskiy-2020-yoto

[62] Dripta S. Raychaudhuri, Yumin Suh, Samuel Schulter, Xiang Yu, Masoud Faraki, Amit K. Roy-Chowdhury, and Manmohan Chandraker (2022). [Controllable Dynamic Multi-Task Architectures](https://openaccess.thecvf.com/content/CVPR2022/html/Raychaudhuri_Controllable_Dynamic_Multi-Task_Architectures_CVPR_2022_paper.html). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 10955–10964. ^ref-raychaudhuri-2022-dymu

[63] Nikolaos Dimitriadis, Pascal Frossard, and François Fleuret (2023). [Pareto Manifold Learning: Tackling Multiple Tasks via Ensembles of Single-Task Models](https://proceedings.mlr.press/v202/dimitriadis23a.html). *Proceedings of the 40th International Conference on Machine Learning*, PMLR **202**, 8015–8052. ^ref-dimitriadis-2023-pamal

[64] Weiyu Chen and James Kwok (2024). [Efficient Pareto Manifold Learning with Low-Rank Structure](https://proceedings.mlr.press/v235/chen24x.html). *Proceedings of the 41st International Conference on Machine Learning*, PMLR **235**, 7015–7032. ^ref-chen-2024-lorpman

[65] Weiyu Chen and James Kwok (2024). [You Only Merge Once: Learning the Pareto Set of Preference-Aware Model Merging](https://arxiv.org/abs/2408.12105v1). arXiv:2408.12105v1. Revised as [Pareto Merging: Multi-Objective Optimization for Preference-Aware Model Merging (2025)](https://arxiv.org/abs/2408.12105). ^ref-chen-2024-yomo

[66] Yifan Zhong, Chengdong Ma, Xiaoyuan Zhang, Ziran Yang, Haojun Chen, Qingfu Zhang, Siyuan Qi, and Yaodong Yang (2024). [Panacea: Pareto Alignment via Preference Adaptation for LLMs](https://proceedings.neurips.cc/paper_files/paper/2024/hash/89f39d0b3d49a47606a165eefba2778c-Abstract-Conference.html). *Advances in Neural Information Processing Systems*, **37**. ^ref-zhong-2024-panacea

[67] Anke Tang, Li Shen, Yong Luo, Shiwei Liu, Han Hu, and Bo Du (2024). [Towards Efficient Pareto Set Approximation via Mixture of Experts Based Model Fusion](https://arxiv.org/abs/2406.09770v1). arXiv:2406.09770v1. ^ref-tang-2024-moe-fusion

[68] Nikolaos Dimitriadis, Pascal Frossard, and François Fleuret (2025). [Pareto Low-Rank Adapters: Efficient Multi-Task Learning with Preferences](https://arxiv.org/abs/2407.08056). *International Conference on Learning Representations*. ^ref-dimitriadis-2025-palora

[69] Pingchuan Ma, Tao Du, and Wojciech Matusik (2020). [Efficient Continuous Pareto Exploration in Multi-Task Learning](https://proceedings.mlr.press/v119/ma20a.html). *Proceedings of the 37th International Conference on Machine Learning*, PMLR **119**, 6522–6531. ^ref-ma-2020-exploration

[70] Achille Messac, A. Ismail-Yahaya, and Christopher A. Mattson (2003). [The Normalized Normal Constraint Method for Generating the Pareto Frontier](https://doi.org/10.1007/s00158-002-0276-1). *Structural and Multidisciplinary Optimization*, **25**, 86–98. ^ref-messac-2003-nnc

[71] George Mavrotas (2009). [Effective Implementation of the ε-Constraint Method in Multi-Objective Mathematical Programming Problems](https://doi.org/10.1016/j.amc.2009.03.037). *Applied Mathematics and Computation*, **213**(2), 455–465. ^ref-mavrotas-2009-augmecon

[72] F. W. Gembicki and Y. Y. Haimes (1975). [Approach to Performance and Sensitivity Multiobjective Optimization: The Goal Attainment Method](https://doi.org/10.1109/TAC.1975.1101105). *IEEE Transactions on Automatic Control*, **20**(6), 769–771. ^ref-gembicki-haimes-1975

[73] Victor Pereyra, Michael A. Saunders, and Jose Castillo (2013). [Equispaced Pareto Front Construction for Constrained Bi-Objective Optimization](https://doi.org/10.1016/j.mcm.2010.12.044). *Mathematical and Computer Modelling*, **57**(9–10), 2122–2131. First available online in 2011. ^ref-pereyra-2013-continuation
