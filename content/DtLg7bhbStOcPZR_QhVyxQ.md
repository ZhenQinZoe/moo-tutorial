# 5. Applications 

[TOC]

Since Sener & Koltun (2018) cast multi-task learning as multi-objective optimization (MOO), gradient-based MOO has been applied across search and recommendation, speech, vision, language, Networking, reinforcement learning, and more recently, multi-reward alignment of large language models (see Chen et al., 2025 for a survey).

## 5.1. Multi-Objective Search and Recommender Systems

Web-scale search and recommendation are where these methods have had their most consequential industrial uptake, for two reasons: the objectives are business and user outcomes that genuinely conflict — clicks, conversions, dwell time, gross merchandise volume (GMV), pricing, delivery speed, content quality, and the avoidance of explicit negative feedback — and are already co-trained in shared multi-task rankers on billions of examples, so gradient conflict arises by default; and these systems have the A/B-testing infrastructure to check whether a training-time Pareto improvement survives online. The result is a continuous line of deployments since 2019 whose lineage is traceable. PE-LTR (Alibaba, RecSys 2019) first brought KKT-based weight solving into a production e-commerce ranker and validated it with a live A/B test on CTR and GMV (Lin et al., 2019a); PAPERec (Tencent, WWW 2021) personalized the objective weights per user by rewarding the norm of the weighted gradient sum (Xie et al., 2021); and PMORS (Tencent, CIKM 2024) applied a Frank–Wolfe MGDA solver at the pre-ranking stage to trade off consistency with the downstream ranker against recency-weighted suppression of disliked content, reporting +1.45% GMV and +0.89% CTR after deployment on WeChat Channels Ads in May 2023 (Jin et al., 2024). A second line attacks the cost of gradient balancing itself. MultiBalance (Meta, 2024) balances per-task gradients at the shared feature representation rather than over billions of shared parameters, cutting the training-throughput overhead of MGDA-style balancing from 70–80% to 0.42% while improving normalized entropy by up to 0.738% on production feeds models (He et al., 2024); GradCraft (Kuaishou, KDD 2024) combines gradient-magnitude alignment with global conflict projection and reports +0.505% watch time in a one-week online experiment (Bai et al., 2024). Closely related gradient-based formulations enforce business guardrails through augmented-Lagrangian learning to rank in Amazon product search (Momma et al., 2020; Garakani et al., 2022), robust min–max objective mixing in query autocomplete (Singh et al., 2023), preference-conditioned policy gradients for controllable re-ranking at Taobao (Chen et al., 2023), and constrained actor–critic learning for sequential objectives at Kuaishou (Cai et al., 2023). 

### 5.1.1. Multi-Objective Recommender Systems
Industrial recommenders are cascades: a retrieval stage produces thousands of candidates, a light pre-ranking model trims them, a heavy ranking model scores the survivors, and a re-ranking or blending stage composes the final list (Jin et al., 2024, Fig. 1). The ranking model is almost always a multi-task network: a shared bottom maps categorical features through large embedding tables and numerical features through small towers, an interaction module produces a *shared representation*, and one head per user behavior — click, like, share, comment, watch time, purchase — is trained on top of it (Figure 1). Two facts about this architecture shape how gradient-based MOO is used. First, the embedding tables dominate the parameter count: in the three public production models He et al. (2024) tabulate, the categorical module holds 5–18 billion parameters against 1 million to 1.7 billion for everything else, so anything that needs per-task gradients over the shared parameters is expensive. Second, the behaviors genuinely conflict — on Taobao data the CTR and GMV losses of a ranker are negatively correlated (Pearson −0.34, Lin et al., 2019a, Fig. 1), and in short-video feeds a model that is better at predicting views is routinely worse at predicting interactions. Gradient-based MOO therefore enters at three distinct layers: the loss weights of the shared ranking model, the objective trade-off of a particular cascade stage, and the policy that produces the final list.


![multibalance_fig1_mtl_framework](https://hackmd.io/_uploads/ry7AGzp2fe.png)


*Figure 1. The general multi-task ranking architecture in industrial recommendation: shared bottom layers (categorical and numerical feature modules plus an interaction module) feed a shared representation consumed by one head per task. Reproduced from He et al. (2024), Fig. 1.*

**KKT-based scalarization at the stage level.** The lineage that starts with PE-LTR (Lin et al., 2019a) keeps the familiar weighted-sum loss $\sum_i \omega_i L_i(\theta)$ but solves for the weights rather than tuning them. At each step the weights are the minimizer of


$$
\min_{\omega}\ \Big\|\sum_{i=1}^{K}\omega_i \nabla_\theta L_i(\theta)\Big\|_2^2
\quad\text{s.t.}\quad \sum_i \omega_i = 1,\ \ \omega_i \ge c_i ,
$$

which is the MGDA min-norm problem (Désidéri, 2012) with lower bounds $c_i$ that encode business priorities (for an e-commerce platform, a floor on the GMV weight). PE-LTR solves it in two steps — a relaxed quadratic program followed by a non-negative least-squares correction — and generates a Pareto frontier by sweeping the bounds; its three-day online test against a price·CTR·CVR heuristic reported CTR +13.80% and GMV +3.62%. PAPERec (Xie et al., 2021) observes that a single global $\omega$ is wrong for a heterogeneous user base and replaces it with a per-user weight vector emitted by an actor network, trained with a critic whose reward is $-\|\sum_j \omega_j \nabla_\theta L_j\|_2$; deployed in WeChat Top Stories, it improved CTR by 1.11% and item dwell time by 3.26% over fixed-weight fusion. PMORS (Jin et al., 2024) moves the same solver to the pre-ranking stage, where the two objectives are *consistency* with the downstream ranker (a pairwise LTR loss against the ranker's scores) and *recency* — a penalty $L_{FG} = -\sum_i w_i \log(1-p(\mathbf{x}_i))$ on candidates that users recently skipped, with weights $w_i$ obtained by Ebbinghaus-style decay $e^{-t/S}$ of fast-slip rates over windows of 10 minutes, 3 hours, 1 day and 7 days (Figure 2). The min-norm weights are computed by Frank–Wolfe at every step. The ablation is the most instructive part of the paper: sweeping a fixed $\alpha_{LTR}\in\{0.5,\dots,1.0\}$ traces the expected consistency–recency trade-off, but the adaptive solver lands *off* that curve, with KuaiRec CTR +11.3% and Recall@10_1 +35.2% over the pure-LTR model, versus +6.6% and +22.2% for the best fixed weight. Online on WeChat Channels Ads the system raised GMV by 1.45% and CTR by 0.89% while lowering the fast-slip rate by 0.63%, and has served main traffic since May 2023.


![pmors_fig2_framework](https://hackmd.io/_uploads/SJDe7Ga2zg.svg)


*Figure 2. PMORS. (a) Conventional pre-ranking: the pre-ranking model (orange) is trained for consistency with a fixed, pretrained ranking model (green). (b) PMORS adds a forgetting model (blue) that scores a recency objective from decayed negative feedback, and a Pareto optimization solver that aggregates the two losses and returns the preference weights that drive the model toward Pareto efficiency. Reproduced from Jin et al. (2024), Fig. 2.*

**Gradient balancing inside the shared model.** The second use of gradient MOO targets negative transfer among the heads of the ranking model itself, and here the cost structure above becomes decisive. Textbook MGDA or PCGrad needs the gradient of every task loss with respect to the shared parameters, i.e. $M$ backward passes through the embedding tables; on Meta's production models this reduced training throughput by 82% when applied to the interaction module alone and could not be run on the categorical module at all (He et al., 2024, Table 2). MultiBalance sidesteps the backbone: the shared representation is copied $M$ times in the forward pass, one copy per head, so that a single backward pass yields each task's gradient with respect to its copy (Figure 3). The min-norm weights are then solved over these representation gradients — with a regularizer pulling toward a prior $\lambda_0$ and an exponential moving average of per-task gradient norms to tame mini-batch variance — and the weighted sum is back-propagated once through the shared bottom. Theorem 1 of the paper shows that an $\epsilon$-stationary point of the representation-level problem is an $(\ell\epsilon+\delta)$-Pareto-stationary point in parameter space, where $\delta$ bounds the residual contributed by the categorical module on the batch. The result is a 0.42% throughput cost, a 0.738% normalized-entropy gain on a feeds model, and a 0.091% CVR gain on ads — alongside a 0.032% CTR *loss* on ads, which the authors accept as the price of the CVR improvement. Removing the moving average flips the feeds-B gain to −0.122%, a reminder that stochastic gradient-combination weights need stabilization (Zhou et al., 2022). GradCraft (Bai et al., 2024) attacks the same problem from the direction side: it first rescales every task gradient toward the largest norm (a hyper-parameter $\tau$ controls how far), then projects each gradient against *all* conflicting gradients simultaneously rather than pairwise as in PCGrad (Figure 4). On Kuaishou's production traffic this delivered +0.505% watch time and +1.746% shares in a week-long A/B test against the deployed multi-task baseline.




![multibalance_fig2b_backward](https://hackmd.io/_uploads/H1Hz7fT3fe.png)


*Figure 3. MultiBalance backward pass. Each head back-propagates into its own copy of the shared representation; MultiBalance combines the \(M\) representation gradients into one aggregated gradient, which then flows once through the shared bottom layers. Reproduced from He et al. (2024), Fig. 2(b).*


![gradcraft_fig1_overview](https://hackmd.io/_uploads/r1b77f6hMg.png)


*Figure 4. GradCraft. Task gradients \(g_i\) are first magnitude-aligned to the largest norm (\(\hat g_i\)), then each is projected against the set of gradients it conflicts with (\(\tilde g_i\)) before aggregation. Reproduced from Bai et al. (2024), Fig. 1.*

**MOO at the policy layer.** Once the ranker exists, a further set of trade-offs — diversity, cold-start exposure, freshness, long-term retention — lives in the list-construction policy, and here the gradient methods are policy gradients. CMR (Chen et al., 2023) conditions a Taobao re-ranker on a preference vector through a hypernetwork and trains it with REINFORCE on the preference-weighted list reward, so that operators can move along the trade-off at serving time without retraining; its online experiments show each metric responding roughly monotonically to its own preference weight (Figure 5). TSCAC (Cai et al., 2023) treats watch time as the primary return and interactions as constraints in a constrained MDP, trains auxiliary actor–critics first and then the main policy under a proximity constraint; fully launched at Kuaishou, it gained 0.379% watch time and 3.376% shares while comments fell 0.619%. Two lessons carry across all three layers. The weights that balance training losses, the weights that fuse predicted scores at serving time, and the business preference over outcomes are different objects — PAPERec is interesting precisely because it ties the first to the third — and an MOO method must be evaluated at the layer it actually changes. And a Pareto-stationary training objective does not imply a Pareto-optimal online operating point: every paper above reports at least one secondary metric that moved the wrong way.



![cmr_fig5_online_vs_preference](https://hackmd.io/_uploads/HJyNmGphfg.png)


*Figure 5. Controllability online: relative change in each business metric as its own preference weight is varied in the CMR re-ranker (blue), with a linear fit (orange). Reproduced from Chen et al. (2023), Fig. 5.*



### 5.1.2. Multi-Objective Search Ranking

Search ranking shares the multi-objective structure of recommendation — a primary relevance or purchase objective plus secondary objectives such as revenue, product quality, defect rate, delivery speed, brand mix, and, in autocomplete, the quality of the suggested query — but differs in three respects that change how gradient-based MOO is applied. The dominant production rankers are still gradient-boosted trees in the LambdaMART family, so gradients are taken with respect to the *scores* $\mathbf{s}$ rather than model parameters and each boosting round fits a tree to a combined pseudo-gradient $g(\mathbf{s}) = \sum_k \alpha_k \nabla_{\mathbf{s}} C_k(\mathbf{s})$; the target metrics (NDCG, purchase NDCG) are non-differentiable and are optimized through metric-weighted pairwise surrogates; and the requirements arrive from stakeholders as bounds — "improve relevance as much as possible while keeping revenue within −1% of production" — which makes a constrained formulation more natural than a scalarization. Early work at Microsoft already reflected this: Svore et al. (2011) modified the lambda gradients so that a click-derived objective could improve only where it did not hurt human-judged relevance, a lexicographic rather than weighted combination.


**Constraints as the interface to the business.** AL-LambdaMART (Momma et al., 2020) made the constrained view explicit. With $C^{pm}$ the primary surrogate cost and $C^t$ the secondary costs, each rescaled so that a bound $b^t = 0.9$ means "no worse than 10% below baseline", the problem is

$$
\min_{\mathbf{s}}\ C^{pm}(\mathbf{s}) \quad\text{s.t.}\quad C^{t}(\mathbf{s}) \le b^{t},\ t=1,\dots,T,
$$

solved by an augmented Lagrangian whose dual variables are updated at every boosting iteration, $\alpha^t_k = \max\!\big(0,\ \mu\,(C^t(\mathbf{s}) - b^t) + \alpha^t_{k-1}\big)$, so that multipliers grow only on constraints that are currently violated. The practical contribution is a "one-shot" protocol: a sensitivity sweep on each secondary objective alone fixes its bound, after which a single constrained run produces the model. On MSLR-10K this reached the target of ≥+1% on all five secondary objectives with relevance held within −1% in one build, where a linear-weighting search needed more than 200 full models to find a single feasible one (Table 2). On Amazon's product-search data the same procedure gained 2–4% on three secondary objectives with the purchase objective statistically flat, and the paper states that an online A/B test confirmed the behavior, although no online numbers are disclosed.


*Table 2. AL-LambdaMART versus linear weighting on MSLR-10K (percentage gain over the unconstrained baseline). From Momma et al. (2020), Table 1.*

| Model | Relevance | PageRank | QualityScore | QualityScore2 | UrlClick | UrlDwellTime | Full model builds |
|---|---|---|---|---|---|---|---|
| AL-LambdaMART | −0.82 | +10.38 | +1.16 | +1.26 | +10.87 | +12.06 | 1 |
| Linear weighting | −0.89 | +4.83 | +1.00 | +1.34 | +13.60 | +15.30 | 200+ |

Garakani et al. (2022) supply the online evidence the 2020 paper withheld. A high-precision BERT cross-encoder, too expensive to score every candidate online, is run offline to produce query–product relevance labels for the full training set; AL-LambdaMART then maximizes relevance NDCG subject to a revenue-surrogate bound set to "at least flat" against the production model. Two online experiments in one marketplace improved overall NDCG@16 by 0.29% and 0.49% (p < 0.05), with gains on both head/torso and tail queries and revenue reported flat (Table 3). The design is worth teaching as a pattern: expensive supervision is paid for once offline, the business guardrail is written into the training problem as a constraint, and the online test — not the satisfied surrogate constraint — is what establishes that revenue did not move.


*Table 3. Online relevance quality (NDCG@16) of the constrained ranker versus production, by query-frequency segment. From Garakani et al. (2022), Table 5; all entries p < 0.05.*

| Experiment | Overall | Head + torso | Tail |
|---|---|---|---|
| 1 | +0.29% | +0.19% | +0.44% |
| 2 | +0.49% | +0.52% | +0.42% |

**Preference-directed and robust combinations.** When the number of objectives grows, bounds become hard to set and the preference-directed methods of the MOO literature become relevant. Mahapatra et al. (2023) bring EPO, weighted-Chebyshev MGDA, augmented-Lagrangian and dynamic-barrier combinations into the boosting loop by computing all of them on score-space gradients and fitting trees to the result, and evaluate on Amazon labels for purchase, historical purchase, relevance, brandedness and delivery speed. Their main practical finding is that the per-iteration combination coefficients fluctuate badly in a tree ensemble and must be smoothed with a moving average before the preference is actually tracked — the boosting analogue of MultiBalance's EMA. The evaluation is offline; the companion MO-LightGBM library (Dong & Momma, 2025) makes the twelve variants reproducible. A different route is taken by Singh et al. (2023) for query autocomplete, where a neural pairwise ranker must balance customer-behavior ranking against query quality. Rather than fix the mixture, they train with ALMO (Cortes et al., 2020), a min–max formulation that optimizes against the worst admissible mixture of the two losses while the mixture weights are learned jointly. A four-week A/B test on one million prefixes raised GMV by 0.26% and shopping-item diversity by 0.41%; the public AOL experiment in the same paper shows an MRR decline, so the robustness is against the mixture, not a guarantee that every metric improves.

Three observations distinguish the search case from recommendation. Because the optimization object is a score vector rather than a parameter vector, every gradient-MOO method — MGDA, EPO, augmented Lagrangian, ALMO — ports to tree ensembles unchanged, which is why Amazon could compare so many of them on one platform. Because requirements come as guardrails, constrained formulations dominate, and the main skill they demand is choosing bounds on surrogate losses that correspond to acceptable movements in online KPIs. And the public online evidence is thinner: Amazon confirms online behavior but withholds magnitudes in two of the three papers, so the tutorial should present search as the domain where the *formulation* is best understood and recommendation as the domain where the *online evidence* is richest.



### 5.1.3. Comparison table

We next summarize Pareto-stationarity and gradient-conflict solvers (MGDA / KKT / projection) in Table 4. 



*Table 4. **Evidence labels.** **Deployed** = explicit production launch / serving main traffic. **Online** = live controlled experiment reported; full rollout not stated. **Industrial offline** = company-scale data and production models, no user-facing A/B metrics. **Research** = public benchmark only.*
| Paper | Venue | Company · surface | Objectives | Gradient-MOO mechanism | Online evidence (verified) | Reported outcome |
|---|---|---|---|---|---|---|
| **PE-LTR** — A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation [Lin et al.](https://doi.org/10.1145/3298689.3346998) | RecSys 2019 | Alibaba · e-commerce recommendation (Taobao-scale) | CTR and GMV (price-weighted purchase loss) | KKT-condition scalarization: min-norm weights with **lower bounds ωᵢ ≥ cᵢ**; two-step solver (relaxed QP → non-negative least squares) | **Online.** 3-day A/B vs. LETORIF (price·CTR·CVR), all p<0.01 (§6.4, Table 3); full rollout not stated | CTR **+13.80%**, IPV **+23.76%**, PAY **+20.09%**, GMV **+3.62%** |
| Multi-Gradient Descent for Multi-Objective Recommender Systems [Milojkovic et al.](https://arxiv.org/abs/2001.00846) | AAAI WICRS workshop 2020 | — (method paper) | Accuracy, revenue, quality-type objectives | Classical **MGDA** with gradient normalization | **Research** | Algorithm demonstration; no industrial deployment |
| **PAPERec** — Personalized Approximate Pareto-Efficient Recommendation [Xie et al.](https://doi.org/10.1145/3442381.3450039) | WWW 2021 | Tencent · WeChat Top Stories ranking | CTR and item dwell time | Per-user objective weights from an actor–critic whose reward is **−‖Σⱼ ωⱼ∇Lⱼ‖₂** (Pareto-stationarity surrogate); KKT-based scalarization | **Deployed** >3 months; 7-day online evaluation, >6M users (§3.5, §4.5, Table 3) | CTR **+1.11%**, system dwell time **+1.33%**, item dwell time **+3.26%**, click-type diversity **+15.34%** |
| Multi-Label Learning to Rank through Multi-Objective Optimization [Mahapatra et al.](https://doi.org/10.1145/3580305.3599870) | KDD 2023 | Amazon / NUS · commerce LTR (proprietary labels) | Purchase, relevance, brandedness, delivery speed, … | Combine **score-space gradients** with EPO, WC-MGDA, augmented Lagrangian; coefficient smoothing; then fit boosted trees | **Industrial offline** only | Better preference alignment and frontier coverage than linear scalarization; no live lift |
| **GradCraft** — Elevating Multi-task Recommendations through Holistic Gradient Crafting [Bai et al.](https://doi.org/10.1145/3637528.3671585) | KDD 2024 | Kuaishou (with USTC) · short-video multi-task ranking | Viewing tasks (effective view, long view) and interaction tasks (like, follow, forward, …) | **Magnitude alignment** to max gradient norm (τ) followed by **global projection** against all conflicting task gradients at once (generalizes PCGrad) | **Online.** 1-week A/B on production traffic, >15M users, vs. Kuaishou's production MTL baseline (§4.5, Table 4); deployment not stated | Watch time **+0.505%**, effective views **+0.950%**, shares **+1.746%** |
| **PMORS** — Pareto-based Multi-Objective Recommender System with Forgetting Curve [Jin et al.](https://doi.org/10.1145/3627673.3680080) | CIKM 2024 (pp. 4603–4611) | Tencent · WeChat Channels Ads · short-video ad **pre-ranking** | Consistency with downstream ranker (LTR loss) vs. recency-weighted fast-slip penalty (Ebbinghaus forgetting curve) | MGDA min-norm weights solved by **Frank–Wolfe** (Alg. 1); weights re-solved each step and used to aggregate losses | **Deployed.** A/B test on WeChat Channels, May 2023, vs. previous online LTR model; "now serving the main traffic" (§4.3, Table 3) | GMV **+1.45%**, CTR **+0.89%**, Recall@10_1 **+3.0%**, FastSlipRate **−0.63%**, FastSlipAction **−0.50%** |
| **MultiBalance** — Multi-Objective Gradient Balancing in Industrial-Scale Multi-Task Recommendation System [He et al.](https://arxiv.org/abs/2411.11871) | arXiv, Nov 2024 (preprint) | Meta · ads and feeds MTL ranking models | Ads: CTR, CVR, Conv\|imp; feeds: undisclosed engagement tasks | Per-task gradients taken w.r.t. the **shared representation** (one backward pass); weights learned by projected update toward min-norm with prior λ₀; **EMA of gradient magnitudes** for stability | **Industrial offline.** Production-scale models, billions of samples; eval NE and training QPS only — no user-facing A/B (Tables 2–3) | Feeds NE gain **+0.476% / +0.738%**; ads CVR NE **+0.091%**, CTR NE **−0.032%**; training QPS drop **0.42%** vs. **82%** for parameter-level MGDA/MoCo |


### References
- **Désidéri, J.-A. (2012).** *Multiple-Gradient Descent Algorithm (MGDA) for Multiobjective Optimization.* Comptes Rendus Mathématique, 350(5–6), 313–318.
- **Sener, O., & Koltun, V. (2018).** *Multi-Task Learning as Multi-Objective Optimization.* NeurIPS 2018. [Paper](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
- **Lin, X., Zhen, H.-L., Li, Z., Zhang, Q., & Kwong, S. (2019b).** *Pareto Multi-Task Learning.* NeurIPS 2019. [arXiv:1912.12854](https://arxiv.org/abs/1912.12854) · `lin2019paretomtl` — note the author-name collision with PE-LTR (Xiao Lin, 2019a)
- **Mahapatra, D., & Rajan, V. (2020).** *Multi-Task Learning with User Preferences: Gradient Descent with Controlled Ascent in Pareto Optimization (EPO).* ICML 2020. [Paper](https://proceedings.mlr.press/v119/mahapatra20a.html)
- **Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020).** *Gradient Surgery for Multi-Task Learning (PCGrad).* NeurIPS 2020. [Paper](https://papers.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
- **Navon, A., Shamsian, A., Achituve, I., et al. (2022).** *Multi-Task Learning as a Bargaining Game (Nash-MTL).* ICML 2022. [Paper](https://proceedings.mlr.press/v162/navon22a.html)
- **Chen, W., Lin, B., Zhang, X., Lin, X., et al. (2025).** *Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond.* arXiv preprint. [arXiv:2501.10945](https://arxiv.org/abs/2501.10945)
- **Lin, X., Chen, H., Pei, C., et al. (2019a).** *A Pareto-Efficient Algorithm for Multiple Objective Optimization in E-Commerce Recommendation.* RecSys 2019, pp. 20–28. [DOI](https://doi.org/10.1145/3298689.3346998) · `lin2019pareto`
- **Milojkovic, N., Antognini, D., Bergamin, G., Faltings, B., & Musat, C. (2020).** *Multi-Gradient Descent for Multi-Objective Recommender Systems.* AAAI WICRS workshop 2020. [arXiv:2001.00846](https://arxiv.org/abs/2001.00846)
- **Xie, R., Liu, Y., Zhang, S., Wang, R., Xia, F., & Lin, L. (2021).** *Personalized Approximate Pareto-Efficient Recommendation.* WWW 2021, pp. 3839–3849. [DOI](https://doi.org/10.1145/3442381.3450039) · `xie2021personalized`
- **Mahapatra, D., Dong, C., Chen, Y., & Momma, M. (2023).** *Multi-Label Learning to Rank through Multi-Objective Optimization.* KDD 2023. [DOI](https://doi.org/10.1145/3580305.3599870)
- **Bai, Y., Zhang, Y., Feng, F., et al. (2024).** *GradCraft: Elevating Multi-task Recommendations through Holistic Gradient Crafting.* KDD 2024. [DOI](https://doi.org/10.1145/3637528.3671585) · [arXiv:2407.19682](https://arxiv.org/abs/2407.19682) · `bai2024gradcraft`
- **Jin, J., Zhang, Z., Li, Z., Gao, X., Yang, X., Xiao, L., & Jiang, J. (2024).** *Pareto-based Multi-Objective Recommender System with Forgetting Curve.* CIKM 2024, pp. 4603–4611. [DOI](https://doi.org/10.1145/3627673.3680080) · [arXiv:2312.16868](https://arxiv.org/abs/2312.16868) · `jin2024pareto`
- **He, Y., Chen, X., Xu, J., et al. (2024).** *MultiBalance: Multi-Objective Gradient Balancing in Industrial-Scale Multi-Task Recommendation System.* arXiv preprint, Nov 2024. [arXiv:2411.11871](https://arxiv.org/abs/2411.11871) · `he2024multibalance`


## 5.2. Multilingual Speech Recognition and Translation

Multilingual speech processing places several learning objectives on the same acoustic encoder. Automatic speech recognition (ASR) produces a transcript in the source language, speech-to-text translation (ST) produces text in another language, and self-supervised learning extracts acoustic structure from unlabeled recordings. Sharing an encoder allows these tasks to transfer information, but also couples their updates: the features useful for predicting a transcript need not yield the best translation, and supervised adaptation can change representations learned from unlabeled speech. In this setting, MOO enters through three concrete choices: how to balance language–task gradients, how to preserve acoustic representations while adapting the model, and how to steer the resulting model toward a requested task trade-off.

### 5.2.1. Constrained Multilingual Learning

[M²ASR (Saif et al., 2024)](https://www.isca-archive.org/interspeech_2024/saif24_interspeech.html) uses a shared Conformer encoder with separate language–task output heads. Its CoVoST configuration has 22 recognition heads and 21 translation heads. Both kinds of supervised head use connectionist temporal classification (CTC), while contrastive predictive coding (CPC) supplies a self-supervised acoustic loss. Thus, this early formulation combines recognition, translation, and representation learning within one model; its translation loss should be distinguished from the cross-entropy objective used in the later Objective Soups experiments.

Let the positive integers $d_{\mathrm{enc}}$ and $d_{\mathrm{head}}$ count encoder and output-head parameters, respectively, and let $n=d_{\mathrm{enc}}+d_{\mathrm{head}}$ be the total number of trainable parameters. Let $\theta\in\mathbb R^{d_{\mathrm{enc}}}$ denote the shared encoder parameters and $\phi\in\mathbb R^{d_{\mathrm{head}}}$ the output-head parameters, so that the full decision vector is $x=(\theta,\phi)\in\mathbb R^n$. Let $M$ be the number of supervised language–task objectives and, for each index $i\in\{1,\ldots,M\}$, let $f_i:\mathbb R^n\to\mathbb R$ be the corresponding training loss. Define their objective vector as $F(x)=(f_1(x),\ldots,f_M(x))^\top\in\mathbb R^M$. Let $u:\mathbb R^{d_{\mathrm{enc}}}\to\mathbb R$ be the self-supervised acoustic loss, $u_\star$ its minimum value, and $\varepsilon\geq0$ the permitted gap from that minimum. The constrained MOO formulation is

$$
\min_{x\in\mathbb R^n}\ F(x)
\quad\text{subject to}\quad u(\theta)-u_\star\leq\varepsilon.
$$

The constraint limits degradation of the self-supervised acoustic objective during supervised adaptation. Let $\eta\geq0$ be a penalty coefficient and define the penalized objective $\widetilde f_i(x)=f_i(x)+\eta[u(\theta)-u_\star]$. Let $\lambda=(\lambda_1,\ldots,\lambda_M)\in\mathbb R^M$ contain nonnegative supervised-objective weights summing to one, and let $\nabla_\theta$ denote the gradient with respect to the encoder. Then

$$
\sum_{i=1}^{M}\lambda_i\nabla_\theta\widetilde f_i
=\sum_{i=1}^{M}\lambda_i\nabla_\theta f_i+\eta\nabla_\theta u.
$$

Figure 6 illustrates the shared model and this constrained encoder update.

![Shared speech model and constrained encoder update](https://hackmd.io/_uploads/Syh0hsBpGe.svg)

*Figure 6. Shared speech model and constrained encoder update. A shared acoustic encoder supports automatic speech recognition (ASR), speech-to-text translation (ST), and acoustic self-supervision. The constrained formulation balances supervised encoder gradients and adds a separately weighted acoustic gradient. Solid arrows show forward computation; dashed arrows show encoder-gradient contributions. Each supervised head also receives its own task gradient, omitted here. M²ASR uses connectionist temporal classification (CTC) for both supervised tasks; Objective Soups uses CTC for ASR and cross-entropy for ST. Original schematic based on [Saif et al. (2024)](https://www.isca-archive.org/interspeech_2024/saif24_interspeech.html) and [Saif et al. (2025)](https://proceedings.nips.cc/paper_files/paper/2025/hash/188584a9e175a84cdf6b09fc6843f308-Abstract-Conference.html).*

Adaptive weights redistribute influence among supervised tasks, while the acoustic gradient retains coefficient $\eta$. The constant $u_\star$ disappears from the update, and each output head receives its own supervised gradient. A finite penalty biases training toward the acoustic constraint; it does not by itself certify feasibility. These are coupled first-order updates, rather than an exact solution of the lower-level problem at every iteration. [M²ASR, Section 3](https://www.isca-archive.org/interspeech_2024/saif24_interspeech.pdf)

The reported static and dynamic variants compare weighting strategies within the joint-training framework. With its 58.4-million-parameter encoder, M²ASR reports high-resource-language average ASR word error rates of 30.0% for pre-training followed by fine-tuning, 27.5% for static joint weighting, and 26.4% for dynamic weighting. Its corresponding translation-output word error rates are 34.9%, 32.7%, and 31.9%. Word error rate (WER) measures word-level substitutions, deletions, and insertions relative to reference length; lower values are better. These averages concern the reported language subsets, and the translation measurements are WER rather than BLEU. [M²ASR, Tables 2–3](https://www.isca-archive.org/interspeech_2024/saif24_interspeech.pdf)

### 5.2.2. Hierarchical Multi-Objective Training

[Objective Soups (Saif et al., 2025)](https://proceedings.nips.cc/paper_files/paper/2025/hash/188584a9e175a84cdf6b09fc6843f308-Abstract-Conference.html) extends this constrained design by comparing three recipes for multilingual speech processing (MSP): vectorized single-level MSP (VS-MSP), vectorized objectives with a lower-level constraint (VC-MSP), and vectorized multilevel MSP (VM-MSP). Its principal Conformer experiment uses CTC recognition heads, a Transformer translation component trained with cross-entropy, and CPC self-supervision. The five recognition objectives cover English, French, German, Spanish, and Catalan; four additional objectives translate the non-English languages into English. Thus the encoder must accommodate nine supervised losses, in addition to the acoustic objective.

**Single-level and constrained formulations.** For any positive integer $r$, let $\Delta_r$ denote the set of nonnegative real vectors of length $r$ whose entries sum to one. Let the nonnegative integer $t$ index training iterations, let $\alpha_t>0$ be the encoder step size, and let $\eta_t\geq0$ be the current acoustic penalty. Subscript $t$ denotes the current parameters or weights, and $\lambda_{\mathrm u,t}$ denotes the dynamic weight of self-supervision in VS-MSP. Evaluate supervised gradients at $x_t=(\theta_t,\phi_t)$ and the acoustic gradient at $\theta_t$. The first two formulations differ as follows:

$$
\begin{aligned}
\text{VS-MSP:}\quad
\theta_{t+1}
&=\theta_t-\alpha_t\left[
\sum_{i=1}^{M}\lambda_{i,t}\nabla_\theta f_i
+\lambda_{\mathrm u,t}\nabla_\theta u\right],
&& (\lambda_{1,t},\ldots,\lambda_{M,t},\lambda_{\mathrm u,t})\in\Delta_{M+1},\\
\text{VC-MSP:}\quad
\theta_{t+1}
&=\theta_t-\alpha_t\left[
\sum_{i=1}^{M}\lambda_{i,t}\nabla_\theta f_i
+\eta_t\nabla_\theta u\right],
&&\lambda_t\in\Delta_M.
\end{aligned}
$$

VS-MSP places every loss in one vector objective, so acoustic learning competes for weight with recognition and translation. VC-MSP balances the supervised objectives and retains acoustic learning through a separately controlled penalty. The difference changes the role of self-supervision, even though both methods use a weighted encoder update.

**Task-based hierarchical optimization.** VM-MSP additionally groups supervised objectives into optimization levels. Let $\mathcal A$ and $\mathcal T$ partition the supervised indices into recognition and translation tasks. Let $a_{i,t}$ for $i\in\mathcal A$ and $b_{j,t}$ for $j\in\mathcal T$ be nonnegative weights summing to one separately within each group. Let $\rho_t\geq0$ control the translation-group penalty. With recognition at the upper level and translation below it, the reported encoder update has the form

$$
\theta_{t+1}=\theta_t-\alpha_t\left[
\sum_{i\in\mathcal A}a_{i,t}\nabla_\theta f_i
+\rho_t\sum_{j\in\mathcal T}b_{j,t}\nabla_\theta f_j
+\eta_t\nabla_\theta u\right].
$$

Swapping the supervised groups gives the other task ordering. Gradient balancing now operates within groups, while penalties control their interaction. The paper calls the lower-to-upper order self-supervision, recognition, translation **UAS**, and the order self-supervision, translation, recognition **USA**. These labels describe the hierarchy used in coupled training, not three independently trained models. This is the application-specific modeling choice: recognition and translation can share acoustic features without requiring all of their language-specific losses to participate in one common balancing problem. [Objective Soups, Equations 5–11 and Table 1](https://arxiv.org/html/2508.09228v1#S3)

Figure 7 compares the three ways of organizing the objectives.

![Three ways to organize multilingual speech objectives](https://hackmd.io/_uploads/ByhCniB6zl.svg)

*Figure 7. Three ways to organize multilingual speech objectives. Objective organization in multilingual speech processing (MSP), with automatic speech recognition (ASR), speech-to-text translation (ST), and acoustic learning. (a) Vectorized single-level MSP (VS-MSP) balances all gradients together. (b) Vectorized objectives with a lower-level constraint (VC-MSP) balances supervised gradients and adds an acoustic penalty. (c) Vectorized multilevel MSP (VM-MSP) balances language objectives separately within recognition and translation. The USA hierarchy shown orders self-supervision, translation, and recognition from lower to upper level; swapping the supervised levels gives UAS. All branches contribute to one coupled training update. Original schematic based on [Saif et al. (2025)](https://arxiv.org/html/2508.09228v1#S3).*

**Stochastic gradient weighting.** The training methods use MoDo to estimate min-norm gradient weights. Let $r$ be the number of objectives in the current balancing subproblem, let $D$ be the number of parameter coordinates used in the weight calculation, and let $G(x)\in\mathbb R^{D\times r}$ collect the corresponding gradients as columns. The full-parameter formulation uses $D=n$; restricting the calculation to selected encoder layers reduces $D$. Write $\lambda\in\Delta_r$ for the weights of this subproblem, and let $\|\cdot\|_2$ denote the Euclidean norm. The ideal weight subproblem minimizes $\tfrac12\|G(x)\lambda\|_2^2$. At iteration $t$, let $\widehat G_{t,1}$ and $\widehat G_{t,2}$ be independent unbiased estimates of $G(x_t)$ conditional on the current parameters, let $\kappa_t>0$ be the weight step size, and let $\Pi_{\Delta_r}$ denote Euclidean projection onto the simplex. MoDo uses

$$
\lambda_{t+1}=\Pi_{\Delta_r}\!\left[
\lambda_t-\kappa_t(\widehat G_{t,1})^\top\widehat G_{t,2}\lambda_t
\right].
$$

The independent estimates make the cross-product unbiased for the population gradient Gram matrix; squaring one noisy estimate would introduce a covariance term. VS-MSP includes the acoustic objective in this balancing problem. In the operational VC-MSP and VM-MSP updates above, supervised objectives are balanced and the acoustic gradient is added separately as a penalty; VM-MSP also balances the supervised groups separately. Accordingly, the published pseudocode samples unlabeled data twice for VS-MSP and once for VC-MSP and VM-MSP. This sampling distinction matters when translating a deterministic min-norm solver into speech-model training. The stochastic implementation should not be interpreted as an exact common-descent guarantee at every update. [Objective Soups, Equation 4 and Algorithms 1–3](https://arxiv.org/html/2508.09228v1#S3.SS1)

### 5.2.3. Empirical Trade-offs and Training Cost

The main CoVoST-2 comparison holds the model family fixed and changes the training method. Table 5 gives the paper's reported averages for the 150-million-parameter Conformer. WER averages cover the five recognition languages; BLEU, a translation-quality score for which higher values are better, averages the four translation directions. Including static and joint-training controls helps distinguish the benefit of concurrent training from that of the proposed objective organization. [Objective Soups, Table 2](https://proceedings.nips.cc/paper_files/paper/2025/file/188584a9e175a84cdf6b09fc6843f308-Paper-Conference.pdf)

*Table 5. Recognition and translation on CoVoST-2 with a 150-million-parameter Conformer. Values are reported averages from Saif et al. (2025), Table 2; lower WER and higher BLEU are preferred.*

| Training method | Mean recognition WER | Mean translation BLEU |
|---|---:|---:|
| Two-stage pre-training and fine-tuning | 20.5% | 26.5 |
| Two-stage static weighting | 20.4% | 26.6 |
| Joint training | 19.8% | 27.5 |
| VS-MSP | 19.7% | 27.4 |
| VC-MSP | 19.2% | 28.1 |
| VM-MSP, UAS | 18.4% | 28.7 |
| VM-MSP, USA | 18.7% | 28.9 |
| VM-MSP, efficient variant | 18.4% | 28.8 |

The reported VS-MSP averages are close to those of joint training, so the largest gains occur in the constrained and hierarchical formulations. Both variants improve the reported averages relative to the two-stage baseline. UAS and USA favor different metrics: UAS has lower WER, while USA has higher BLEU. The table supports a useful hierarchy in these experiments, without establishing that one order is uniformly best.

**Penalty sensitivity.** On the 83-million-parameter model, reducing the penalty increase from 0.02 to 0.002 per epoch changes UAS from 23.6% WER and 24.6 BLEU to 25.5% WER and 25.3 BLEU. Recognition worsens while translation improves. Under USA, the same schedule change moves WER from 23.9% to 23.1% and BLEU from 24.9 to 23.3. These results show that the penalty schedule changes the recognition–translation trade-off even when the hierarchy is fixed. [Objective Soups, Table 4](https://arxiv.org/html/2508.09228v1#S4)

**Computational efficiency.** The authors screen encoder layers for negative pairwise gradient cosine similarities during the first 20 epochs, reporting persistent conflict in roughly 20–30% of backbone layers. The efficient variant uses selected-layer gradient information to compute the balancing weights while continuing model training. This reduces the size of the gradient-balancing calculation; it does not require freezing all other layers. Table 6 reports the corresponding resource measurements.

*Table 6. GPU memory and training time per epoch for the two-stage baseline and dynamic-weighting implementations. From Saif et al. (2025), Table 5.*

| Training implementation | GPU memory | Time per epoch |
|---|---:|---:|
| Two-stage baseline | 12.5 GB | 3.5 hours |
| Full dynamic weighting | 14.7 GB | 4.1 hours |
| Efficient weighting | 12.8 GB | 3.7 hours |

The resource measurements in Table 6 and the efficient variant's accuracy in Table 5 quantify the cost–performance trade-off. As with representation-level balancing in recommender systems, selecting the gradient information used to determine weights can reduce overhead. Here the selection is based on where acoustic-task conflict occurs in the encoder. [Objective Soups, Section 4.7 and Table 5](https://arxiv.org/html/2508.09228v1#S4.SS7)

### 5.2.4. Preference-Guided Speech LLM Adaptation

[EA-SURF (Jiang et al., 2026)](https://arxiv.org/html/2610.27200v1) addresses a complementary problem: adapting a speech large language model (LLM) to a specified preference among recognition, timestamp prediction, and translation. Its model uses a frozen acoustic encoder and a frozen base LLM, with a trainable speech adapter and low-rank adaptation modules totaling 81 million trainable parameters. Timestamp targets interleave words with their end times. All three training objectives are token cross-entropies; WER, timing error, and BLEU are evaluation metrics.

The key distinction is between **requested task priorities and loss weights**. For this model, let $n_{\mathrm L}$ be the positive integer counting trainable parameters, and let $x_{\mathrm L}\in\mathbb R^{n_{\mathrm L}}$ collect them. Let $L(x_{\mathrm L})=(L_1(x_{\mathrm L}),L_2(x_{\mathrm L}),L_3(x_{\mathrm L}))\in\mathbb R^3$ collect the recognition, timestamp, and translation losses in that order, with $j\in\{1,2,3\}$ indexing these tasks. Let $v\in\Delta_3$, with $v_j>0$ for every task, be the requested preference, and let $w\in\Delta_3$ be the training weights in the linear scalarization $\sum_{j=1}^3w_jL_j(x_{\mathrm L})$. Setting $w=v$ need not produce the desired compromise.

EA-SURF instead measures the model relative to specialist anchors. Let $e_j\in\mathbb R^3$ be the full loss vector of a model trained predominantly for task $j$. Define $d_j(x_{\mathrm L})$ as its Euclidean distance from the current loss vector, $q_j(x_{\mathrm L})$ as its share of the total distance, and $q_j^\star(v)$ as the requested distance share. With $h\in\{1,2,3\}$ a summation index and a positive total distance, these quantities are

$$
d_j(x_{\mathrm L})=\|L(x_{\mathrm L})-e_j\|_2,
\qquad
q_j(x_{\mathrm L})=\frac{d_j(x_{\mathrm L})}{\sum_{h=1}^3d_h(x_{\mathrm L})},
\qquad
q_j^\star(v)=\frac{1-v_j}{2}.
$$

Greater priority means a smaller target distance share from that task's specialist. For example, preference $(0.4,0.4,0.2)$ maps to distance shares $(0.3,0.3,0.4)$. These targets describe geometry in training-loss space; they do not prescribe numerical WER or BLEU values.

For this algorithm, let the nonnegative integer $t$ index outer weight-adjustment rounds, and write $x_{\mathrm L,t}$ and $w_t$ for the parameters and training weights at round $t$. Let $\epsilon_0>0$ stabilize ratios, and let $\tau\in(0,1]$ be an adjustment gain. Define the current-to-target ratio $r_{j,t}$ and the next weight by

$$
r_{j,t}=\frac{q_j(x_{\mathrm L,t})+\epsilon_0}{q_j^\star(v)+\epsilon_0},
\qquad
w_{j,t+1}=\frac{w_{j,t}(r_{j,t})^\tau}{\sum_{h=1}^3w_{h,t}(r_{h,t})^\tau}.
$$

Starting from $w_0=v$, the method multiplies each weight by its current-to-target distance-share ratio raised to the adjustment gain, renormalizes the weights, and continues training the same model. A larger ratio gives that task a larger multiplicative adjustment. Task sampling remains fixed. EA-SURF adjusts scalarization weights using measured training-loss geometry, whereas MoDo computes weights from gradient interactions. [EA-SURF, Sections 3.1–3.2](https://arxiv.org/html/2610.27200v1#S3)

Figure 8 summarizes this preference-calibration loop.

![Preference calibration during speech-model adaptation](https://hackmd.io/_uploads/By2AnsBafg.svg)

*Figure 8. Preference calibration during speech-model adaptation. EA-SURF converts a fixed request into target distance shares and compares them with the shares attained by the current model. Each attained share is the Euclidean distance to one of three fixed specialist loss vectors, divided by the sum of the three distances. Every loss vector contains token cross-entropies for recognition, timestamp prediction, and translation. Initial loss weights equal the requested preference. Subsequent feedback rounds adjust and normalize the weights, then continue training the same model. The preference request, specialist anchors, and task sampler remain fixed. Original schematic based on [Jiang et al. (2026)](https://arxiv.org/html/2610.27200v1#S3).*

Across six preference requests, EA-SURF reduces mean accumulated averaging shift (AAS), the reported timestamp-error metric, from 46.6 to 39.2 milliseconds and increases endpoint-inclusive normalized hypervolume, a summary of the attained trade-off set relative to a reference point, from 0.519 to 0.612. Mean BLEU improves by 0.45, while mean recognition WER worsens by 0.16 percentage points. These are trade-offs, rather than uniform task improvements. The comparison also requires three reusable specialist-training runs in addition to request-level adaptation; the reported requests are single runs, not repeated-seed uncertainty estimates. Its preference-realizability and fixed-point results are conditional and do not establish convergence of the finite-budget adaptive procedure. [EA-SURF, Section 4 and Table 2](https://arxiv.org/html/2610.27200v1#S4)

Together, these applications expose distinct controls over a shared speech model: an acoustic penalty discourages degradation of representation quality, grouped gradient balancing changes how task conflicts are handled, and preference calibration adjusts the attained compromise. Their effectiveness should be evaluated through recognition, translation, timing, and training-cost measurements appropriate to the particular formulation.

### References

- Saif, A. F. M., Chen, L., Cui, X., Lu, S., Kingsbury, B., & Chen, T. (2024). *M²ASR: Multilingual Multi-task Automatic Speech Recognition via Multi-objective Optimization.* Interspeech 2024, 1240–1244. [DOI](https://doi.org/10.21437/Interspeech.2024-1492).
- Saif, A. F. M., Chen, L., Cui, X., Lu, S., Kingsbury, B., & Chen, T. (2025). *Objective Soups: Multilingual Multi-Task Modeling for Speech Processing.* Advances in Neural Information Processing Systems 38. [Proceedings](https://proceedings.nips.cc/paper_files/paper/2025/hash/188584a9e175a84cdf6b09fc6843f308-Abstract-Conference.html).
- Jiang, L., Sunder, V., Thomas, S., Kingsbury, B., Kuo, J., Chen, L., & Saon, G. (2026). *Preference-Aware Pareto Steering for Multi-Task Speech LLMs.* arXiv:2610.27200, version 1, October 19, 2026. [Preprint](https://arxiv.org/abs/2610.27200).
## 5.3 Networking Applications

### 5.3.1 Background

Wireless networks connect devices over shared communication channels, requiring coordination of transmission decisions to use limited resources efficiently. Improving one device's performance may reduce the transmission opportunities available to others, creating trade-offs among device-level objectives and network-wide criteria such as throughput, information freshness, and fairness. This distributed decision-making structure naturally motivates multi-agent reinforcement learning (MARL), which enables devices to learn transmission policies through interaction with the network. When agents jointly optimize an aggregation of their individual returns, cooperative MARL can be interpreted as solving a scalarized MOO problem. We next review two representative works that illustrate this connection in wireless random access networks.

### 5.3.2 Cooperative MARL for Wireless Network Optimization

**Consensus-based decentralized MARL.** Oh et al. (2025) develop a decentralized actor–critic framework in which each device learns when to transmit or wait based on its observations and interaction history. Each device receives a local reward defined as a negative weighted sum of its normalized queue length and the time since its last successful transmission, encouraging timely transmissions and reduced queue backlogs. Neighboring devices exchange only scalar rewards through average consensus to estimate the network-wide reward, which guides local actor and critic updates without centralized training or critic-parameter exchanges. The paper provides finite-time convergence bounds and reports improvements in throughput and transmission fairness. From an MOO perspective, the reward design combines two levels of scalarization: weighting performance criteria within each device and averaging rewards across devices. This produces a shared optimization criterion that coordinates competing transmission demands.

![MARL_Architecture](https://hackmd.io/_uploads/H1olUFkCMg.png)



**Foundation model-aided MARL.** Building on this framework, Oh et al. (2026) address the training overhead associated with adapting MARL to different network objectives. They pretrain a reward-agnostic foundation model through forward-dynamics prediction, learning representations of network dynamics from observation–action sequences without reward labels. During online learning, the pretrained model remains frozen as the critic backbone, while a task-specific head learns value estimates and the actor updates its transmission policy. The framework retains local reward consensus and supports three downstream tasks: minimizing worst-case age of information (Fair-AoI), maximizing sum-rate subject to minimum per-device rate requirements (Max-rate), and maximizing the minimum device throughput (Fair-rate). These tasks encode different priorities concerning efficiency and fairness and are optimized separately. The paper provides finite-time analysis with nonlinear value approximation and demonstrates faster online learning with comparable final performance. For MOO, its central relevance is the reuse of network representations across different objective formulations, providing a foundation for learning policies under varying performance preferences.

![FM_MARL_Architecture](https://hackmd.io/_uploads/rJAkIKyRGx.png)



### 5.3.3 Extensions toward Explicit MOO Framework

These studies motivate extensions that explicitly characterize trade-offs among competing objectives. One promising direction is to retain device returns or network performance criteria as a vector objective and apply preference-guided scalarization or multi-objective policy-gradient methods to seek policies representing different trade-offs, while incorporating service requirements as constraints. The reward-agnostic backbone could further support objective-specific value heads and policies conditioned on user preferences, potentially reducing the cost of exploring the Pareto front. A key challenge is to preserve objective-specific information during decentralized coordination and establish convergence guarantees that account for communication errors and representation approximation.


### Reference

Oh, M. S., Zhang, Z., Hairi, F., Velasquez, A., & Liu, J. (2025). Consensus-based decentralized multi-agent reinforcement learning for random access network optimization. In Proceedings of the Twenty-sixth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc ’25) (pp. 241–250). https://arxiv.org/abs/2508.07001

Oh, M. S., Zhang, Z., Velasquez, A., Bastian, N. D., & Liu, J. (2026). Foundation model-aided multi-agent reinforcement learning for wireless random access network optimization. In The Twenty-seventh International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing (MobiHoc ’26). https://arxiv.org/abs/2610.07550



## 5.4 LLM Unlearning

### 5.4.1 Background

LLM unlearning is an emerging paradigm in machine learning that aims to mitigate risks such as privacy breaches, toxic outputs, and copyright infringement by removing undesirable knowledge from models while maintaining safe, compliant, and useful model behavior. 

Several benchmarks are widely used to evaluate LLM unlearning methods, including TOFU (Maini et al., 2024), MUSE (Shi et al., 2025), and WMDP (Li et al., 2024). These benchmarks assess how effectively models forget targeted information while preserving knowledge and capabilities unrelated to the unlearning targets. Common evaluation metrics include ROUGE-L scores, question-answering accuracy, answer probabilities, and truth ratios, which capture complementary aspects of forgetting efficacy and utility preservation.

### 5.4.2 Multi-Objective Approaches to LLM Unlearning

Indeed, LLM unlearning naturally involves *competing* objectives: on the one hand, it seeks to remove undesirable knowledge to reduce the associated risks; on the other hand, it must preserve model utility. These objectives can conflict because the knowledge targeted for removal may also include useful basic knowledge, and thus removing it can degrade the LLM's performance even on questions unrelated to the unlearning target. Prior work does not explicitly incorporate MOO techniques but still aims to balance the forget and retain gradients (Fan et al., 2024; Jang et al., 2023; Liu et al., 2024). Recently, several MOO-based LLM unlearning methods are proposed to systematically address the problem.

Recent studies explicitly adopt multi-objective optimization to balance *forgetting* and *retaining* in LLM unlearning. Jin et al. (2025) formulate unlearning as a two-task optimization problem that minimizes the retain loss while maximizing the forget loss. Their normalized gradient difference (NGDiff) method normalizes the two gradients before taking their difference, dynamically balancing their directions. Pan et al. (2025) formulate a three-objective problem comprising an unlearning cross-entropy loss, a retain cross-entropy loss, and a KL-divergence term that preserves the original model's output distribution on retain data. Their Dual Space Multiple Gradient Descent Algorithm (DS-MGDA) combines projected objective gradients to construct a common descent direction under appropriate rank conditions. Despite these algorithmic differences, both methods primarily balance forgetting against retention. However, balancing these objectives alone does not ensure the preservation of neighboring-domain knowledge or robustness to adversarial probing. Moreover, their update rules do not explicitly incorporate user-specified preferences over competing objectives, limiting direct control over the desired trade-offs. Although these works discuss Pareto optimality or stationarity, they do not establish any finite-time convergence rates for their proposed unlearning algorithms.

### 5.4.3 CLEAR: Preference-Guided Constrained Multi-Objective Unlearning

To address these limitations, CLEAR (**OUR PREPRINT**) formulates LLM unlearning as a *constrained* multi-objective optimization problem encompassing target forgetting, general retention, neighboring-domain preservation, and robustness to prefilling attacks:
\begin{align*}
\min_{\theta}\quad
\mathbf{f}(\theta)
&=
\big[
f_{\mathrm{fg}}(\theta),
f_{\mathrm{rt}}(\theta),
f_{\mathrm{nbr}}(\theta),
f_{\mathrm{rob}}(\theta)
\big]^{\top},
\quad\text{s.t.}\quad
f_i(\theta)
\le \delta_i,\quad \forall i\in\mathcal{I},
\end{align*}
where $f_i$ denotes the loss associated with the $i$-th objective, $\mathcal{I}:=\{\mathrm{fg},\mathrm{rt},\mathrm{nbr},\mathrm{rob}\}$, and $\delta_i$ specifies the corresponding *task-level* tolerance. Notably, (i) the tolerances $\delta_i$ are tunable, allowing users to specify acceptable thresholds for individual objectives; and (ii) although no finite set of objectives can capture every aspect of LLM unlearning, the numerical results suggest that these four objectives provide a representative selection and that optimizing them can also yield favorable performance in neighboring domains.

To solve this constrained MOO problem, CLEAR applies weighted-Chebyshev scalarization with an explicit user preference vector. It then establishes the first finite-time convergence guarantee for LLM unlearning, showing that the KKT residual decreases at a rate of $\mathcal{O}(T^{-\frac{1}{2}})$. Moreover, by varying the preference vector over the simplex, users can approximate the Pareto front and select a solution that best reflects their preferences.

![wmdp_clear_surface_no_basemodel_with_sum_smoothed](https://hackmd.io/_uploads/S1pH8FkRGg.png)





### References

Maini, P., Feng, Z., Schwarzschild, A., Lipton, Z. C., & Kolter, J. Z. (2024). Tofu: A task of fictitious unlearning for llms. arXiv preprint arXiv:2401.06121. https://arxiv.org/abs/2401.06121

Shi, W., Lee, J., Huang, Y., Malladi, S., Zhao, J., Holtzman, A., Liu, D., Zettlemoyer, L., Smith, N. and Zhang, C., 2025, May. Muse: Machine unlearning six-way evaluation for language models. In International Conference on Learning Representations (Vol. 2025, pp. 27797-27818). https://arxiv.org/abs/2407.06460

Li, N., Pan, A., Gopal, A., Yue, S., Berrios, D., Gatti, A., Li, J.D., Dombrowski, A.K., Goel, S., Phan, L. and Mukobi, G., 2024. The wmdp benchmark: Measuring and reducing malicious use with unlearning. arXiv preprint arXiv:2403.03218. https://arxiv.org/abs/2403.03218


Fan, C., Liu, J., Lin, L., Jia, J., Zhang, R., Mei, S., & Liu, S. (2024). Simplicity prevails: Rethinking negative preference optimization for LLM unlearning. arXiv. https://arxiv.org/abs/2410.07163

Jang, J., Yoon, D., Yang, S., Cha, S., Lee, M., Logeswaran, L., & Seo, M. (2023). Knowledge unlearning for mitigating privacy risks in language models. In A. Rogers, J. Boyd-Graber, & N. Okazaki (Eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 14389–14408). https://arxiv.org/abs/2210.01504

Liu, C. Y., Wang, Y., Flanigan, J., & Liu, Y. (2024). Large language model unlearning via embedding-corrupted prompts. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, & C. Zhang (Eds.), Advances in neural information processing systems (Vol. 37, pp. 118198–118266). https://arxiv.org/abs/2406.07933

Jin, X., Bu, Z., Vinzamuri, B., Ramakrishna, A., Chang, K.-W., Cevher, V., & Hong, M. (2025). Unlearning as multi-task optimization: A normalized gradient difference approach with an adaptive learning rate. In L. Chiruzzo, A. Ritter, & L. Wang (Eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) (pp. 11278–11294). https://arxiv.org/abs/2410.22086

Pan, Z., Zhang, S., Zheng, Y., Li, C., Cheng, Y., & Zhao, J. (2025). Multi-objective large language model unlearning. In ICASSP 2025—2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 1–5). IEEE. https://arxiv.org/abs/2412.20412



## 5.5. Multi-Objective Reinforcement Learning 


### The sequential structure of MORL

MORL retains the abstract objective $F(x)=(f_1(x),\ldots,f_M(x))^\top$ and the Pareto-stationarity geometry of [Section 1](https://hackmd.io/ROaYhbMiQ3OzhgIWLNnL-w). Its additional structure lies in **how a policy generates the objectives, the data, and the gradient estimates**. An action affects both the current reward vector and the future states in which all subsequent rewards are collected.

#### One policy generates a vector of long-term returns

Consider a discounted MDP with finite state and action spaces $\mathcal S$ and $\mathcal A$, transition kernel $P(s'\mid s,a)$, initial-state distribution $\mu$, bounded reward vector $r(s,a)=(r_1(s,a),\ldots,r_M(s,a))^\top$, and discount $\gamma\in[0,1)$. Following Section 1, $x\in\mathbb R^n$ is the decision variable, $m\in[M]$ indexes objectives, and all objectives are minimized. Here $x$ parameterizes a differentiable policy $\pi_x(a\mid s)$, and
$$
f_m(x)=-J_m(x),
\qquad
J_m(x)=\mathbb E_{\mu,P,\pi_x}\!\left[\sum_{\ell=0}^{\infty}\gamma^\ell r_m(s_\ell,a_\ell)\right].
$$
All $M$ return components arise from the **same controlled trajectory**. Thus, the policy cannot choose a separate state-visitation pattern for each objective: reducing energy use now, for example, can change which states are reached and what future task rewards are achievable.

#### The decision variable also controls the sampling distribution

Define the normalized discounted state–action occupancy
$$
\nu_x(s,a)=(1-\gamma)\sum_{\ell=0}^{\infty}\gamma^\ell
\Pr_{\mu,P,\pi_x}(s_\ell=s,a_\ell=a).
$$
Then
$$
f_m(x)=-\frac{1}{1-\gamma}\sum_{s,a}\nu_x(s,a)r_m(s,a).
$$
The dependence on $x$ enters through the trajectory distribution itself. These occupancies obey the common flow constraints
$$
\sum_a\nu_x(s,a)
=(1-\gamma)\mu(s)
+\gamma\sum_{s',a'}P(s\mid s',a')\nu_x(s',a'),
\qquad s\in\mathcal S.
$$
Although the returns are linear in occupancy, the map $x\mapsto\nu_x$ induced by a policy parameterization is generally nonlinear. Section 1 abstracts this structure into the functions $f_m$ and assumes their exact gradients are available. In model-free MORL, even evaluating these functions requires interacting with an environment or using previously collected trajectories. Updating $x$ changes the distribution needed for the next evaluation; reusing old data requires accounting for that distribution mismatch.

#### Actor–critic gradients use policy-dependent value functions

For each reward component, the action-value function satisfies the Bellman equation
$$
Q_m^x(s,a)=r_m(s,a)
+\gamma\,\mathbb E_{s'\sim P(\cdot\mid s,a),\,a'\sim\pi_x(\cdot\mid s')}
\bigl[Q_m^x(s',a')\bigr].
$$
For a differentiable policy with positive action probabilities, the [policy-gradient theorem (Sutton et al., 1999)](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) gives
$$
g_m(x):=\nabla f_m(x)
=-\frac{1}{1-\gamma}
\mathbb E_{(s,a)\sim\nu_x}
\left[\nabla_x\log\pi_x(a\mid s)\,Q_m^x(s,a)\right].
$$
The factor $1/(1-\gamma)$ matches the normalized occupancy above. This expression accounts for the policy's effect on future visitation. Replacing $Q_m^x$ by an approximate critic provides an actor-gradient estimate; the ordinary actor update does not additionally differentiate through the critic-fitting procedure.

For fixed $x$, the $M$ Bellman equations share the same transition operator and policy, while their reward components can be evaluated separately. Their coupling comes from the shared actor: changing $x$ can change all the value functions and the data distribution used to learn them.

#### Actor updates, critic tracking, and gradient balancing interact

Let $Q_m(s,a;z_{m,t})$ be the current critic for objective $m$. Conditional on the current actor and critic parameters, even ideal sampling from $\nu_{x_t}$ leaves an actor-gradient bias due to critic error:
$$
e_{m,t}^{\mathrm{critic}}
=-\frac{1}{1-\gamma}
\mathbb E_{\nu_{x_t}}\!\left[
\nabla_x\log\pi_{x_t}(a\mid s)
\bigl(Q_m(s,a;z_{m,t})-Q_m^{x_t}(s,a)\bigr)
\right].
$$
The critic is tracking a moving Bellman fixed point. Trajectory samples are also temporally dependent, and gradients for different reward components can have correlated errors because they use the same transitions. These are structural estimation issues that the exact-gradient formulation in Section 1 leaves implicit.

An MGDA-style actor update uses Section 1's sign convention,
$$
\widehat d_t=-\sum_{m\in[M]}\widehat\lambda_{m,t}\widehat g_{m,t},
\qquad x_{t+1}=x_t+\alpha_t\widehat d_t,
$$
where $\widehat\lambda_t\in\Delta_M$ balances the estimated gradients. But now a critic error can change both an individual gradient and the weights assigned to all gradients. Adaptive aggregation adds its own bias, already discussed in the [stochastic-methods section](https://hackmd.io/A5fgBHc1S0K2zfTfR-tI7A), on top of critic tracking and dependent sampling. MORL analysis must therefore control these errors together and relate them to environment-interaction cost.

#### MoCo: tracking the critic, the gradients, and their combination

The actor–critic remark in [MoCo, Section 3.2](https://openreview.net/forum?id=dLAYGdKTi2) motivates the paper's abstract nested model
$$
f_m(x)=\varphi_m\bigl(x,z_m^\star(x)\bigr),
\qquad
z_m^\star(x)=\operatorname*{argmin}_z\ell_m(x,z),
\qquad m\in[M].
$$
Here $\varphi_m$ denotes an outer objective, while $\ell_m$ is an inner fitting objective. The actor–critic interpretation associates the inner variable with policy evaluation: the critic should approach a policy-dependent target before its output is used to estimate the actor gradient. The model assumes $\ell_m(x,\cdot)$ is strongly convex. It is an analysis template, rather than a Bellman-specific derivation for arbitrary neural critics. Its implicit-hypergradient estimator is distinct from the ordinary policy-gradient evaluation above.

The source makes the resulting **four coupled quantities** explicit: actor parameters $x_t$, inner estimates $z_{m,t}$, gradient trackers $y_{m,t}$, and combination weights $\lambda_t$. Using Section 1's dimensions and direction convention, its recursions can be written as follows.

1. **Update the inner estimates.** In the abstract nested algorithm, one stochastic fitting step is
   $$
   z_{m,t+1}=z_{m,t}-\beta_t\widehat\nabla_z\ell_m(x_t,z_{m,t}).
   $$
   Its inexact outer-gradient estimator, denoted by $h_{m,t}$, uses $z_{m,t}$, the pre-update inner estimate in the source's formula. In actor–critic learning, the analogous estimation issue is that the available critic is an approximation to the current policy's value function. The precise actor estimator depends on the RL algorithm.

2. **Track each objective's gradient before combining objectives.** MoCo maintains
   $$
   y_{m,t+1}=\Pi_{\mathcal Y_m}
   \left[(1-\beta_t)y_{m,t}+\beta_t h_{m,t}\right],
   \qquad
   Y_t=[y_{1,t},\ldots,y_{M,t}]\in\mathbb R^{n\times M},
   $$
   where $\mathcal Y_m=\{y:\|y\|\leq B_m\}$ is a ball containing the true gradients under the bounded-gradient assumption. The manuscript uses the same $\beta_t$ for inner fitting and gradient tracking. This corrects the gradient information entering the combination step; it does not merely average the final actor updates.

3. **Update the combination weights and actor.** With $\kappa_t$ denoting the weight step size, to distinguish it from the RL discount $\gamma$, the source's updates become
   $$
   \lambda_{t+1}=\Pi_{\Delta_M}
   \left[\lambda_t-\kappa_t(Y_t^\top Y_t+\rho I)\lambda_t\right],
   \qquad
   d_t=-Y_t\lambda_t,
   \qquad
   x_{t+1}=x_t+\alpha_t d_t.
   $$
   These formulas use $Y_t$ and $\lambda_t$ for the actor step, as in the manuscript. The displayed actor update specializes its projected update to Section 1's unconstrained setting. The regularizer $\rho>0$ makes the weight subproblem strongly convex and stabilizes its moving solution; this perturbs the unregularized MGDA weights, so its choice must also enter the convergence analysis.

This mechanism exposes an RL-specific chain of errors. An actor update moves the critic target and the visitation distribution; critic lag affects $h_{m,t}$; the gradient trackers must follow the changing true gradients; and their errors alter both the direction and the combination weights. Gradient correction therefore operates alongside policy evaluation and data collection, rather than replacing them.

**How the relative update speeds enter.** The manuscript's appendix bounds inner tracking error under its strong-convexity, smoothness, and moment assumptions. For horizon-dependent constant steps $\alpha$ and $\beta$ over $T$ iterations, its intermediate calculation has the form
$$
\frac1T\sum_{t=0}^{T-1}\mathbb E\|z_{m,t}-z_m^\star(x_t)\|^2
=\mathcal O\!\left(
\frac{1}{T\beta}+\beta+\alpha+\frac{\alpha^2}{\beta^2}
\right).
$$
The terms reflect initial tracking error, stochastic fitting noise, and movement of the inner target as the actor changes. In particular, the ratio $\alpha/\beta$ shows why moving the actor too quickly can prevent accurate policy evaluation. This is conditional nested-model analysis; it does not establish the same bound for nonlinear SAC critics with replay data. A fixed regularizer or persistent critic error also cannot be assumed to recover the exact MGDA direction asymptotically.

**What the RL experiment establishes.** The source's RL appendix uses [MTRL](https://github.com/facebookresearch/mtrl) with SAC on Meta-World MT10: two million training steps, batch size 1280, evaluation every 10,000 steps, and five random seeds. These objectives correspond to different tasks with shared learned parameters, rather than reward channels on one common trajectory. Reported best average test success during training is:

| Method | Success, mean $\pm$ standard error |
|---|---|
| Multi-task SAC | $0.49\pm0.073$ |
| CAGrad | $0.83\pm0.045$ |
| MoCo | $0.75\pm0.050$ |
| CAGrad + MoCo correction | $0.86\pm0.022$ |

The combined method has a higher reported point estimate than CAGrad, while MoCo alone is below CAGrad. This supports studying gradient correction together with conflict handling in multi-task actor–critic learning. It does not establish statistical significance, a neural-SAC convergence theorem, or Pareto-front coverage. [MOAC (Zhou et al., 2024)](https://arxiv.org/abs/2405.03082) provides a complementary analysis directly organized around Markovian actor–critic estimation. Algorithmic comparisons with other stochastic MOO methods remain in the [stochastic-methods section](https://hackmd.io/A5fgBHc1S0K2zfTfR-tI7A).









