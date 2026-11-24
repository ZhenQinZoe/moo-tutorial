---
title: "Benchmarks and Libraries for Multi-Objective Optimization"
author: Lisha Chen
create date: 2026-11-09
update date: 2026-11-09
description: MOO libraries, the benchmark datasets in Chen et al.'s survey, complementary benchmark families, and evaluation protocols.
tags:
  - Multi-objective-optimization
  - Benchmarks
  - Software
type: blog
status: draft
---

# Benchmarks and Libraries for Multi-Objective Optimization

A useful MOO benchmark specifies both the optimization problem and the desired output: one model with a selected trade-off, a finite approximation of the Pareto front, or a preference-conditioned family of models. Libraries serve different parts of this workflow. LibMOON emphasizes gradient-based MOO and trade-off exploration, while LibMTL emphasizes shared-model multi-task learning. Evolutionary platforms provide complementary algorithms, test problems, and front-quality indicators.

Throughout this note, we use the notation in [Introduction to Multi-objective Optimization](Intro%20Multi-Objective%20Optimization.md): $\mathbf{x}\in X\subseteq\mathbb R^d$ is the decision vector and $\mathbf{F}(\mathbf{x})=(f_1(\mathbf{x}),\ldots,f_S(\mathbf{x}))^\top$ contains $S$ objectives to minimize. For algorithms, see [Gradient Weighting and Manipulation](Intro%20MOO%20--%20Gradient%20Manipulation.md) and [Preference-Guided MOO](Intro%20MOO%20--%20Preference%20Guided%20MOO.md).

## Libraries and their scope

| Library | Setting and implementation | Main role | Paper and resources |
| --- | --- | --- | --- |
| **LibMOON** | Gradient-based MOO; Python/PyTorch | Compare finite-solution MOO, Pareto set learning, and multi-objective Bayesian optimization on synthetic and learning problems. | [Zhang et al. (2024)](https://arxiv.org/abs/2409.02969 "LibMOON paper") [\[1\]](#^ref-zhang-2024-libmoon); [code](https://github.com/xzhang2523/libmoon); [docs](https://libmoondocs.readthedocs.io/en/latest/) |
| **LibMTL** | Deep multi-task learning; Python/PyTorch | Compare loss/gradient balancing strategies and network architectures within a common training framework. | [Lin and Zhang (2023)](https://www.jmlr.org/papers/v24/22-0347.html "LibMTL paper") [\[2\]](#^ref-lin-2023-libmtl); [code](https://github.com/median-research-group/LibMTL); [docs](https://libmtl.readthedocs.io/) |
| **pymoo** | General MOO, especially evolutionary methods; Python | Customize algorithms and operators, run constrained and unconstrained problems, and evaluate or visualize solution sets. | [Blank and Deb (2020)](https://arxiv.org/abs/2002.04504 "pymoo paper") [\[3\]](#^ref-blank-2020-pymoo); [code](https://github.com/anyoptimization/pymoo); [docs](https://pymoo.org/) |
| **PlatEMO** | Evolutionary MOO; MATLAB | Run comparative experiments across algorithms and benchmark families, with a graphical interface and result export. | [Tian et al. (2017)](https://arxiv.org/abs/1701.00879 "PlatEMO paper") [\[4\]](#^ref-tian-2017-platemo); [code and documentation](https://github.com/BIMK/PlatEMO) |
| **jMetal** | Multi-objective metaheuristics; Java | Develop algorithms and conduct experiments using shared problems, quality indicators, and statistical reporting. | [Durillo and Nebro (2011)](https://doi.org/10.1016/j.advengsoft.2011.05.014 "jMetal paper") [\[5\]](#^ref-durillo-2011-jmetal); [code](https://github.com/jMetal/jMetal); [docs](https://jmetal.readthedocs.io/) |
| **DeepMTL2R** | Multi-task learning to rank; deep learning | Compare MTL strategies and inspect trade-offs among relevance criteria in ranking models. | [Dong et al. (2026)](https://arxiv.org/abs/2602.14519 "DeepMTL2R paper") [\[6\]](#^ref-dong-2026-deepmtl2r); [code and documentation](https://github.com/amazon-science/DeepMTL2R) |
| **MO-Gymnasium / MORL-Baselines** | Multi-objective reinforcement learning; Python | Pair environments with vector rewards with algorithm implementations and evaluation tools. | [Felten et al. (2023)](https://proceedings.neurips.cc/paper_files/paper/2023/file/4aa8891583f07ae200ba07843954caeb-Paper-Datasets_and_Benchmarks.pdf "MORL benchmarking toolkit paper") [\[7\]](#^ref-felten-2023-morl-toolkit); [environments](https://mo-gymnasium.farama.org/); [baselines](https://github.com/LucasAlegre/morl-baselines) |

For gradient aggregation experiments, LibMTL provides a common training pipeline in which the balancing rule can be varied while the architecture and data protocol are held fixed [\[2\]](#^ref-lin-2023-libmtl). For preference sweeps and Pareto-set approximation, LibMOON provides solvers and metrics organized around those outputs [\[1\]](#^ref-zhang-2024-libmoon). This distinction also appears in [Chen et al., Section 8.2](https://arxiv.org/html/2501.10945v3#S8.SS2) [\[8\]](#^ref-chen-2025-survey). These are different experimental questions; an algorithm's inclusion in a library does not make every benchmark configuration comparable.

## Benchmark datasets in the multi-task learning

The following table covers all 13 entries in [Table 3 and Section 8.1 of Chen et al. (2025)](https://arxiv.org/html/2501.10945v3#S8.SS1 "Survey benchmark datasets, Table 3") [\[8\]](#^ref-chen-2025-survey), with links to the original datasets and papers. The column $S$ describes the task or attribute configuration used in the survey; the optimization objectives must still be specified for each experiment. Dataset sizes refer to different units, as indicated below.

| Benchmark and source | $S$ in survey setup | Tasks or attributes | Scale and protocol |
| --- | --- | --- | --- |
| [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html "NYU Depth V2 dataset") [\[9\]](#^ref-silberman-2012-nyuv2) | 3 | Segmentation, depth, surface normals | 1,449 labeled images: 795 train / 654 test. |
| [Taskonomy](https://github.com/StanfordVL/taskonomy "Taskonomy dataset and code") [\[10\]](#^ref-zamir-2018-taskonomy) | 26 | Visual prediction tasks | Approximately 4M images; experiments often select a task subset. |
| [Cityscapes](https://www.cityscapes-dataset.com/ "Cityscapes dataset") [\[11\]](#^ref-cordts-2016-cityscapes) | 2 | Segmentation and depth | 2,975 train / 500 validation images in the usual MTL protocol. |
| [QM9](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html "QM9 data loader and target definitions") [\[12\]](#^ref-ramakrishnan-2014-qm9) | 11 | Selected molecular-property regression targets | Approximately 130K molecules; specify the selected targets and filtering. |
| [Office-31](https://faculty.cc.gatech.edu/~judy/domainadapt/ "Office domain-adaptation dataset") [\[13\]](#^ref-saenko-2010-office) | 3 | Amazon, DSLR, Webcam classification | 4,110 images; 31 classes per domain. |
| [Office-Home](https://www.hemanthdv.org/officeHomeDataset.html "Office-Home dataset") [\[14\]](#^ref-venkateswara-2017-officehome) | 4 | Art, Clipart, Product, Real-World classification | Approximately 15,500 images; 65 classes per domain. |
| [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html "CelebA dataset") [\[15\]](#^ref-liu-2015-celeba) | 40 | Binary facial-attribute classification | 202,599 images; use the official train/validation/test partitions. |
| [CIFAR-100](https://cave.cs.toronto.edu/kriz/cifar.html "CIFAR datasets and superclass definitions") [\[16\]](#^ref-krizhevsky-2009-cifar) | 20 | One five-class task per superclass | 50K train / 10K test images; this is a task decomposition of the 100-class dataset. |
| [Meta-World MT10 / MT50](https://meta-world.github.io/ "Original Meta-World benchmark") [\[17\]](#^ref-yu-2020-metaworld) | 10 / 50 | Robot-manipulation tasks | Interactive environments; report environment version, interaction budget, and evaluation episodes. |
| [XTREME](https://github.com/google-research/xtreme "Original XTREME benchmark") [\[18\]](#^ref-hu-2020-xtreme) | 9 | Cross-lingual classification, structured prediction, QA, retrieval | 40 languages; approximately 597K examples reported in the survey. |
| [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF "PKU-SafeRLHF dataset card") [\[19\]](#^ref-ji-2024-pkusaferlhf) | 2 | Helpfulness and harmlessness preferences | Approximately 82K in the survey; the linked card describes 83.4K preference entries, each with two responses. |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback "UltraFeedback dataset card") [\[20\]](#^ref-cui-2023-ultrafeedback) | 4 | Instruction following, truthfulness, honesty, helpfulness | 63,967 prompts, four responses per prompt, with GPT-4 feedback. |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2 "HelpSteer2 dataset card") [\[21\]](#^ref-wang-2024-helpsteer2) | 5 | Helpfulness, correctness, coherence, complexity, verbosity | 21,362 prompt–response records with human ratings; paired records share prompts. |

**Task counts depend on the formulation.** Office domains become separate objectives when each domain has its own loss. CIFAR-100 becomes a 20-objective problem only under the specified superclass decomposition. For QM9, the survey selects 11 targets, whereas the linked PyTorch Geometric loader exposes 19 regression targets [\[12\]](#^ref-ramakrishnan-2014-qm9). Record target indices, units, and normalization. Likewise, Meta-World's task returns and the multiple reward components within one MORL environment define different objective constructions [\[17\]](#^ref-yu-2020-metaworld).

**Sample counts depend on the release and unit.** Cityscapes has 5,000 finely annotated images overall; the 3,475 used here are the training and validation images, rather than the full dataset [\[11\]](#^ref-cordts-2016-cityscapes). For LLM data, distinguish prompts, responses, and preference pairs, and pin the dataset revision and split. Multiple annotations supply candidate training signals; define which become losses or rewards and how they are evaluated. For example, complexity and verbosity may be target attributes rather than quantities to maximize without qualification [\[21\]](#^ref-wang-2024-helpsteer2).

## Complementary benchmark families

These benchmarks extend the deep-learning dataset table above to synthetic geometry, fairness, ranking, and vector-reward control.

| Family | Examples and source | What to examine |
| --- | --- | --- |
| Synthetic objective geometry | [ZDT](https://pymoo.org/problems/multi/zdt.html), [DTLZ](https://pymoo.org/problems/many/dtlz.html), and [WFG](https://pymoo.org/problems/many/wfg.html), available through evolutionary benchmark platforms | Front shape, multimodality, scaling with $d$ and $S$, and coverage of different trade-offs. Check each problem's differentiability and variable types before using a gradient solver. |
| Accuracy–fairness trade-offs | Adult fair-classification experiments in [LibMOON](https://arxiv.org/html/2409.02969v3#S4.SS3) [\[1\]](#^ref-zhang-2024-libmoon) | The trade-off between predictive loss and a fairness surrogate, together with the corresponding held-out fairness measure. |
| Multiple ranking criteria | [DeepMTL2R](https://github.com/amazon-science/DeepMTL2R) [\[6\]](#^ref-dong-2026-deepmtl2r) | Per-criterion ranking quality and trade-offs among relevance signals. |
| Sequential decision making | Deep-Sea-Treasure, Minecart, and multi-objective control environments in [MO-Gymnasium](https://mo-gymnasium.farama.org/) [\[7\]](#^ref-felten-2023-morl-toolkit) | Expected return vectors, policy-set coverage, preference-dependent utility, and environment interaction cost. |

A dataset alone does not define a MOO benchmark. The objectives, task grouping, train/validation/test split, preprocessing, model, and evaluation budget are part of the benchmark. In reinforcement learning, rewards are typically maximized; negate the return objectives when expressing them in the minimization notation above.

## Matching evaluation to the output

**One solution.** Report all held-out task metrics and the stated preference criterion, together with single-task and simple scalarization baselines. For gradient methods, record whether gradients are computed with respect to shared parameters or shared representations. Report training time and memory alongside iteration counts, since gradient aggregation can change the cost per iteration.

For the datasets above, also report the metrics appropriate to each task: classification accuracy, segmentation overlap, depth or normal errors, molecular-property errors, ranking scores, or task success rates. For multilingual benchmarks, specify aggregation across tasks and languages. For LLM alignment, report the attribute scores and the evaluator used, and distinguish reward-model fitting from evaluation of the generated responses. A single average can conceal which objectives improve or deteriorate.

**A solution set or a preference-conditioned model.** Evaluate objective vectors at a fixed number of returned solutions or queried preferences. Hypervolume measures the region dominated by the returned objective vectors relative to a reference point; larger values are better. IGD measures distance from a reference front sample to the returned set; smaller values are better. Both are described in the [pymoo indicator documentation](https://pymoo.org/misc/indicators.html). State objective normalization, the hypervolume reference point, and how the IGD reference set was obtained. On learning problems, the true front is generally unavailable, so an empirical reference set should be identified as such.

**Controlled comparisons.** Fix data splits, model capacity, tuning budgets, stopping rules, and random seeds across methods. For front approximation, include the cost of all preference runs or population members. For Pareto set learning, distinguish model-training cost from the cost of querying additional preferences. Report variation across independent runs. A stationarity residual, a test-task score, and a front-coverage indicator answer different questions and should be interpreted separately.

## References

[1] Xiaoyuan Zhang, Liang Zhao, Yingying Yu, Xi Lin, Yifan Chen, Han Zhao, and Qingfu Zhang (2024). [LibMOON: A Gradient-based MultiObjective OptimizatioN Library in PyTorch](https://openreview.net/forum?id=etdXLAMZoc "LibMOON — NeurIPS 2024 Datasets and Benchmarks Track"). *Advances in Neural Information Processing Systems*, **37**, Datasets and Benchmarks Track. [arXiv:2409.02969](https://arxiv.org/abs/2409.02969). ^ref-zhang-2024-libmoon

[2] Baijiong Lin and Yu Zhang (2023). [LibMTL: A Python Library for Deep Multi-Task Learning](https://www.jmlr.org/papers/v24/22-0347.html "LibMTL — JMLR 24(209), 1–7"). *Journal of Machine Learning Research*, **24**(209), 1–7. ^ref-lin-2023-libmtl

[3] Julian Blank and Kalyanmoy Deb (2020). [pymoo: Multi-Objective Optimization in Python](https://doi.org/10.1109/ACCESS.2020.2990567 "pymoo — IEEE Access 8, 89497–89509"). *IEEE Access*, **8**, 89497–89509. [arXiv:2002.04504](https://arxiv.org/abs/2002.04504). ^ref-blank-2020-pymoo

[4] Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin (2017). [PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective Optimization](https://arxiv.org/abs/1701.00879 "PlatEMO — IEEE Computational Intelligence Magazine 12(4), 73–87"). *IEEE Computational Intelligence Magazine*, **12**(4), 73–87. ^ref-tian-2017-platemo

[5] Juan J. Durillo and Antonio J. Nebro (2011). [jMetal: A Java framework for multi-objective optimization](https://doi.org/10.1016/j.advengsoft.2011.05.014 "jMetal — Advances in Engineering Software 42(10), 760–771"). *Advances in Engineering Software*, **42**(10), 760–771. ^ref-durillo-2011-jmetal

[6] Chaosheng Dong, Peiyao Xiao, Yijia Wang, and Kaiyi Ji (2026). [DeepMTL2R: A Library for Deep Multi-task Learning to Rank](https://arxiv.org/abs/2602.14519 "DeepMTL2R — arXiv:2602.14519"). arXiv:2602.14519. ^ref-dong-2026-deepmtl2r

[7] Florian Felten, Lucas N. Alegre, Ann Nowé, Ana L. C. Bazzan, El-Ghazali Talbi, Grégoire Danoy, and Bruno C. da Silva (2023). [A Toolkit for Reliable Benchmarking and Research in Multi-Objective Reinforcement Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/4aa8891583f07ae200ba07843954caeb-Paper-Datasets_and_Benchmarks.pdf "MORL toolkit — NeurIPS 2023 Datasets and Benchmarks Track"). *Advances in Neural Information Processing Systems*, **36**, 23671–23700, Datasets and Benchmarks Track. ^ref-felten-2023-morl-toolkit

[8] Weiyu Chen, Baijiong Lin, Xiaoyuan Zhang, Xi Lin, Han Zhao, Qingfu Zhang, and James T. Kwok (2025). [Gradient-Based Multi-Objective Deep Learning: Algorithms, Theories, Applications, and Beyond](https://arxiv.org/abs/2501.10945v3 "Chen et al. — Table 3 and Sections 8.1–8.2"). *arXiv preprint arXiv:2501.10945v3*, Table 3 and Sections 8.1–8.2. ^ref-chen-2025-survey

[9] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus (2012). [Indoor Segmentation and Support Inference from RGBD Images](https://cs.nyu.edu/~fergus/datasets/indoor_seg_support.pdf "NYUv2 dataset paper"). *European Conference on Computer Vision (ECCV)*. ^ref-silberman-2012-nyuv2

[10] Amir R. Zamir et al. (2018). [Taskonomy: Disentangling Task Transfer Learning](https://arxiv.org/abs/1804.08328v1 "Taskonomy — CVPR 2018"). *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. ^ref-zamir-2018-taskonomy

[11] Marius Cordts et al. (2016). [The Cityscapes Dataset for Semantic Urban Scene Understanding](https://arxiv.org/abs/1604.01685 "Cityscapes — CVPR 2016"). *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. ^ref-cordts-2016-cityscapes

[12] Raghunathan Ramakrishnan et al. (2014). [Quantum chemistry structures and properties of 134 kilo molecules](https://doi.org/10.1038/sdata.2014.22 "QM9 — Scientific Data 1, 140022"). *Scientific Data*, **1**, 140022. [PyTorch Geometric target definitions](https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.datasets.QM9.html). ^ref-ramakrishnan-2014-qm9

[13] Kate Saenko, Brian Kulis, Mario Fritz, and Trevor Darrell (2010). [Adapting Visual Category Models to New Domains](https://www.cs.uml.edu/~saenko/saenko_eccv_2010.pdf "Office-31 — ECCV 2010"). *European Conference on Computer Vision (ECCV)*. ^ref-saenko-2010-office

[14] Hemanth Venkateswara, Jose Eusebio, Shayok Chakraborty, and Sethuraman Panchanathan (2017). [Deep Hashing Network for Unsupervised Domain Adaptation](https://arxiv.org/abs/1706.07522 "Office-Home — CVPR 2017"). *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. ^ref-venkateswara-2017-officehome

[15] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang (2015). [Deep Learning Face Attributes in the Wild](https://arxiv.org/abs/1411.7766 "CelebA — ICCV 2015"). *IEEE International Conference on Computer Vision (ICCV)*. ^ref-liu-2015-celeba

[16] Alex Krizhevsky (2009). [Learning Multiple Layers of Features from Tiny Images](https://cave.cs.toronto.edu/kriz/learning-features-2009-TR.pdf "CIFAR dataset technical report"). Technical report, University of Toronto. ^ref-krizhevsky-2009-cifar

[17] Tianhe Yu et al. (2020). [Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning](https://proceedings.mlr.press/v100/yu20a.html "Original Meta-World — CoRL 2019 proceedings"). *Proceedings of the Conference on Robot Learning (CoRL 2019)*, PMLR **100**, 1094–1100. ^ref-yu-2020-metaworld

[18] Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson (2020). [XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation](https://proceedings.mlr.press/v119/hu20b.html "XTREME — ICML 2020"). *Proceedings of the 37th International Conference on Machine Learning*, PMLR **119**, 4411–4421. ^ref-hu-2020-xtreme

[19] Jiaming Ji et al. (2024). [PKU-SafeRLHF: A Safety Alignment Preference Dataset for Llama Family Models](https://arxiv.org/abs/2406.15513v1 "PKU-SafeRLHF — original 2024 preprint"). *arXiv preprint arXiv:2406.15513v1*. [Dataset card and release details](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF). ^ref-ji-2024-pkusaferlhf

[20] Ganqu Cui et al. (2023). [UltraFeedback: Boosting Language Models with High-quality Feedback](https://arxiv.org/abs/2310.01377v1 "UltraFeedback — original 2023 preprint"). *arXiv preprint arXiv:2310.01377v1*. [Dataset card](https://huggingface.co/datasets/openbmb/UltraFeedback). ^ref-cui-2023-ultrafeedback

[21] Zhilin Wang et al. (2024). [HelpSteer2: Open-source dataset for training top-performing reward models](https://proceedings.neurips.cc/paper_files/paper/2024/hash/02fd91a387a6a5a5751e81b58a75af90-Abstract-Datasets_and_Benchmarks_Track.html "HelpSteer2 — NeurIPS 2024 Datasets and Benchmarks Track"). *Advances in Neural Information Processing Systems*, **37**, Datasets and Benchmarks Track. [Dataset card](https://huggingface.co/datasets/nvidia/HelpSteer2). ^ref-wang-2024-helpsteer2
