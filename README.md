<<<<<<< HEAD
# HierarchicalContrast

This repository contains code and data for the **EMNLP2023** paper **"HierarchicalContrast: A Coarse-to-Fine Contrastive Learning Framework for Cross-Domain Zero-Shot Slot Filling"**

Full version with Appendix: [PDF](https://arxiv.org/abs/2310.09135v2)


## Abstract

In task-oriented dialogue scenarios, cross-domain zero-shot slot filling plays a vital role in leveraging source domain knowledge to learn a model with high generalization ability in unknown target domain where annotated data is unavailable. However, the existing state-of-the-art zero-shot slot filling methods have limited generalization ability in target domain, they only show effective knowledge transfer on seen slots and perform poorly on unseen slots. To alleviate this issue, we present a novel Hierarchical Contrastive Learning Framework (HiCL) for zero-shot slot filling. Specifically, we propose a coarse- to fine-grained contrastive learning based on Gaussian-distributed embedding to learn the generalized deep semantic relations between utterance-tokens, by optimizing inter- and intra-token distribution distance. This encourages HiCL to generalize to the slot types unseen at training phase. Furthermore, we present a new iterative label set semantics inference method to unbiasedly and separately evaluate the performance of unseen slot types which entangled with their counterparts (i.e., seen slot types) in the previous zero-shot slot filling evaluation methods. The extensive empirical experiments on four datasets demonstrate that the proposed method achieves comparable or even better performance than the current state-of-the-art zero-shot slot filling approaches.

## Dataset

We evaluate our approach on four datasets, namely
[SNIPS](https://arxiv.org/abs/1805.10190) (Coucke et al., 2018), [ATIS](https://aclanthology.org/H90-1021) (Hemphill et al., 1990), [MIT_corpus](https://ojs.aaai.org/index.php/AAAI/article/view/17603) (Nie et al., 2021) and [SGD](https://arxiv.org/abs/2002.01359) (Rastogi et al., 2020)

## Main Results

We examine the effectiveness of HiCL by comparing it with the competing baselines. The results of
the average performance across different target domains on dataset of SNIPS, ATIS, MIT_corpus and
SGD are reported in Table 1, 2, 3, 4, respectively,
which show that the proposed method consistently
outperforms the previous BERT-based and ELMobased SOTA methods, and performs comparably to the previous RNN-based SOTA methods. The detailed results of seen-slots and unseen-slots performance across different target domains on dataset of SNIPS, ATIS, MIT_corpus and SGD are reported in Table 6, 7, 8, 9, respectively. On seen-slots side, the proposed method performs comparably to prior SOTA methods, and on unseen-slots side, the
proposed method consistently outperforms other
SOTA methods.

## Requirements

Here are the most commonly used options: config.json file.

config.py file explains which options are used and how.

- target_domain: The domain to be the target of the test, and the train data is configured with the remaining domains except the domain
- n_samples: Specify how much data from the target domain to use with train data in a few shot learning. 0 to perform zero-shot learning
- learning_rate: learning rate
- dropout_rate: dropout rate to be applied by BERT output hidden
- max_steps: Maximum Minibatch Training Step
- eval_steps: How many steps to perform an evaluation
- early_stopping_patience: Patience steps to end
- learning after discovering the best model parameters
- run_mode: Can give train and test options

## Running HierarchicalContrast

Install all dependencies in config.py/config.json

You can run the dataset like this:

```python
python main.py --target_domain="dataset name"
```

## Citation
If you use our code or find HierarchicalContrast useful in your work, please cite our paper as:

	arXiv:2310.09135 [cs.AI]
 	(or arXiv:2310.09135v2 [cs.AI] for this version)
    https://doi.org/10.48550/arXiv.2310.09135
=======
# HiCL
The official code for EMNLP 2023 paper HiCL [HierarchicalContrast: A Coarse-to-Fine Contrastive Learning Framework for Cross-Domain Zero-Shot Slot Filling.](https://arxiv.org/pdf/2310.09135.pdf)
>>>>>>> b0d3138dd3d354232ef6a1fbb326a25d47184bff
