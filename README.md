# CgMCR2

This repository contians the Pytorch implementation of our paper in ACCV2024: ``[Graph Cut-guided Maximal Coding Rate Reduction for Learning Image Embedding and Clustering](https://link.springer.com/chapter/10.1007/978-981-96-0972-7_21#citeas)''

# Reference

@InProceedings{10.1007/978-981-96-0972-7_21,
author="He, Wei
and Huang, Zhiyuan
and Meng, Xianghan
and Qi, Xianbiao
and Xiao, Rong
and Li, Chun-Guang",
editor="Cho, Minsu
and Laptev, Ivan
and Tran, Du
and Yao, Angela
and Zha, Hongbin",
title="Graph Cut-Guided Maximal Coding Rate Reduction for Learning Image Embedding and Clustering",
booktitle="Computer Vision -- ACCV 2024",
year="2025",
publisher="Springer Nature Singapore",
address="Singapore",
pages="359--376",
isbn="978-981-96-0972-7"
}

# How to Start

An example for reproducing CgMCR2 on CIFAR-10 using CLIP-L/14 features:

> python main.py --optim adam --data cifar10 --lr 1e-4 --w_epo 10 --bs 512 --nz 400 --epo 20 --wd 1e-3 --gamma 30

# Feature Generation

To extract more CLIP/MoCo features, please refer to the codes in ./data folder.

> python preprocess.py --data imagenet

# Contact
Please contact [wei.he@bupt.edu.cn](wei.he@bupt.edu.cn) if you have any question on the codes.
