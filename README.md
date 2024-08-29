# PU_Learning_Notebook

### DATASET

- UCI
  - 



### Selection Bias

- PUSB
- SAR-EM
- PUe
- LBE-LF
- Local Certainty(LC,Recovering)
- VAE-PU
- PU-EM(Learning from Positive and Unlabeled Data under the Selected At Random Assumption)

## PUSB

[仓库地址](https://github.com/MasaKat0/PUlearning/tree/master)

1. selection bias 策略问题dataset_linear.py里面是不是人工的标注选择偏差
2. 要把评分标准统一

使用的数据集：

- UCI(mushrooms、shuttle、pageblocks、usps、connect-4、spambase)
- CIFAR-10
- MNIST
- SwissProt的文档数据集

其中MNIST被预处理为0、2、4、6、8构成正类，而1、3、5、7、9构成负类；对于CIFAR-10，正类数据由“airplane”、“automobile”、“ship”和“truck”构成，负类数据由“bird”、“cat”、“deer”、“dog”、“frog”和“horse”构成。

## SAR-EM

[仓库地址](https://github.com/ML-KULeuven/SAR-PU)

使用的数据集：

- UCI(20 News Groups、Cover Type、Diabetes、Image Segmentation、Adult、Breast Cancer、Mushroom)
- Splice(from SVMLIB DATA repo)

需要增加

- CIFAR-10
- MNIST

## PUe

代码在想办法要

- MNIST
- CIFAR-10
-  Alzheimer

## LBE-LF

代码已复现，并正在要官方版本

- UCI
- USPS(类似于*MNIST*)
- HockeyFight
- SwissProt

## lC

冇代码

- UCI
- Yelp Reviews
- PASCAL VOC 2007



## VAE-PU

[仓库地址](https://github.com/byeonghu-na/vae-pu)

- MNIST
- CIFAR-10
- 20 Newsgroups

## PU-EM

无代码

- UCI
