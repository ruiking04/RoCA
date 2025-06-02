# RoCA: Robust Contrastive One-class Time Series Anomaly Detection with Contaminated Data
This repository provides the implementation of the _RoCA: Robust Contrastive One-class Time Series Anomaly Detection with Contaminated Data_ method, called _RoCA_ below. 

## Abstract
> The increasing volume of time-series signals and the scarcity of labels make time-series anomaly detection a natural 
> fit for self-supervised deep learning. However, existing normality-based approaches face two key limitations: 
> (1) relying on a single assumption often fails to capture the whole normal patterns, leading to biased representations; 
> and (2) they typically presume clean training data, which is unrealistic in practice and undermines model robustness. 
> In this paper, we propose RoCA, a unified and robust anomaly detection framework that simultaneously addresses assumption incompleteness and data contamination. 
> The key insight is that normal samples tend to satisfy multiple normality assumptions, whereas anomalous or contaminated samples should violate at least one. 
> RoCA employs a composite loss function consisting of a multi-normality alignment term, a dynamic abnormality-aware term, 
> and a variance regularization term to maintain training stability. This design enables RoCA to dynamically discover 
> and isolate latent anomalies during training, without requiring clean supervision.
> Extensive experiments on both univariate and multivariate time-series benchmarks demonstrate that RoCA consistently 
> outperforms state-of-the-art methods, achieving up to 7.3% improvement under real-world contamination. 
> Our theoretical analysis further reveals the intrinsic synergy between contrastive learning and one-class classification under the RoCA framework.



## Citation
Link to our paper xxx.
If you use this code for your research, please cite our paper:

```
We will add citation information later.
```

## Installation
This code is based on `Python 3.8`, all requirements are written in `requirements.txt`. Additionally, we should install `saleforce-merlion v1.1.1` and `ts_dataset` as Merlion suggested.

```
pip install salesforce-merlion==1.1.1
pip install -r requirements.txt
```

## Dataset
We acknowledge the dataset's contributors, including AIOps, UCR, SWaT, and WADI.
This repository already includes Merlion's data loading package `ts_datasets`.

### AIOps (KPI, IOpsCompetition) and UCR. 
1. AIOps Link: https://github.com/NetManAIOps/KPI-Anomaly-Detection
2. UCR Link: https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/ 
and https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip
3. Download and unzip the data in `data/iops_competition` and `data/ucr` respectively. 
e.g. For AIOps, download `phase2.zip` and unzip the `data/iops_competition/phase2.zip` before running the program.

### SWaT and WADI. 
1. For SWaT and WADI, you need to apply by their official tutorial. Link: https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
2. Because multiple versions of these two datasets exist, 
we used their newer versions: `SWaT.SWaT.A2_Dec2015, version 0` and `WADI.A2_19Nov2019`.
3. Download and unzip the data in `data/swat` and `data/wadi` respectively. Then run the 
`swat_preprocessing()` and `wadi_preprocessing()` functions in `dataloader/data_preprocessing.py` for preprocessing.

## Repository Structure

### `conf`
This directory contains experiment parameters for all models on AIOps (as `IOpsCompetition` in the code), UCR, SWaT, and WADI datasets.

### `models`
Source code of the RoCA model.

### `data`
Processed datasets. Such as data/UCR, data/WADI.

### `results`
Directory where the experiment result is saved.

## RoCA Usage
```
# RoCA Method (dataset_name: IOpsCompetition, UCR, SWaT, WADI)
python roca.py --selected_dataset <dataset_name> --device cuda --seed 2
```

## Baselines
Anomaly Transformer(AnoTrans, AOT), AOC, RandomScore(RAS), NCAD, LSTMED, OC_SVM, IF, SR, RRCF, SVDD, DAMP, TS_AD(TCC)

We reiterate that in addition to our method, the source code of other baselines is based on the GitHub source code 
provided by their papers. For reproducibility, we changed the source code of their models as little as possible. 
We are grateful for the work on these papers.

We consult the GitHub source code of the paper corresponding to the baseline and then reproduce it. 
For baselines that use the same datasets as ours, we use their own recommended hyperparameters. 
For different datasets, we use the same hyperparameter optimization method Grid Search as our model to find the optimal hyperparameters.

### Acknowledgements
Part of the code, especially the baseline code, is based on the following source code.
1. [Anomaly Transformer(AOT)](https://github.com/thuml/Anomaly-Transformer)
2. [AOC](https://github.com/alsike22/AOC)
3. [Deep-SVDD-PyTorch](https://github.com/lukasruff/Deep-SVDD-PyTorch)
4. [TS-TCC](https://github.com/emadeldeen24/TS-TCC)
5. [DAMP](https://sites.google.com/view/discord-aware-matrix-profile/documentation) and 
[DAMP-python](https://github.com/sihohan/DAMP)
6. LSTM_ED, SR, and IF are reproduced based on [saleforce-merlion](https://github.com/salesforce/Merlion/tree/main/merlion/models/anomaly)
7. [RRCF](https://github.com/kLabUM/rrcf?tab=readme-ov-file)
8. [Metrics:affiliation-metrics](https://github.com/ahstat/affiliation-metrics-py)

