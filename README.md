# Representation Learning for High-dimensional Categorical Dataset

## Introduction

各个数据集的特征以及样本数量:

| 数据集  | 特征数量 | 样本数量 |
| ------- | -------- | -------- |
| 裴丹    | 42       | 10万+    |
| apascal | 64       | 1.2万+   |
| bank    | 62       | 4.1万+   |
| lung    | 3311     | 140+     |
| probe   | 34       | 6.4万+   |
| secom   | 590      | 1500+    |
| U2r     | 33       | 6万+     |


## Experimental Results

### Apascal

~~conducted with 200 epochs and 10 individual runs.~~, just use train.py to train model and use test.py to test 
model

| Criterion / Improvements | Ori  | Neither                        | Pairwise Loss                   | Momentum Updating                   | Both                               |
| ------------------------ | ---- | ------------------------------ | ------------------------------- | ----------------------------------- | ---------------------------------- |
| Sample Distance          | /    | AUROC: 0.8445<br/>AUPR: 0.0574 | AUROC: 0.8425<br/>AUPR: 0.0573  | AUC-ROC: 0.8237<br/>AUC-PR: 0.0472  | AUC-ROC: 0.8321<br/>AUC-PR: 0.0519 |
| LOF                      | /    | AUROC: 0.5193<br/>AUPR: 0.0133 | AUROC: 0.5039<br/> AUPR: 0.0130 | AUC-ROC: 0.5384<br/> AUC-PR: 0.0138 | AUC-ROC: 0.5281<br/>AUC-PR: 0.0136 |
| Isolation Forest         | /    | AUROC: 0.6341<br/>AUPR: 0.0254 | AUROC: 0.6745<br/>AUPR: 0.0343  | AUC-ROC: 0.7919<br/>AUC-PR: 0.0425  | AUC-ROC: 0.8011<br/>AUC-PR: 0.0470 |

### Bank

| Criterion / Improvements | Ori  | Neither                            | Pairwise Loss | Momentum Updating                  | Both                                |
| ------------------------ | ---- | ---------------------------------- | ------------- | ---------------------------------- | ----------------------------------- |
| Sample Distance          | /    | AUC-ROC: 0.7377<br/>AUC-PR: 0.3414 | /             | AUC-ROC: 0.7441<br/>AUC-PR: 0.3190 | AUC-ROC: 0.7435<br/> AUC-PR: 0.3336 |
| LOF                      | /    | /                                  | /             | /                                  | /                                   |
| Isolation Forest         | /    | AUC-ROC: 0.6892<br/>AUC-PR: 0.2975 | /             | AUC-ROC: 0.7268<br/>AUC-PR: 0.3172 | AUC-ROC: 0.7301<br/>AUC-PR: 0.3295  |

### Probe

| Criterion / Improvements | Ori  | Neither                            | Pairwise Loss | Momentum Updating                  | Both                               |
| ------------------------ | ---- | ---------------------------------- | ------------- | ---------------------------------- | ---------------------------------- |
| Sample Distance          | /    | AUC-ROC: 0.9260<br/>AUC-PR: 0.5999 | /             | AUC-ROC: 0.9880<br/>AUC-PR: 0.7782 | AUC-ROC: 0.9955<br/>AUC-PR: 0.9265 |
| LOF                      | /    | /                                  | /             | /                                  | /                                  |
| Isolation Forest         | /    | AUC-ROC: 0.9916<br/>AUC-PR: 0.8617 | /             | AUC-ROC: 0.9740<br/>AUC-PR: 0.6881 | AUC-ROC: 0.9818<br/>AUC-PR: 0.7351 |