# 亞東醫院乳房攝影AI病徵偵測模型訓練

訓練資料統計

*陽性個案數(BIRADS 3, 4, 5)
Mass 72 cases
Calcifications 371 cases

*訓練個案數
Mass 56 cases
Calcifications 295 cases
Normal 161

*測試個案數
Mass 16 cases
Calcifications 72 cases
Normal 39

Detection CNN model: DenseNet-121

Multi-view CNN model: Multi-input DenseNet-121

Method: Patch-wise (Patch size: 224*224)

Data balance: Positive data augmentation
