## 亞東醫院乳房攝影AI病徵偵測模型訓練


![Image](https://github.com/yeahshow/Mammography-DenseNet-121/blob/master/sample_cal1.png)

![Image](https://github.com/yeahshow/Mammography-DenseNet-121/blob/master/sample_cal2.png)

# 訓練資料統計

- 陽性個案數(BIRADS 3, 4, 5): Mass 72 cases, Calcifications 371 cases
- 訓練個案數: Mass 56 cases, Calcifications 295 cases, Normal 161 cases
- 測試個案數: Mass 16 cases, Calcifications 72 cases, Normal 39 cases

# 模型輸入與輸出

Model
- Detection CNN model: DenseNet-121
- Multi-view CNN model: Multi-input DenseNet-121
- Method: Patch-wise (Patch size: 224*224)
- Data balance: Positive data augmentation


Detection
- Input: Image
- Output: Heatmap, ROI (lesion bounding boxes)

Multi-view false positive filter
- Input: Heatmaps (4 views)
- Output: Abnormal confidence value


# 計算方式

- 以單一乳篩案例（4 views）為單位，計算是否有正確偵測到其中的特定病徵，在陽性案例中偵測到ground truth病徵則該案例被視為TP，在陰性案例中沒有被偵測出任何FP病徵則該案例被視為TN。
- 如何判定「偵測到」病徵：AI偵測結果的中心需落在ground truth範圍內，或ground truth中心落在AI偵測結果範圍內。
- Ground truth定義: 依據亞東醫院判讀報告回標病徵位置作為ground truth

# 技術指標

![Image](https://github.com/yeahshow/Mammography-DenseNet-121/blob/master/ROC_mass.png)
![Image](https://github.com/yeahshow/Mammography-DenseNet-121/blob/master/ROC_cal.png)

- 仿照現行CAD使用習慣，列出數個門檻值（Threshold）的靈敏度（Sensitivity, 特異度（Specificity）
- 陰性影像中的偽陽性病徵數量（FP/Normal Image以及整體的ROC Curve, AUC數值
