import pandas as pd
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from google.colab import drive

class KNN:
    def __init__(self, k=3):
        self.k = k

    def euclidean_distance(self, pointA, pointB):
        return np.sqrt(np.sum(np.square(pointB - pointA), axis=1))

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = []
        for test_point in X_test.values:
            distances = self.euclidean_distance(self.X_train.values, test_point)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train.iloc[k_nearest_indices].values
            prediction = np.mean(k_nearest_labels, axis=0)
            y_pred.append(prediction)
        return np.array(y_pred)


    def score(self, X_test, y_test, save_predictions=False, predictions_path=None):
        y_pred = np.round(self.predict(X_test))
        mse = mean_squared_error(y_test.values, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test.values, y_pred)
        r2 = r2_score(y_test.values, y_pred)
        acc = np.mean(np.isclose(y_test.values, y_pred, atol=1))

        if save_predictions and predictions_path:
            predictions_df = pd.DataFrame(y_pred)
            predictions_df.to_csv(predictions_path, index=False)

        return mse, rmse, mae, r2, acc, y_pred

drive.mount('/content/drive')
data = pd.read_csv('/content/drive/MyDrive/NCKH_K16/Android12.csv')
print("Kich thuoc du lieu: ", data.shape)

# Kiểm tra dữ liệu thiếu
missing_data = data.isnull().sum()
print("Dữ liệu thiếu:\n", missing_data)

# Xóa các dòng chứa dữ liệu thiếu (nếu có)
data = data.dropna()

# Lấy đầu vào và nhãn
features = data.iloc[:, :-3]
print("Du lieu vao: ", features)
labels = data.iloc[:, -1]
print("Nhan: ", labels)

# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
results_list = []

for k in range(1, 20, 1):
    knn = KNN(k=k)
    knn.fit(X_train, y_train)

    mse, rmse, mae, r2, acc, predictions = knn.score(X_test, y_test, save_predictions=True, predictions_path=f'/content/drive/MyDrive/NCKH_K16/predictions_knn_z_3T_k_{k}.csv')

    result = {'k': k, 'MSE': mse, 'RMSE': rmse, 'MAE' : mae, 'R2' : r2, 'ACC' : acc}
    results_list.append(result)

# Save the results
results_df = pd.DataFrame(results_list)
results_df.to_csv('/content/drive/MyDrive/NCKH_K16/results_knn_z_3T.csv', index=False)