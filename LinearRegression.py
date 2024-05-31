import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from google.colab import drive

class MyLinearRegression:
    def __init__(self, learning_rate=0.001, epochs=100, regularization=None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        n_outputs = y.shape[1]

        self.weights = np.zeros((n_features, n_outputs))
        self.bias = np.zeros(n_outputs)

        for epoch in range(self.epochs):
            y_predicted = np.dot(X, self.weights) + self.bias
            errors = y_predicted - y
            dw = (1 / n_samples) * np.dot(X.T, errors)
            db = (1 / n_samples) * np.sum(errors, axis=0)

            if self.regularization:
                dw += self.regularization * self.weights

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 10 == 0:  # Print debug information every 10 epochs
                loss = mean_squared_error(y, y_predicted)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Weights: {self.weights.flatten()}, Bias: {self.bias}")

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def evaluate(self, X, y):
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        return mse, mae, r2

# Mount Google Drive
drive.mount('/content/drive', force_remount=True)

# File paths
file11 = [
    '/content/drive/MyDrive/NCKH_K16/Android13.csv',
    '/content/drive/MyDrive/NCKH_K16/Android11.csv',
    '/content/drive/MyDrive/NCKH_K16/Android12.csv'
]

results_list = []

for i in file11:
    print(f"Đang đọc dữ liệu {i}")
    data = pd.read_csv(i, low_memory=False)

    print("Đang chia dữ liệu thành các tập kiểm tra và huấn luyện")
    X = data.iloc[:, :-3].values  # Lấy tất cả các cột trừ 3 cột cuối
    y = data.iloc[:, -3:].values  # Lấy 3 cột cuối (nếu chỉ có 1 cột, vẫn hoạt động)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Đang chuẩn hóa dữ liệu")
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Đang huấn luyện mô hình")
    lr_model = MyLinearRegression(learning_rate=0.01, epochs=200, regularization=0.01)  # Adjusted learning rate, epochs, and added regularization
    lr_model.fit(X_train, y_train)

    mse, mae, r2 = lr_model.evaluate(X_test, y_test)
    print(f"MSE: {mse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")

    result = {'File': i, 'MSE': mse, 'MAE': mae, 'R-squared': r2}
    results_list.append(result)

# Save results to CSV
results_df = pd.DataFrame(results_list)
results_df.to_csv('/content/drive/MyDrive/NCKH_K16/results_lr_3T.csv', index=False)
