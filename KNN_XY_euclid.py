{"cells":[{"cell_type":"code","execution_count":5,"metadata":{"colab":{"base_uri":"https://localhost:8080/"},"id":"g8oLAGK6U1aa","outputId":"878b1262-5eb1-4243-e0b8-5a3556edf2b1","executionInfo":{"status":"ok","timestamp":1716863859294,"user_tz":-420,"elapsed":1203984,"user":{"displayName":"K16 NCKH","userId":"11588807329835209005"}}},"outputs":[{"output_type":"stream","name":"stdout","text":["Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n","Kich thuoc du lieu:  (7026, 1016)\n","Dữ liệu thiếu:\n"," -59          0\n","-60          0\n","-62          0\n","-77          0\n","-79          0\n","            ..\n","-100.1005    0\n","-100.1006    0\n","0            0\n","0.1          0\n","13           0\n","Length: 1016, dtype: int64\n","Du lieu vao:        -59  -60  -62  -77  -79  -81  -100  -100.1  -100.2  -100.3  ...  \\\n","0     -57  -61  -60 -100 -100  -85   -69     -76     -82     -83  ...   \n","1     -59  -61  -61  -80  -79  -86   -76     -75    -100    -100  ...   \n","2     -60  -64  -59  -78  -75  -84   -73     -76    -100     -88  ...   \n","3     -59  -62  -59  -77  -76 -100   -71    -100    -100    -100  ...   \n","4     -59  -65  -59  -76 -100 -100   -69     -80    -100     -86  ...   \n","...   ...  ...  ...  ...  ...  ...   ...     ...     ...     ...  ...   \n","7021 -100  -77 -100 -100  -67 -100  -100    -100    -100    -100  ...   \n","7022 -100  -76 -100 -100 -100 -100  -100    -100    -100    -100  ...   \n","7023 -100  -76 -100 -100  -65 -100  -100    -100    -100    -100  ...   \n","7024 -100 -100 -100 -100  -66 -100  -100    -100    -100    -100  ...   \n","7025 -100  -80  -85 -100  -65 -100  -100    -100    -100    -100  ...   \n","\n","      -100.997  -100.998  -100.999  -100.1000  -100.1001  -100.1002  \\\n","0         -100      -100      -100       -100       -100       -100   \n","1         -100      -100      -100       -100       -100       -100   \n","2         -100      -100      -100       -100       -100       -100   \n","3         -100      -100      -100       -100       -100       -100   \n","4         -100      -100      -100       -100       -100       -100   \n","...        ...       ...       ...        ...        ...        ...   \n","7021      -100      -100      -100       -100       -100       -100   \n","7022      -100      -100      -100       -100       -100       -100   \n","7023      -100      -100      -100       -100       -100       -100   \n","7024      -100      -100      -100       -100       -100       -100   \n","7025      -100      -100      -100       -100       -100        -89   \n","\n","      -100.1003  -100.1004  -100.1005  -100.1006  \n","0          -100       -100       -100       -100  \n","1          -100       -100       -100       -100  \n","2          -100       -100       -100       -100  \n","3          -100       -100       -100       -100  \n","4          -100       -100       -100       -100  \n","...         ...        ...        ...        ...  \n","7021       -100       -100       -100       -100  \n","7022       -100       -100       -100       -100  \n","7023       -100       -100       -100       -100  \n","7024       -100       -100       -100       -100  \n","7025       -100       -100       -100       -100  \n","\n","[7026 rows x 1013 columns]\n","Nhan:            0    0.1\n","0      0.00   0.00\n","1      0.00   0.00\n","2      0.00   0.00\n","3      0.00   0.00\n","4      0.00   0.00\n","...     ...    ...\n","7021  26.25  19.25\n","7022  26.25  19.25\n","7023  26.25  19.25\n","7024  26.25  19.25\n","7025  26.25  19.25\n","\n","[7026 rows x 2 columns]\n"]}],"source":["import pandas as pd\n","import numpy as np\n","import os\n","import time\n","from sklearn.model_selection import train_test_split\n","from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score\n","from google.colab import drive\n","\n","class KNN:\n","    def __init__(self, k=3):\n","        self.k = k\n","\n","    def euclidean_distance(self, pointA, pointB):\n","        return np.sqrt(np.sum(np.square(pointB - pointA), axis=1))\n","\n","    def fit(self, X_train, y_train):\n","        self.X_train = X_train\n","        self.y_train = y_train\n","\n","    def predict(self, X_test):\n","        y_pred = []\n","        for test_point in X_test.values:\n","            distances = self.euclidean_distance(self.X_train.values, test_point)\n","            k_nearest_indices = np.argsort(distances)[:self.k]\n","            k_nearest_labels = self.y_train.iloc[k_nearest_indices].values\n","            prediction = np.mean(k_nearest_labels, axis=0)\n","            y_pred.append(prediction)\n","        return np.array(y_pred)\n","\n","\n","    def score(self, X_test, y_test, save_predictions=False, predictions_path=None):\n","        y_pred = self.predict(X_test)\n","        mse = mean_squared_error(y_test.values, y_pred)\n","        rmse = np.sqrt(mse)\n","        mae = mean_absolute_error(y_test.values, y_pred)\n","        r2 = r2_score(y_test.values, y_pred)\n","        acc = np.mean(np.isclose(y_test.values, y_pred, atol=1))\n","\n","        if save_predictions and predictions_path:\n","            predictions_df = pd.DataFrame(y_pred, columns=y_test.columns)\n","            predictions_df.to_csv(predictions_path, index=False)\n","\n","        return mse, rmse, mae, r2, acc, y_pred\n","\n","drive.mount('/content/drive')\n","data = pd.read_csv('/content/drive/MyDrive/NCKH_K16/Tu_3tang_fn.csv')\n","print(\"Kich thuoc du lieu: \", data.shape)\n","\n","# Kiểm tra dữ liệu thiếu\n","missing_data = data.isnull().sum()\n","print(\"Dữ liệu thiếu:\\n\", missing_data)\n","\n","# Xóa các dòng chứa dữ liệu thiếu (nếu có)\n","data = data.dropna()\n","\n","# Lấy đầu vào và nhãn\n","features = data.iloc[:, :-3]\n","print(\"Du lieu vao: \", features)\n","labels = data.iloc[:, -3: -1]\n","print(\"Nhan: \", labels)\n","\n","# Phân chia dữ liệu thành tập huấn luyện và tập kiểm tra5\n","X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)\n","results_list = []\n","\n","for k in range(1, 20, 1):\n","    knn = KNN(k)\n","    knn.fit(X_train, y_train)\n","\n","    mse, rmse, mae, r2, acc, predictions = knn.score(X_test, y_test, save_predictions=True, predictions_path=f'/content/drive/MyDrive/NCKH_K16/predictions_Tu_knn_xy_3T_k_{k}.csv')\n","\n","    result = {'k': k, 'MSE': mse, 'RMSE': rmse, 'MAE' : mae, 'R2' : r2, 'ACC' : acc}\n","    results_list.append(result)\n","\n","# Save the results\n","results_df = pd.DataFrame(results_list)\n","results_df.to_csv('/content/drive/MyDrive/NCKH_K16/results_Tu_knn_xy_3T.csv', index=False)"]}],"metadata":{"colab":{"provenance":[]},"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":0}