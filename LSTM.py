{"cells":[{"cell_type":"code","execution_count":3,"metadata":{"colab":{"base_uri":"https://localhost:8080/"},"id":"dKUVVk2uL0Q2","executionInfo":{"status":"ok","timestamp":1716871880172,"user_tz":-420,"elapsed":809854,"user":{"displayName":"K16 NCKH","userId":"11588807329835209005"}},"outputId":"469e9582-04df-4d20-eb7e-003b616148ac"},"outputs":[{"output_type":"stream","name":"stdout","text":["Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount(\"/content/drive\", force_remount=True).\n","Epoch 1/50\n","1124/1124 [==============================] - 17s 14ms/step - loss: 21.0448\n","Epoch 2/50\n","1124/1124 [==============================] - 17s 15ms/step - loss: 5.8642\n","Epoch 3/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 4.7331\n","Epoch 4/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 3.9716\n","Epoch 5/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 3.5595\n","Epoch 6/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 3.1042\n","Epoch 7/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 2.8185\n","Epoch 8/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 2.5306\n","Epoch 9/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 2.3490\n","Epoch 10/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 2.1743\n","Epoch 11/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 2.0472\n","Epoch 12/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 1.9419\n","Epoch 13/50\n","1124/1124 [==============================] - 15s 13ms/step - loss: 1.8066\n","Epoch 14/50\n","1124/1124 [==============================] - 15s 13ms/step - loss: 1.7434\n","Epoch 15/50\n","1124/1124 [==============================] - 15s 13ms/step - loss: 1.6700\n","Epoch 16/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 1.5790\n","Epoch 17/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 1.4856\n","Epoch 18/50\n","1124/1124 [==============================] - 15s 13ms/step - loss: 1.4670\n","Epoch 19/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 1.3841\n","Epoch 20/50\n","1124/1124 [==============================] - 15s 13ms/step - loss: 1.3636\n","Epoch 21/50\n","1124/1124 [==============================] - 15s 13ms/step - loss: 1.3057\n","Epoch 22/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 1.2656\n","Epoch 23/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 1.2325\n","Epoch 24/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 1.1623\n","Epoch 25/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 1.1433\n","Epoch 26/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 1.1178\n","Epoch 27/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 1.0642\n","Epoch 28/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 1.0677\n","Epoch 29/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 1.0442\n","Epoch 30/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.9947\n","Epoch 31/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.9915\n","Epoch 32/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.9613\n","Epoch 33/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.9353\n","Epoch 34/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.9196\n","Epoch 35/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 0.9015\n","Epoch 36/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.9066\n","Epoch 37/50\n","1124/1124 [==============================] - 15s 13ms/step - loss: 0.8775\n","Epoch 38/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.8423\n","Epoch 39/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 0.8386\n","Epoch 40/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 0.8262\n","Epoch 41/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.7876\n","Epoch 42/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.8021\n","Epoch 43/50\n","1124/1124 [==============================] - 16s 15ms/step - loss: 0.7809\n","Epoch 44/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.7456\n","Epoch 45/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 0.7448\n","Epoch 46/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 0.7409\n","Epoch 47/50\n","1124/1124 [==============================] - 15s 14ms/step - loss: 0.7517\n","Epoch 48/50\n","1124/1124 [==============================] - 15s 13ms/step - loss: 0.7180\n","Epoch 49/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.7264\n","Epoch 50/50\n","1124/1124 [==============================] - 16s 14ms/step - loss: 0.7213\n"]},{"output_type":"stream","name":"stderr","text":["/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.\n","  saving_api.save_model(\n"]},{"output_type":"stream","name":"stdout","text":["44/44 [==============================] - 1s 4ms/step\n","44/44 [==============================] - 1s 4ms/step - loss: 1.6487\n","Mean Squared Error: 1.6487435102462769\n","Root Mean Square Error: 1.2840340767465157\n","Mean Absolute Error: 0.7053799328785928\n","R^2 Score: 0.9471652291225818\n","Accuracy: 0.009246088193456615\n"]}],"source":["import pandas as pd\n","import numpy as np\n","import time\n","import math\n","from sklearn.model_selection import train_test_split\n","from sklearn.preprocessing import MinMaxScaler\n","from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error\n","from keras.models import Sequential\n","from keras.layers import LSTM, Dropout, Dense\n","from google.colab import drive\n","\n","drive.mount('/content/drive')\n","data = pd.read_csv('/content/drive/MyDrive/NCKH_K16/Tu_3tang_fn.csv')\n","features = data.iloc[:, :-3]\n","labels = data.iloc[:, -3:]\n","\n","# Split data into train and test sets\n","X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)\n","\n","# Normalize the data\n","scaler = MinMaxScaler()\n","X_train = scaler.fit_transform(X_train)\n","X_test = scaler.transform(X_test)\n","\n","# Reshape data to 3D (samples, time steps, features)\n","X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])\n","X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])\n","\n","# Build the LSTM model\n","model = Sequential()\n","model.add(LSTM(240, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))\n","model.add(Dropout(0.3))  # To prevent overfitting\n","model.add(Dense(250, activation='relu'))\n","model.add(Dense(3, activation='linear'))  # Adjusted to output 3 values\n","\n","model.compile(optimizer='adam', loss='mse')\n","\n","# Train the model\n","model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)\n","\n","# Save the model\n","model.save('LSTM.h5')\n","\n","# Predict on test set\n","y_pred = model.predict(X_test)\n","\n","# Save predictions to CSV file\n","predictions_df = pd.DataFrame(y_pred, columns=y_test.columns)\n","predictions_df.to_csv('/content/drive/MyDrive/NCKH_K16/predictions_lstm_tu.csv', index=False)\n","\n","# Evaluate the model\n","mse = model.evaluate(X_test, y_test)\n","print('Mean Squared Error:', mse)\n","print('Root Mean Square Error:', math.sqrt(mse))\n","mae = mean_absolute_error(y_test, y_pred)\n","print('Mean Absolute Error:', mae)\n","\n","# Calculate R^2 score\n","print(\"R^2 Score:\", r2_score(y_test, y_pred))\n","\n","threshold = 0.1  # Define a threshold for acceptable error margin\n","accuracy = np.mean(np.all(np.abs(y_test - y_pred) < threshold, axis=1))\n","print('Accuracy:', accuracy)"]}],"metadata":{"colab":{"provenance":[],"mount_file_id":"1iX8i_D0PXwQKF28-A6nRs6tahcgP-Vm2","authorship_tag":"ABX9TyMBWLgIL6fPgGkfAlpLB3/a"},"kernelspec":{"display_name":"Python 3","name":"python3"},"language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":0}