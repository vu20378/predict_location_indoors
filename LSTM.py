import pandas as pd
import numpy as np
import time
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from google.colab import drive

drive.mount('/content/drive')
data = pd.read_csv('/content/drive/MyDrive/NCKH_K16/Android12.csv')
features = data.iloc[:, :-3]
labels = data.iloc[:, -3:]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data to 3D (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Build the LSTM model
model = Sequential()
model.add(LSTM(240, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))  # To prevent overfitting
model.add(Dense(250, activation='relu'))
model.add(Dense(3, activation='linear'))  # Adjusted to output 3 values

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)

# Save the model
model.save('LSTM.h5')

# Predict on test set
y_pred = model.predict(X_test)

# Save predictions to CSV file
predictions_df = pd.DataFrame(y_pred, columns=y_test.columns)
predictions_df.to_csv('/content/drive/MyDrive/NCKH_K16/predictions_lstm.csv', index=False)

# Evaluate the model
mse = model.evaluate(X_test, y_test)
print('Mean Squared Error:', mse)
print('Root Mean Square Error:', math.sqrt(mse))
mae = mean_absolute_error(y_test, y_pred)
print('Mean Absolute Error:', mae)

# Calculate R^2 score
print("R^2 Score:", r2_score(y_test, y_pred))

threshold = 0.1  # Define a threshold for acceptable error margin
accuracy = np.mean(np.all(np.abs(y_test - y_pred) < threshold, axis=1))
print('Accuracy:', accuracy)