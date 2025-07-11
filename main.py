import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv('data/ecg_data.csv')
df['ecg_smooth'] = df.groupby('patient_id')['ecg_signal'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
df['ecg_norm'] = (df['ecg_smooth'] - df['ecg_smooth'].mean()) / df['ecg_smooth'].std()

signals = df['ecg_norm'].values
labels = df['afib'].values
X, y = [], []
for i in range(len(signals) - 5):
    X.append(signals[i:i+5])
    y.append(int(labels[i:i+5].max()))
X = np.array(X).reshape(-1, 5, 1)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = Sequential()
model.add(Conv1D(8, 3, activation='relu', input_shape=(5,1)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split=0.1)

y_pred = (model.predict(X_test) > 0.5).astype(int).reshape(-1)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

model.save('results/afib_cnn_model.h5')
