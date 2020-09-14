from MoASubmitter import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD

import numpy as np

print("[*] Extracting data...")
X, Y = extract_xy('./data/train_features.csv', './data/train_targets_scored.csv')
assert np.shape(X)[0] == np.shape(Y)[0]

print("[+] Extraction complete")
print(f'X shape: {np.shape(X)}')
print(f'\t{np.shape(X)[0]} training examples with {np.shape(X)[1]} features')
print(f'Y shape: {np.shape(Y)}')
print(f'\t{np.shape(Y)[0]} training examples with {np.shape(Y)[1]} possible labels')

print("[*] Building model..")

model = Sequential()
model.add(Dense(1000, activation='relu', input_dim=np.shape(X)[1]))
model.add(Dropout(0.1))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(np.shape(Y)[1], activation='sigmoid'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)

print("[+] Model built:")
model.summary()

print("[*] Fitting model...")
model.fit(X, Y, epochs=100)

print("[+] Fit done, saving..")
model.save('./models/model.h5')
print("[+] Model saved")
