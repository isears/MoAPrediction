from LocalUtil import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import KFold
from tqdm.keras import TqdmCallback
from datetime import datetime
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf

DEPTH = 1
EPOCHS = 100
DROPOUT = 0.5


def get_model():
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=np.shape(X)[1]))
    model.add(Dropout(DROPOUT))

    for idx in range(0, DEPTH):
        model.add(Dense(800, activation='relu'))
        model.add(Dropout(DROPOUT))

    assert np.shape(Y)[1] == 1, "[-] This should be binary classification, but output layer is not 1"
    model.add(Dense(np.shape(Y)[1], activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adm = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=adm, metrics=['accuracy'])

    return model


print("[*] Extracting data...")
# X, Y = extract_xy_nfkbi('./data/train_features.csv', './data/train_targets_scored.csv')
X, Y = load_cache('./data/cache/X_nfkbi.npy', './data/cache/Y_nfkbi.npy')

assert np.shape(X)[0] == np.shape(Y)[0]

print("[+] Extraction complete")
print(f'X shape: {np.shape(X)}')
print(f'\t{np.shape(X)[0]} training examples with {np.shape(X)[1]} features')
print(f'Y shape: {np.shape(Y)}')
print(f'\t{np.shape(Y)[0]} training examples with {np.shape(Y)[1]} possible labels')


start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


curr_model = get_model()

tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=f'logs/SingleClassifierNFKBI',
    histogram_freq=1
)

history = curr_model.fit(
    X_train,
    Y_train,
    epochs=EPOCHS,
    verbose=0,
    validation_data=(X_test, Y_test),
    callbacks=[TqdmCallback(verbose=0), tb_callback]
)

print('[*] Training model on all data...')
full_data_model = get_model()
full_data_model.fit(X, Y, epochs=EPOCHS, verbose=0, callbacks=[TqdmCallback(verbose=0)])

full_data_model.save('./models/nkbi-full-data-model.h5')