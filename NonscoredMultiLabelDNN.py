from MoASubmitter import *
from LocalUtil import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD, Adam
from sklearn.model_selection import KFold
from tqdm.keras import TqdmCallback
from datetime import datetime

import numpy as np
import tensorflow as tf

DEPTH = 5
EPOCHS = 30

def get_model():
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=np.shape(X)[1]))
    #model.add(Dropout(0.3))

    for idx in range(0, DEPTH):
        model.add(Dense(500, activation='relu'))
        #model.add(Dropout(0.3))

    model.add(Dense(np.shape(Y)[1], activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adm = Adam()
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    return model


print("[*] Extracting data...")
X, Y = load_cache('./data/cache/X.npy', './data/cache/Y_nonscored.npy')
assert np.shape(X)[0] == np.shape(Y)[0]

print("[+] Extraction complete")
print(f'X shape: {np.shape(X)}')
print(f'\t{np.shape(X)[0]} training examples with {np.shape(X)[1]} features')
print(f'Y shape: {np.shape(Y)}')
print(f'\t{np.shape(Y)[0]} training examples with {np.shape(Y)[1]} possible labels')

print("[*] Running k-fold cross validation")
kfold = KFold(n_splits=5, shuffle=True)
fold_idx = 0
all_scores = list()
bestscore = np.inf

start_time = datetime.now().strftime("%Y%m%d-%H%M%S")

for train, test in kfold.split(X, Y):
    curr_model = get_model()

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=f'logs/{start_time}-cv{fold_idx}',
        histogram_freq=1
    )

    history = curr_model.fit(
        X[train],
        Y[train],
        epochs=EPOCHS,
        verbose=0,
        validation_data=(X[test], Y[test]),
        callbacks=[TqdmCallback(verbose=0), tb_callback]
    )

    score = curr_model.evaluate(X[test], Y[test])
    print(f'Score for fold {fold_idx}: {score}')

    curr_model.save(f'./models/model-{fold_idx}.h5')

    if score < bestscore:
        print("[+] Best-yet model, saving...")
        bestscore = score
        curr_model.save("./models/model-best.h5")

    all_scores.append(score)

    fold_idx += 1

print(f'[+] Completed CV, average score: {sum(all_scores) / len(all_scores)}')
print('[*] Training model on all data...')
full_data_model = get_model()
history = full_data_model.fit(X, Y, epochs=EPOCHS, verbose=0, callbacks=[TqdmCallback(verbose=0)])

full_data_model.save('./models/full-data-model.h5')