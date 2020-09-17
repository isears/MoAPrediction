from MoASubmitter import *
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import KFold
from tqdm.keras import TqdmCallback

import numpy as np

DEPTH = 5
EPOCHS = 100

def get_model():
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=np.shape(X)[1]))
    #model.add(Dropout(0.1))

    for idx in range(0, DEPTH):
        model.add(Dense(500, activation='relu'))
        #model.add(Dropout(0.1))

    model.add(Dense(np.shape(Y)[1], activation='sigmoid'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    return model


print("[*] Extracting data...")
X, Y = extract_xy('./data/train_features.csv', './data/train_targets_scored.csv')
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

for train, test in kfold.split(X, Y):
    curr_model = get_model()
    history = curr_model.fit(
        X[train],
        Y[train],
        epochs=EPOCHS,
        verbose=0,
        callbacks=[TqdmCallback(verbose=0)]
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