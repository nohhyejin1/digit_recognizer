import os
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from PIL import Image

def train():
    data_path = os.environ.get('DATAPATH')
    if data_path is None:
            data_path = "../data/"
    model_path = os.environ.get('MODELPATH')
    if model_path is None:
            model_path = '../model/'
    lib_path = os.environ.get('LIBPATH')
    if lib_path is None:
        lib_path = '../lib/'

    sys.path.append(lib_path)
    from digit_recognizer import DigitRecognizer

    X, y = [], []
    for n in range(0,10):
        fnames = os.listdir(os.path.join(data_path, str(n)))
        for fname in fnames:
            f = os.path.join(data_path, str(n), fname)
            X.append(np.array(Image.open(f)))
            y.append(n)

    X = np.array(X)/255
    y = to_categorical(np.array(y))

    s = np.arange(X.shape[0])
    np.random.shuffle(s)

    X = X[s]
    y = y[s]

    train_size = int(len(y)*0.8)
    X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    model = DigitRecognizer()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    es = EarlyStopping(
        patience = 20
    )

    ckpt = ModelCheckpoint(
        model_path+"/model.h5",
        save_best_only = True,
        monitor = 'val_loss'
    )

    EPOCHS = 50
    BATCH_SIZE = 64

    history = model.fit(
        X_train, y_train,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        validation_data = (X_val, y_val),
        callbacks = [es, ckpt],
        verbose = 1
    )
    
    return history.history['accuracy'][-1]*100

if __name__ == "__main__":
    train()