import os
import sys

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from PIL import Image
import yaml

def train():
    with open('config.yaml', mode='r', encoding='utf8') as file:
        config = yaml.load(file, Loader=yaml.Loader)

    data_path = config['data_path']
    model_path = config['model_path']
    lib_path = config['lib_path']

    epoch = config['epoch']
    batch_size = config['batch_size']
    patience = config['patience']

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

    X, y = X[s], y[s]

    train_size = int(len(y)*0.8)
    X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    model = DigitRecognizer()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    es = EarlyStopping(
        patience = patience
    )

    ckpt = ModelCheckpoint(
        model_path+"/model.h5",
        save_best_only = True,
        monitor = 'val_loss'
    )

    history = model.fit(
        X_train, y_train,
        epochs = epoch,
        batch_size = batch_size,
        validation_data = (X_val, y_val),
        callbacks = [es, ckpt],
        verbose = 1
    )
    
    return history.history['accuracy'][-1]*100

if __name__ == "__main__":
    train()