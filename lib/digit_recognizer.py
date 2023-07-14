from tensorflow import keras
from keras import layers

def DigitRecognizer():
    model = keras.Sequential([
        layers.Conv2D(1, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2),
        layers.Conv2D(8, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),
        layers.Conv2D(16, 3, activation='relu'),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model

if __name__ == "__main__":
    DigitRecognizer()