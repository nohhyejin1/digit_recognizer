from tensorflow import keras
from keras import layers

def DigitRecognizer():
    model = keras.Sequential([
        layers.Conv2D(128, 3, activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),

        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(16, 3, activation='relu'),
        layers.MaxPooling2D(2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    return model

if __name__ == "__main__":
    DigitRecognizer()