# 导入tensorflow
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data()
x_train = keras.preprocessing.sequence.pad_sequences(
    x_train, maxlen=200, padding="post"
)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=200, padding="post")
print(x_train.shape, " ", y_train.shape)
print(x_test.shape, " ", y_test.shape)

model = keras.Sequential(
    [
        layers.Embedding(input_dim=90000, output_dim=32, input_length=200),
        layers.LSTM(128, return_sequences=True),
        layers.BatchNormalization(),
        layers.LSTM(1, activation="sigmoid", return_sequences=False),
    ]
)
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=["accuracy"],
)
model.summary()
history = model.fit(x_train, y_train, batch_size=40, epochs=5, validation_split=0.1)

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["training", "valivation"], loc="upper left")
plt.show()

result = model.evaluate(x_test, y_test)
