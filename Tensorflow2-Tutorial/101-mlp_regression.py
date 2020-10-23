# 导入tensorflow
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])
x_test = x_test.reshape([x_test.shape[0], -1])
print(x_train.shape, " ", y_train.shape)
print(x_test.shape, " ", y_test.shape)

# 配置模型
model = keras.Sequential(
    [
        layers.Dense(128, activation="relu", input_shape=(784,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Nadam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.summary()

history = model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.3, verbose=0)

result = model.evaluate(x_test, y_test)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()
