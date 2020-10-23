from tensorflow.keras import layers
import tensorflow.keras as keras
import matplotlib.pyplot as plt

inputs = keras.Input(shape=(32, 32, 3), name="img")
h1 = layers.Conv2D(32, 3, activation="relu")(inputs)
h1 = layers.Conv2D(64, 3, activation="relu")(h1)
block1_out = layers.MaxPooling2D(3)(h1)

h2 = layers.Conv2D(64, 3, activation="relu", padding="same")(block1_out)
h2 = layers.Conv2D(64, 3, activation="relu", padding="same")(h2)
block2_out = layers.add([h2, block1_out])

h3 = layers.Conv2D(64, 3, activation="relu", padding="same")(block2_out)
h3 = layers.Conv2D(64, 3, activation="relu", padding="same")(h3)
block3_out = layers.add([h3, block2_out])

h4 = layers.Conv2D(64, 3, activation="relu")(block3_out)
h4 = layers.GlobalMaxPool2D()(h4)
h4 = layers.Dense(256, activation="relu")(h4)
h4 = layers.Dropout(0.5)(h4)
outputs = layers.Dense(10, activation="softmax")(h4)

model = keras.Model(inputs, outputs, name="test_resnet")
model.summary()
keras.utils.plot_model(model, "small_resnet_model.png", show_shapes=True)


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
history = model.fit(x_train, y_train, batch_size=64, epochs=20, validation_split=0.2)
result = model.evaluate(x_test, y_test)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()
