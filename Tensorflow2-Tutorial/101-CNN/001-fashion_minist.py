# 导入tensorflow
import tensorflow.keras as keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 获取数据集
(train_images, train_labels), (
    test_images,
    test_labels,
) = keras.datasets.fashion_mnist.load_data()

# 初始化
train_images = train_images / 255.0
test_images = test_images / 255.0
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

x_shape = train_images.shape
model = keras.Sequential(
    [
        layers.Conv2D(
            input_shape=((x_shape[1], x_shape[2], x_shape[3])),
            filters=32,
            kernel_size=3,
            activation="relu",
        ),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Conv2D(
            filters=16,
            kernel_size=3,
            activation="relu",
        ),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    train_images, train_labels, batch_size=64, epochs=5, validation_split=0
)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["training", "valivation"], loc="upper left")
plt.show()

result = model.evaluate(test_images, test_labels)
