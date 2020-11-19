import numpy as np
import tensorflow as tf

train, test = tf.keras.datasets.fashion_mnist.load_data()

images, labels = train
images = images / 255.0
labels = labels.astype(np.int32)

fmnist_train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
print(fmnist_train_ds)
fmnist_train_ds = fmnist_train_ds.shuffle(5000).batch(32)

# for example in fmnist_train_ds.take(1):
#     print(example)

print(fmnist_train_ds)
