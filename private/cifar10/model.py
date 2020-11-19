import tensorflow as tf
from tensorflow.keras import models, layers

import private.cifar10.tfrecord_preprocessor as preprocess

# Read data from tfrecord and preprocess each example
# filenames = ['train.tfrecord']
tfrecord_pattern = "../data/cifar10_tfrecords/train/*"
filenames = tf.data.Dataset.list_files(tfrecord_pattern, shuffle=True)
for filename in filenames:
    print(filename)
train_ds = preprocess.get_dataset_from_tfrecord(filenames=filenames).shuffle(1024).batch(32)

print(train_ds)

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(32, 32)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(10)
# ])

# x_train = x_train.map(lambda x: tf.reshape(x, shape=[32, 32, 3]))\
#     .map(lambda x: tf.dtypes.cast(x, tf.float32) / 255.0)

# train = x_train.from_tensor_slices((x_train, y_train))

# train = x_train.zip(y_train).map(lambda x, y: (x, y))

# for i, batch in enumerate(x_train)


# for batch in train.take(1):
#     print(batch.shape)
#     predictions = model(batch)
#     print(predictions)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_ds, epochs=1)

# for x, y in x_train, y_train:
#     predictions = model(x)
#     print(loss_fn(y, predictions))

# example_batch = list(x_train.batch(batch_size=16, drop_remainder=True).take(1))[0]
# example_batch = tf.dtypes.cast(example_batch, tf.float32) / 255.0
# print(example_batch.shape)

# example = list(ds.take(1))
# id = example['id']
# feature = tf.io.parse_tensor(example['image'], out_type=tf.uint8)
# feature = tf.dtypes.cast(feature, tf.float32) / 255.0
# label = example['label']
#
# predictions = model(example_batch)
# print(predictions)
