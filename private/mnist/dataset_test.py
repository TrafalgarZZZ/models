import tensorflow as tf

parse_tensor_uint8 = lambda x: tf.io.parse_tensor(x, tf.uint8)

ds = tf.data.TFRecordDataset('train.tfrecord').map(parse_tensor_uint8).shuffle(buffer_size=1000).batch(64)

for x in ds.take(1):
    tf.print(x.shape)
