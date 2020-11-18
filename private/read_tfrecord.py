import tensorflow as tf

filenames = ['test.tfrecord']

raw_dataset = tf.data.TFRecordDataset(filenames)

for raw_record in raw_dataset.take(10):
    print(repr(raw_record))
