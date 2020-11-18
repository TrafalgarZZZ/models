import numpy as np
import tensorflow as tf

n_observations = int(1e4)

feature0 = np.random.choice([False, True], n_observations)
feature1 = np.random.randint(0, 5, n_observations)

strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

feature3 = np.random.randn(n_observations)

features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(feature0, feature1, feature2, feature3):
    feature = {
        'f0': _int64_feature(feature0),
        'f1': _int64_feature(feature1),
        'f2': _bytes_feature(feature2),
        'f3': _float_feature(feature3)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def generator():
    for features in features_dataset:
        yield serialize_example(*features)


serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=()
)

print(serialized_features_dataset)
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
