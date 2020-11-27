import tensorflow as tf
import tensorflow_datasets as tfds


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


def serialize_example(example):
    id = example['id']
    image = example['image']
    label = example['label']

    feature = {
        'id': _bytes_feature(id.numpy()),
        'image': _bytes_feature(tf.io.serialize_tensor(image).numpy()),
        'label': _int64_feature(label.numpy())
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


# os.putenv("http_proxy", "http://172.17.0.1:10809")
# os.putenv("https_proxy", "http://172.17.0.1:10809")

ds = tfds.load('cifar10', shuffle_files=True, data_dir="../data")
train = ds['train']
test = ds['test']

feature_description = {
    'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


# for example in train.take(1):
#     parsed_example = _parse_function(serialize_example(example))
#     print(parsed_example['id'])
#     print(tf.io.parse_tensor(parsed_example['image'], out_type=tf.uint8))
#     print(parsed_example['label'])
#     print(parsed_example)

train_tfrecord_filename_template = "../data/cifar10_tfrecords_5g/train/train-%s"

# Write Cifar10 dataset and repeat 10 times to get several TFRecord files
for epoch in range(50):
    filename = train_tfrecord_filename_template % epoch
    print("Writing tfrecord file %s..." % filename)
    with tf.io.TFRecordWriter(filename) as writer:
        for example in train:
            writer.write(serialize_example(example))
