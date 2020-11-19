import tensorflow as tf

feature_description = {
    'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
}


def _parse_function(example_proto):
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = _extract_feature(example)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [32, 32, 3])

    label = _extract_label(example)
    return image, label


def _extract_feature(example):
    return tf.io.parse_tensor(example['image'], out_type=tf.uint8)


def _extract_label(example):
    return example['label']


def get_dataset_from_tfrecord(filenames):
    ds = tf.data.TFRecordDataset(filenames=filenames)
    ds = ds.map(_parse_function)
    return ds
