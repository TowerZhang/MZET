import numpy as np
import tensorflow as tf
from general_utils import get_dir

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def encode_to_tfrecords(mention, mention_emb, label, length, tfrecords_file):

    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    for men, men_emb, lab, length in zip(mention, mention_emb, label, length):
        if not isinstance(lab, list):
            lab = lab.tolist()
        # print(isinstance(men, list), len(men),
        #       isinstance(lab, list), len(lab))
        # print(lab)
        men_emb = np.array(men_emb, dtype=np.float32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'mention': _bytes_feature(tf.compat.as_bytes(men)),
            'mention_emb': _bytes_feature(men_emb.tostring()),
            'label': _float_feature(lab),
            'length': _int64_feature([length])
            }))
        writer.write(example.SerializeToString())
    writer.close()


def encode_to_tfrecords2(mention_emb, left, right, label, tfrecords_file):

    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    for men_emb, l_ctxt, r_ctxt, lab in zip(mention_emb, left, right, label):
        if not isinstance(lab, list):
            lab = lab.tolist()
        men_emb = np.array(men_emb, dtype=np.float32)
        l_ctxt = np.array(l_ctxt, dtype=np.float32)
        r_ctxt = np.array(r_ctxt, dtype=np.float32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'mention_emb': _bytes_feature(men_emb.tostring()),
            'left_context': _bytes_feature(l_ctxt.tostring()),
            'right_context': _bytes_feature(r_ctxt.tostring()),
            'label': _float_feature(lab)
            }))
        writer.write(example.SerializeToString())
    writer.close()


def encode_to_tfrecords3(mention, mention_emb, left, right, label, mention_len, tfrecords_file):

    writer = tf.python_io.TFRecordWriter(tfrecords_file)
    for men, l_ctxt, r_ctxt, lab, men_emb, length in zip(mention, left, right, label,
                                                         mention_emb, mention_len):
        if not isinstance(lab, list):
            lab = lab.tolist()
        men_emb = np.array(men_emb, dtype=np.float32)
        l_ctxt = np.array(l_ctxt, dtype=np.float32)
        r_ctxt = np.array(r_ctxt, dtype=np.float32)
        example = tf.train.Example(features=tf.train.Features(feature={
            'mention': _bytes_feature(tf.compat.as_bytes(men)),
            'mention_emb': _bytes_feature(men_emb.tostring()),
            'left_context': _bytes_feature(l_ctxt.tostring()),
            'right_context': _bytes_feature(r_ctxt.tostring()),
            'label': _float_feature(lab),
            'length': _int64_feature([length])
            }))
        writer.write(example.SerializeToString())
    writer.close()


def decode_from_tfrecords(filename_queue, timesteps, mention_emb_len,
                          label_emb_len, is_batch, batch=10):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'mention': tf.FixedLenFeature([], tf.string),
                                           'mention_emb': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([label_emb_len], tf.float32),
                                           'length': tf.FixedLenFeature([], tf.int64)
                                       })
    mention = features['mention']
    mention_emb = tf.decode_raw(features['mention_emb'], tf.float32)
    length = tf.cast(features['length'], tf.int64)
    mention_emb = tf.reshape(mention_emb, [timesteps, mention_emb_len])
    label = tf.cast(features['label'], tf.float32)

    if is_batch:
        batch_size = batch
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        mention, mention_emb, label, length = tf.train.shuffle_batch(
                                              [mention, mention_emb, label, length],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return mention, mention_emb, label, length


def decode_from_tfrecords2(filename_queue, window_size, avg_emb_len,
                          label_emb_len, is_batch, batch=36):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'mention_emb': tf.FixedLenFeature([], tf.string),
                                           'left_context': tf.FixedLenFeature([], tf.string),
                                           'right_context': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([label_emb_len], tf.float32)
                                       })
    mention_emb = tf.decode_raw(features['mention_emb'], tf.float32)
    mention_emb = tf.reshape(mention_emb, [1, avg_emb_len])
    left_context = tf.decode_raw(features['left_context'], tf.float32)
    left_context = tf.reshape(left_context, [window_size, avg_emb_len])
    right_context = tf.decode_raw(features['right_context'], tf.float32)
    right_context = tf.reshape(right_context, [window_size, avg_emb_len])
    label = tf.cast(features['label'], tf.float32)
    if is_batch:
        batch_size = batch
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        mention_emb, left_context, right_context, label = tf.train.shuffle_batch(
                                              [mention_emb, left_context, right_context, label],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return mention_emb, left_context, right_context, label


def decode_from_tfrecords3(filename_queue, window_size, men_emb_len, timesteps,
                          label_emb_len, is_batch, batch=10):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'mention': tf.FixedLenFeature([], tf.string),
                                           'mention_emb': tf.FixedLenFeature([], tf.string),
                                           'left_context': tf.FixedLenFeature([], tf.string),
                                           'right_context': tf.FixedLenFeature([], tf.string),
                                           'label': tf.FixedLenFeature([label_emb_len], tf.float32),
                                           'length': tf.FixedLenFeature([], tf.int64)
                                       })
    mention = features['mention']
    mention_emb = tf.decode_raw(features['mention_emb'], tf.float32)
    mention_emb = tf.reshape(mention_emb, [timesteps, men_emb_len])
    left_context = tf.decode_raw(features['left_context'], tf.float32)
    left_context = tf.reshape(left_context, [window_size, men_emb_len])
    right_context = tf.decode_raw(features['right_context'], tf.float32)
    right_context = tf.reshape(right_context, [window_size, men_emb_len])
    label = tf.cast(features['label'], tf.float32)
    length = tf.cast(features['length'], tf.int64)
    if is_batch:
        batch_size = batch
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        mention, mention_emb, left_context, right_context, label, length = tf.train.shuffle_batch(
                                [mention, mention_emb, left_context, right_context, label, length],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return mention, mention_emb, left_context, right_context, label, length


def decode_from_bertdata(filename_queue, max_seq_len, n_label, is_batch, batch=10):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_len], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_len], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_len], tf.int64),
        "label_ids": tf.FixedLenFeature([n_label], tf.int64)
    }
    features = tf.parse_single_example(serialized_example,
                                       name_to_features)
    input_ids = tf.cast(features['input_ids'], tf.int64)
    input_mask = tf.cast(features['input_mask'], tf.int64)
    segment_ids = tf.cast(features['segment_ids'], tf.int64)
    label_ids = tf.cast(features['label_ids'], tf.int64)
    if is_batch:
        batch_size = batch
        min_after_dequeue = 10
        capacity = min_after_dequeue + 3 * batch_size
        input_ids, input_mask, segment_ids, label_ids = tf.train.shuffle_batch(
                                [input_ids, input_mask, segment_ids, label_ids],
                                              batch_size=batch_size,
                                              num_threads=3,
                                              capacity=capacity,
                                              min_after_dequeue=min_after_dequeue)
    return input_ids, input_mask, segment_ids, label_ids


if __name__ == '__main__':
    train_filename = get_dir("BBN", "train_ctxt.tf.records")
    # test_filename = get_dir("BBN", "bert_output/train.tf_record")


    filename_queue = tf.train.string_input_producer([train_filename], num_epochs=None)
    # , mention_emb, left_context, right_context, label, length
    mention = decode_from_tfrecords3(
        filename_queue, 10, 768, 10,
                          16, is_batch=True, batch=10)

    # filename_queue = tf.train.string_input_producer([train_filename], num_epochs=None)
    # input_ids, input_mask, segment_ids, label_ids = decode_from_tfrecords2(filename_queue, 100, 16, is_batch=False)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        
        # while not coord.should_stop():
        for i in range(1):
            example = sess.run([mention])
            # print(set(np.where(example[0] == 1)[0]))
            print(example)
        

        coord.request_stop()
        coord.join(threads)
        sess.close()