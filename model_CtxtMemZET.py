"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
from general_utils import Progbar, write_file
import modeling
import numpy as np
import math
import optimization
import mention_tokenization as tokenization
import tensorflow as tf
from datetime import datetime
from tensorflow.python.estimator.model_fn import EstimatorSpec
from evaluation import load_raw_labels, evaluate
from tensorflow.contrib.layers.python.layers import initializers
from evaluation import load_raw_labels, evaluate, evaluation_level


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class EntityProcessor(DataProcessor):
    """Processor for the Entity data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(data_dir, "test")

    def _create_examples(self, data_dir, set_type):
        """Creates examples for the training and dev sets."""
        lines = self._read_tsv(data_dir)
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%d" % (set_type, int(line[0]))
            text_a = tokenization.convert_to_unicode(line[2])
            text_b = tokenization.convert_to_unicode(line[3])
            labels = [int(item) for item in line[1].split(",")]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=labels))
        return examples


def convert_single_example(ex_index, example, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=example.label)
    return feature


def file_based_convert_examples_to_features(
        examples, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        if isinstance(feature.label_id, list):
            label_ids = feature.label_id
        else:
            label_ids = feature.label_id[0]
        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training, label_length,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([label_length], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        #stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        #tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

class CtxtMemZET:
    def __init__(self, config):
        self.max_seq_length = config.max_seq_len
        self.tokenizer = tokenization.FullTokenizer(vocab_file=config.VOCAB_FILE, do_lower_case=True)
        self.batch_size = config.batch_size
        self.estimator = None
        self.processor = EntityProcessor()
        self.config = config
        self.logger = config.logger
        self.checkpoint_dir = self.config.dir_checkpoint
        self.checkpoint_path = self.config.dir_checkpoint + "/znet.ckpt"
        self.bert_config = modeling.BertConfig.from_json_file(self.config.BERT_CONFIG)
        self.train_examples = self.processor.get_train_examples(self.config.tsv_train_sampling)
        self.test_examples = self.processor.get_train_examples(self.config.tsv_test)
        self.num_train_steps = int(
            len(self.train_examples) / self.batch_size * self.config.nepochs)
        self.num_warmup_steps = int(self.num_train_steps * 0.1)
        self.is_training = True
        self.train_file = os.path.join(self.config.dir_bert_date, "train.tf_record")
        self.test_file = os.path.join(self.config.dir_bert_date, "test.tf_record")
        self.initializer = initializers.xavier_initializer()
        tf.logging.set_verbosity(tf.logging.INFO)

    def __creat_model(self):

        # embbeding layer
        self._init_bert_placeholder()
        self.bert_layer() 

        # bi-Lstm layer
        self.biLSTM_layer()

        self.add_memory_net_op()

        # logits_layer
        self.logits_layer()

        # loss_layer
        self.loss_layer()

        # optimizer_layer        
        self.bert_optimizer_layer()
    
    def _init_bert_placeholder(self):
        self.input_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, self.config.max_seq_len],
            name="bert_input_ids"
        )
        self.input_mask = tf.placeholder(
            dtype=tf.int32,
            shape=[None, self.config.max_seq_len],
            name="bert_input_mask"
        )
        self.segment_ids = tf.placeholder(
            dtype=tf.int32,
            shape=[None, self.config.max_seq_len],
            name="bert_segment_ids"
        )
        self.targets = tf.compat.v1.placeholder(
            dtype=tf.float32, 
            shape=(None, self.config.label_len_train), 
            name="targets"
        )
        self.targets_test = tf.compat.v1.placeholder(
            dtype=tf.float32, 
            shape=(None, self.config.label_len_test), 
            name="targets_test"
        )
        self.label_embedding_train = tf.compat.v1.constant(
            value=self.config.label_emb_train,
            dtype=tf.float32,
            name="label_embedding_train"
        )
        self.label_embedding_test = tf.compat.v1.constant(
            value=self.config.label_emb_test,
            dtype=tf.float32,
            name="label_embedding_test"
            )
        self.sim_matrix_train = tf.compat.v1.constant(
            value=self.config.sim_train,
            dtype=tf.float32, 
            name="sim_matrix_train_train"
        )        
        self.sim_matrix_test = tf.compat.v1.constant(
            value=self.config.sim_test,
            dtype=tf.float32, 
            name="sim_matrix_train_test"
        )
        self.dropout = tf.placeholder(
            dtype=tf.float32,
            shape=None,
            name="bert_dropout"
        )
        used = tf.sign(tf.abs(self.input_ids))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.length = tf.cast(length, tf.int32)
        self.nums_steps = tf.shape(self.input_ids)[-1]
        

    def bert_layer(self):
        model = modeling.BertModel(
            config=self.bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=False       
        )
        self.embedded = model.get_sequence_output() # model.get_pooled_output() 
        self.model_inputs = tf.nn.dropout(
            self.embedded, self.dropout
        )
        self.hidden_size = self.embedded.shape[-1].value

    def biLSTM_layer(self):
        with tf.variable_scope("bi-LSTM") as scope:
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = tf.contrib.rnn.GRUCell(
                        num_units=self.config.hidden_size_lstm,
                        # use_peepholes=True,
                        # initializer=self.initializer,
                        # state_is_tuple=True
                    )

            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_cell['forward'],
                cell_bw=lstm_cell['backward'],
                inputs=self.model_inputs,
                sequence_length=self.length,
                dtype=tf.float32,
            )
            lstm_outputs = tf.concat(outputs, axis=2)
            lstm_outputs = tf.nn.dropout(lstm_outputs, self.dropout)
            batch_size = tf.shape(lstm_outputs)[0]
            index = tf.range(0, batch_size) * self.nums_steps + (self.length - 1)
            # Indexing
            lstm_outputs = tf.reshape(lstm_outputs, [-1, 2 * self.config.hidden_size_lstm])
            self.lstm_outputs = tf.gather(lstm_outputs, index)

    def linear_layer(self, inputs, num_outputs, **kwargs):
        ''' Add a linear fully connected layer'''
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.005)
        return tf.contrib.layers.fully_connected(inputs, num_outputs,
                                                     activation_fn=None, weights_regularizer=regularizer, **kwargs)

    def add_memory_net_op(self):
        memory_size = self.config.label_len_train
        memory_embedding_size = 300

        with tf.variable_scope('u_memory'):
            u = self.linear_layer(self.lstm_outputs, memory_embedding_size)
            variable_summaries(u)

        with tf.name_scope('input_memory'):
            input_memory = self.label_embedding_train
            label_memory = tf.nn.dropout(input_memory, self.config.dropout)

        with tf.variable_scope('matrix_memory'):
            matrix_mem = self.linear_layer(label_memory, memory_embedding_size)
            variable_summaries(matrix_mem)

        with tf.name_scope('pi_k_memory'):
            score = tf.matmul(u, tf.transpose(matrix_mem))
            pi_k = tf.nn.softmax(score)
            # self.variable_summaries(pi_k)

        with tf.variable_scope('C_memory'):
            C = self.linear_layer(label_memory, memory_embedding_size)
            variable_summaries(C)

        with tf.name_scope('o_memory'):
            o = tf.matmul(pi_k, C)
            variable_summaries(o)

        with tf.name_scope('a'):
            sum_out = o + u
            self.a = self.linear_layer(sum_out, memory_size)
            variable_summaries(self.a)
    
    # def add_multi_memory_net_op(self):
    #     memory_size = self.config.label_len_train
    #     memory_embedding_size = 300
    #     hop = 2

    #     u = self.linear_layer(self.concat_hidden, memory_embedding_size)
    #     input_memory = self.label_embedding_train
    #     label_memory = tf.nn.dropout(input_memory, self.config.dropout)
    #     sum_out = None
    #     for hopn in range(hop):
    #         matrix_mem = self.linear_layer(label_memory, memory_embedding_size)
    #         score = tf.matmul(u, tf.transpose(matrix_mem))
    #         pi_k = tf.nn.softmax(score)
    #         C = self.linear_layer(label_memory, memory_embedding_size)
    #         o = tf.matmul(pi_k, C)
    #         sum_out = o + u
    #         u = sum_out
    #     self.a = self.linear_layer(sum_out, memory_size)

    def logits_layer(self):

        # with tf.variable_scope("hidden"):
        #     w = tf.get_variable("W", shape=[self.config.hidden_size_lstm*2, self.config.hidden_size_lstm],
        #                         dtype=tf.float32, initializer=self.initializer
        #                         )
        #     b = tf.get_variable("b", shape=[self.config.hidden_size_lstm], dtype=tf.float32,
        #                         initializer=self.initializer
        #                         )
        #     hidden = tf.tanh(tf.nn.xw_plus_b(self.lstm_outputs, w, b))
        #     self.hidden = hidden

        # with tf.variable_scope("logits"):
        #     w = tf.get_variable("W", shape=[self.config.hidden_size_lstm, self.config.n_label_emb],
        #                         dtype=tf.float32, initializer=self.initializer
        #                         )
        #     b = tf.get_variable("b", shape=[self.config.n_label_emb], dtype=tf.float32,
        #                         initializer=self.initializer
        #                         )
        #     out = tf.tanh(tf.nn.xw_plus_b(self.hidden, w, b))
        #     self.logits_train = tf.matmul(out, tf.transpose(self.label_embedding_train, perm=[1, 0]))
        #     self.logits_test = tf.matmul(out, tf.transpose(self.label_embedding_test, perm=[1, 0]))
        #     # self.outputs_train = tf.nn.softmax(self.logits_train, name="outputs_train")
        #     # self.outputs_test = tf.nn.softmax(self.logits_test, name="outputs_test")
        #     self.outputs_train = tf.sigmoid(self.logits_train, name="outputs_train")
        #     self.outputs_test = tf.sigmoid(self.logits_test, name="outputs_test")

        self.logits = tf.matmul(self.a, self.sim_matrix_train)
        self.outputs_train = tf.nn.softmax(self.logits)
        # self.outputs_train = tf.sigmoid(self.logits)
        variable_summaries(self.outputs_train)
        self.logits_test = tf.matmul(self.a, self.sim_matrix_test)
        self.outputs_test = tf.nn.softmax(self.logits_test)
        # self.outputs_test = tf.sigmoid(self.logits_test)

        with tf.name_scope('Label_pred'):
            self.labels_pred_train = tf.cast(tf.argmax(self.outputs_train, axis=1), tf.int32)
            self.labels_pred_test = tf.cast(tf.argmax(self.outputs_test, axis=1), tf.int32)
    
    def loss_layer(self):
        with tf.variable_scope("loss_layer"):
            pos_label = tf.ones_like(self.targets)
            neg_label = -tf.ones_like(self.targets)
            IND = tf.where(tf.equal(self.targets, 1), x=pos_label, y=neg_label)
            max_margin_loss = tf.square(tf.maximum(0.0, 1 - tf.multiply(IND, self.outputs_train)))
            self.loss = tf.reduce_mean(max_margin_loss)

            # per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.targets, tf.float32),
            #                                                            logits=self.outputs_train)
            # self.loss = tf.reduce_mean(per_example_loss)
    
    def bert_optimizer_layer(self):
        num_train_steps = int(
            len(self.train_examples) / self.batch_size * self.config.nepochs)
        num_warmup_steps = int(num_train_steps * 0.1)
        self.train_op = optimization.create_optimizer(
            self.loss, self.config.lr, num_train_steps, num_warmup_steps, False
        )
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def add_summary(self, sess):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_checkpoint,
                sess.graph)

    def get_data_processing(self):
        file_based_convert_examples_to_features(self.train_examples, self.config.max_seq_len,
                                                self.tokenizer,
                                                self.train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(self.train_examples))
        tf.logging.info("  Batch size = %d", self.config.batch_size)
        tf.logging.info("  Num steps = %d", self.num_train_steps)

        file_based_convert_examples_to_features(
            self.test_examples, self.max_seq_length, self.tokenizer, self.test_file)

        tf.logging.info("***** Running testing *****")
        tf.logging.info("  Num examples = %d", len(self.test_examples))
        tf.logging.info("  Batch size = %d", self.batch_size)

    def build(self):
        self.get_data_processing()

    def train(self):
        self.__creat_model()
        length = float(len(self.train_examples))
        nbatches = 100 # math.ceil(length/self.batch_size)
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("restore model")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())

            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = \
                modeling.get_assignment_map_from_checkpoint(tvars,
                                                            self.config.BERT_CHECKPOINT)
            tf.train.init_from_checkpoint(self.config.BERT_CHECKPOINT, assignment_map)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                print("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            self.add_summary(sess)
            for i in range(self.config.nepochs):
                # print("-"*50)
                # print("epoch {}".format(i))
                prog = Progbar(target=nbatches)
                steps = 0
                
                for j in range(nbatches):
                    steps += 1
                    train = [self.config.input_ids_train, self.config.input_mask_train, 
                             self.config.segment_ids_train, self.config.label_ids_train]
                    inputs_ids, segment_ids, input_mask, label_ids = sess.run(train)

                    feed = {
                        self.input_ids: inputs_ids,
                        self.targets: label_ids,
                        self.segment_ids: segment_ids,
                        self.input_mask: input_mask,
                        self.dropout: 0.5
                    }
                   
                    embedding, loss, _, logits, summary = sess.run(
                        [self.embedded, self.loss, self.train_op, self.labels_pred_train, self.merged],
                        feed_dict=feed)
                    prog.update(j + 1, [("train loss", loss)])
                    print(logits)
                    if j % 10 == 0:
                        self.file_writer.add_summary(summary, i * nbatches + j)

                self.config.lr *= self.config.lr_decay
                
                self.saver.save(sess, self.checkpoint_path)
            coord.request_stop()
            coord.join(threads)
            sess.close()

    def evaluate_all(self):

        self.__creat_model()
        length = float(len(self.test_examples))
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print("restore model")
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            
            test = [self.config.input_ids_train, self.config.input_mask_train,
                             self.config.segment_ids_train, self.config.label_ids_train]
                # [self.config.input_ids_test, self.config.input_mask_test,
                #     self.config.segment_ids_test, self.config.label_ids_test]
            inputs_ids, segment_ids, input_mask, label = sess.run(test)
            feed = {
                        self.input_ids: inputs_ids,
                        self.targets: label,
                        self.segment_ids: segment_ids,
                        self.input_mask: input_mask,
                        self.dropout: 1
                    }
            labels_pred = sess.run([self.labels_pred_train], feed_dict=feed)
            # write_file(self.config.dir_output + "/label.txt", list(labels_pred), label)
            labels_pred = load_raw_labels(self.config.supertypefile_common, labels_pred[0])
            print(labels_pred)
            temp = []
            for x in label:
                print(np.where(np.array(x) == 1))
            strict_acc, mi_f1, ma_f1, mi_p, mi_r, ma_p, ma_r = evaluate(labels_pred, label)
            msg = "Test Evaluation on Overall---\n"
            msg += "Overall---strict_acc: {:.4f}, mi_f1: {:.4f}, ma_f1: {:.4f}, ".format(strict_acc, mi_f1, ma_f1)
            msg += "mi_p: {:.4f}, mi_r: {:.4f}, ma_p: {:.4f}, ma_r: {:.4f}\n".format(mi_p, mi_r, ma_p, ma_r)
            self.logger.info(msg)
            
            coord.request_stop()
            coord.join(threads)
            sess.close()