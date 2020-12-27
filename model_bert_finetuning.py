"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import mention_tokenization as tokenization
import tensorflow as tf
from datetime import datetime
from tensorflow.python.estimator.model_fn import EstimatorSpec
from evaluation import load_raw_labels, evaluate


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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, model_config, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [model_config.n_label_emb, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [model_config.n_label_emb], initializer=tf.zeros_initializer())

    # label_weights = tf.get_variable(
    #     "label_weights", [model_config.n_label_emb, model_config.n_label_emb],
    #     initializer=tf.truncated_normal_initializer(stddev=0.02))

    # label_bias = tf.get_variable(
    #     "label_bias", [model_config.n_label_emb], initializer=tf.zeros_initializer())

    label_embedding_train = tf.compat.v1.constant(model_config.label_emb_train,
                    dtype=tf.float32, name="label_embedding_train")
    label_embedding_test = tf.compat.v1.constant(value=model_config.label_emb_test,
                    dtype=tf.float32, name="label_embedding_test")

    with tf.variable_scope("Logits"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        feature_layer = tf.matmul(output_layer, output_weights, transpose_b=True)
        feature_layer = tf.nn.relu(tf.nn.bias_add(feature_layer, output_bias))
        # label_layer = tf.matmul(label_embedding, label_weights)
        # label_layer = tf.nn.relu(tf.nn.bias_add(label_layer, label_bias))
        logits = tf.matmul(feature_layer, label_embedding_train, transpose_b=True)
        logits_test = tf.matmul(feature_layer, label_embedding_test, transpose_b=True)
        probabilities = tf.nn.softmax(logits, name="outputs")
        probabilities_test = tf.nn.softmax(logits_test, name="outputs")
        # probabilities = tf.sigmoid(logits, name="outputs")
        # probabilities_test = tf.sigmoid(logits_test, name="outputs")

    with tf.name_scope('Label_pred'):
        labels_pred = tf.cast(tf.argmax(probabilities, axis=1), tf.int32)
        labels_pred_test = tf.cast(tf.argmax(probabilities_test, axis=1), tf.int32)

    if is_training:
        with tf.name_scope('loss'):
            # pos_label = tf.ones_like(labels)
            # neg_label = -tf.ones_like(labels)
            # IND = tf.where(tf.equal(labels, 1), x=pos_label, y=neg_label)
            # print(IND.shape)
            # max_margin_loss = tf.maximum(0.0, 0.1 - tf.multiply(tf.cast(IND, tf.float32), probabilities))
            # loss = tf.reduce_mean(max_margin_loss)
            per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32),
                                                                       logits=logits)
            loss = tf.reduce_mean(per_example_loss)
        
            return (loss, probabilities, labels_pred) 
    else:
        return (logits_test, labels_pred_test)


def model_fn_builder(bert_config, model_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if is_training:
            (total_loss, probabilities, labels_pred) = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                model_config, use_one_hot_embeddings)
        else:
            (probabilities, labels_pred) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            model_config, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

            output_spec = EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_ids, probabilities):
                logits_split = tf.split(probabilities, label_ids.shape[1], axis=-1)
                label_ids_split = tf.split(label_ids, label_ids.shape[1], axis=-1)
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    current_auc, update_op_auc = tf.metrics.accuracy(label_id_, logits)
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                return eval_dict

            eval_metrics = metric_fn(label_ids, probabilities)
            output_spec = EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics)

        else:
            output_spec = EstimatorSpec(
                mode=mode, predictions={
                    "probabilities": probabilities,
                    "pred_label": labels_pred
                })
        return output_spec

    return model_fn


class bertZET:
    def __init__(self, config):
        self.max_seq_length = config.max_seq_len
        self.tokenizer = tokenization.FullTokenizer(vocab_file=config.VOCAB_FILE, do_lower_case=True)
        self.batch_size = config.batch_size
        self.estimator = None
        self.processor = EntityProcessor()
        self.config = config
        self.bert_config = modeling.BertConfig.from_json_file(self.config.BERT_CONFIG)
        self.train_examples = self.processor.get_train_examples(self.config.tsv_train)
        self.num_train_steps = int(
            len(self.train_examples) / self.batch_size * self.config.nepochs)
        self.num_warmup_steps = int(self.num_train_steps * 0.1)
        self.train_file = os.path.join(self.config.dir_checkpoint, "train.tf_record")
        tf.logging.set_verbosity(tf.logging.INFO)

    def add_placeholders(self):
        self.label_embedding_train = tf.compat.v1.constant(value=self.config.label_emb_train,
                                                           dtype=tf.float32, name="label_embedding_train")
        self.label_embedding_test = tf.compat.v1.constant(value=self.config.label_emb_test,
                                                          dtype=tf.float32, name="label_embedding_test")

    def get_estimator(self):
        run_config = tf.estimator.RunConfig(
            model_dir=self.config.dir_model,
            save_summary_steps=self.config.SAVE_SUMMARY_STEPS,
            keep_checkpoint_max=1,
            save_checkpoints_steps=self.config.SAVE_CHECKPOINTS_STEPS)
        model_fn = model_fn_builder(
            bert_config=self.bert_config,
            model_config= self.config,
            init_checkpoint=self.config.BERT_CHECKPOINT,
            learning_rate=self.config.lr,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_one_hot_embeddings=False)
        return tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            params={"batch_size": self.config.batch_size})

    def get_data_processing(self):
        file_based_convert_examples_to_features(self.train_examples, self.config.max_seq_len,
                                                self.tokenizer,
                                                self.train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(self.train_examples))
        tf.logging.info("  Batch size = %d", self.config.batch_size)
        tf.logging.info("  Num steps = %d", self.num_train_steps)

    def build(self):
        self.get_data_processing()

    def train(self):
        train_input_fn = file_based_input_fn_builder(input_file=self.train_file, seq_length=self.config.max_seq_len,
                                                     is_training=True,
                                                     label_length=self.config.label_len_train,
                                                     drop_remainder=True)
        print(f'Beginning Training!')
        current_time = datetime.now()
        estimator = self.get_estimator()
        estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)
        print("Training took time ", datetime.now() - current_time)

    def eval(self):
        eval_examples = self.processor.get_dev_examples(self.config.tsv_test)
        eval_file = os.path.join(self.config.dir_checkpoint, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, self.max_seq_length, self.tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(eval_examples))
        tf.logging.info("  Batch size = %d", self.batch_size)

        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            is_training=False,
            label_length=self.config.label_len_test,
            drop_remainder=False)

        estimator = self.get_estimator()
        result = estimator.evaluate(input_fn=eval_input_fn, steps=None)

        output_eval_file = os.path.join(self.config.dir_checkpoint, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def test(self):
        test_examples = self.processor.get_dev_examples(self.config.tsv_test)
        test_file = os.path.join(self.config.dir_checkpoint, "test.tf_record")
        file_based_convert_examples_to_features(
            test_examples, self.max_seq_length, self.tokenizer, test_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d", len(test_examples))
        tf.logging.info("  Batch size = %d", self.batch_size)

        predict_input_fn = file_based_input_fn_builder(
            input_file=test_file,
            seq_length=self.max_seq_length,
            is_training=False,
            label_length=self.config.label_len_test,
            drop_remainder=False)

        estimator = self.get_estimator()
        result = estimator.predict(input_fn=predict_input_fn)
        print(result)
        output_eval_file = os.path.join(self.config.dir_checkpoint, "test_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                labels = prediction["pred_label"]
                if i >= len(test_examples):
                    break
                # print(probabilities)
                print(labels)
        #         output_line = "\t".join(
        #             str(class_probability)
        #             for class_probability in probabilities) + "\n"
        #         writer.write(output_line)
        #         num_written_lines += 1
        # assert num_written_lines == len(test_examples)
