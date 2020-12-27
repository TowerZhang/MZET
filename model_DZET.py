import math
import warnings
warnings.filterwarnings("ignore")

import random
import tensorflow as tf
from word_character_embedding import get_batch_word_char_ids, pad_sequences
from general_utils import Progbar
from model_base import BaseModel
from evaluation import load_raw_labels, evaluate
import numpy as np

random.seed(1)

class DZETModel(BaseModel):
    def __init__(self, config):
        super(DZETModel, self).__init__(config)

    def add_placeholders(self):
        """Define placeholders = entries to computational graph"""

        self.mention_embedding = tf.placeholder("float", [None,
                                                self.config.glove_emb_len], name='mention_embedding_input')

        self.left_context_emb = tf.placeholder("float", [None, self.config.window_size,
                                                self.config.glove_emb_len], name='mention_left_context_input')

        self.right_context_emb = tf.placeholder("float", [None, self.config.window_size,
                                                self.config.glove_emb_len], name='mention_right_context_input')

        self.targets = tf.compat.v1.placeholder(dtype=tf.float32,
                                                shape=[None, self.config.label_len_train], name="targets")

        self.targets_test = tf.compat.v1.placeholder(dtype=tf.float32,
                                                     shape=[None, self.config.label_len_test], name="targets_test")

        self.targets_leve1 = tf.compat.v1.placeholder(dtype=tf.float32,
                                                      shape=[None, self.config.label_len_level1], name="target_level1")

        self.targets_leve2 = tf.compat.v1.placeholder(dtype=tf.float32,
                                                      shape=[None, self.config.label_len_level2], name="target_level2")

        # label embedding matrix for training and testing data
        self.label_embedding_train = tf.compat.v1.constant(value=self.config.label_emb_train,
                                                           dtype=tf.float32, name="label_embedding_train")
        self.label_embedding_test = tf.compat.v1.constant(value=self.config.label_emb_test,
                                                          dtype=tf.float32, name="label_embedding_test")
        self.label_embedding_level1 = tf.compat.v1.constant(value=self.config.label_emb_level1,
                                                            dtype=tf.float32, name="label_embedding_level1")
        self.label_embedding_level2 = tf.compat.v1.constant(value=self.config.label_emb_level2,
                                                            dtype=tf.float32, name="label_embedding_level2")

        # hyper parameters
        self.dropout = tf.compat.v1.placeholder(dtype=tf.float32, shape=[],
                                                name="dropout")
        self.lr = tf.compat.v1.placeholder(dtype=tf.float32, shape=[],
                                           name="lr")

    def get_feed_dict(self, mention_emb, left_context, right_context, label, lr=None,
                      dropout=None, is_train="train"):
        """Given some data, pad it and build a feed dictionary"""

        # build feed dictionary
        feed = {self.left_context_emb: left_context,
                self.right_context_emb: right_context,
                self.mention_embedding: mention_emb}

        if is_train == "train":
            feed[self.targets] = label
        elif is_train == "test":
            feed[self.targets_test] = label
        elif is_train == "level1":
            feed[self.targets_leve1] = label
        elif is_train == "level2":
            feed[self.targets_leve2] = label

        if lr is not None:
            feed[self.lr] = lr

        if dropout is not None:
            feed[self.dropout] = dropout

        return feed

    def add_left_context_op(self):
        with tf.variable_scope("left_context"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_ctxt)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_ctxt)
            bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.left_context_emb, dtype=tf.float32)
            output_fw, output_bw = bi_outputs
            self.output_left = tf.concat([output_fw, output_bw], axis=-1)

            w_omega = tf.Variable(tf.random_normal([2 * self.config.hidden_size_ctxt,
                                                    self.config.attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([self.config.attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([self.config.attention_size], stddev=0.1))

            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(self.output_left, w_omega, axes=1) + b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, u_omega, axes=1, name='vu_left')
            self.alphas_left = tf.nn.softmax(vu, name='alphas_left')

    def add_right_context_op(self):
        with tf.variable_scope("right_context"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_ctxt)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_ctxt)
            bi_outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, self.right_context_emb, dtype=tf.float32)
            output_fw, output_bw = bi_outputs
            self.output_right = tf.concat([output_fw, output_bw], axis=-1)

            w_omega = tf.Variable(tf.random_normal([2 * self.config.hidden_size_ctxt,
                                                    self.config.attention_size], stddev=0.1))
            b_omega = tf.Variable(tf.random_normal([self.config.attention_size], stddev=0.1))
            u_omega = tf.Variable(tf.random_normal([self.config.attention_size], stddev=0.1))

            with tf.name_scope('v'):
                # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
                #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
                v = tf.tanh(tf.tensordot(self.output_right, w_omega, axes=1) + b_omega)

            # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
            vu = tf.tensordot(v, u_omega, axes=1, name='vu_right')
            self.alphas_right = tf.nn.softmax(vu, name='alphas_right')

    def add_context_op(self):
        with tf.variable_scope("total_context"):
            sum = tf.reduce_sum(self.alphas_left, 1) + tf.reduce_sum(self.alphas_right, 1)
            sum = tf.expand_dims(sum, -1)
            self.alphas_left = tf.divide(self.alphas_left, sum)
            self.alphas_right = tf.divide(self.alphas_right, sum)
            self.output_left = tf.expand_dims(self.output_left * tf.expand_dims(self.alphas_left, -1), 1)
            self.output_right = tf.expand_dims(self.output_right * tf.expand_dims(self.alphas_right, -1), 1)
            self.context = tf.reduce_sum(tf.concat([self.output_left, self.output_right], 1), 1)
            self.context = tf.reduce_sum(self.context, 1)
            self.concat = tf.concat([self.mention_embedding, self.context], axis=-1)
            self.concat = self.mention_embedding

    def add_logits_op(self):
        """Defines self.logits
        """

        with tf.compat.v1.variable_scope("proj"):
            weights = {
                "fc1": tf.get_variable("fc1_w", dtype=tf.float32,
                                shape=[self.config.glove_emb_len,
                                       # + 2 * self.config.hidden_size_ctxt,
                                       self.config.hidden_size_fc]),
                "fc2": tf.get_variable("fc2_w", dtype=tf.float32,
                                shape=[self.config.hidden_size_fc, self.config.n_label_emb])
            }
            biases = {
                "fc1": tf.get_variable("fc1_b", shape=[self.config.hidden_size_fc],
                                dtype=tf.float32, initializer=tf.zeros_initializer()),
                "fc2": tf.get_variable("fc2_b", shape=[self.config.n_label_emb],
                                dtype=tf.float32, initializer=tf.zeros_initializer())
            }
            fc_layer = tf.matmul(self.concat, weights["fc1"]) + biases["fc1"]
            self.feature_layer = tf.matmul(fc_layer, weights["fc2"]) + biases["fc2"]

        with tf.name_scope('Logits'):
            self.logits_train = tf.matmul(self.feature_layer, tf.transpose(self.label_embedding_train, perm=[1, 0]))
            self.logits_test = tf.matmul(self.feature_layer, tf.transpose(self.label_embedding_test, perm=[1, 0]))
            self.logits_level1 = tf.matmul(self.feature_layer, tf.transpose(self.label_embedding_level1, perm=[1, 0]))
            self.logits_level2 = tf.matmul(self.feature_layer, tf.transpose(self.label_embedding_level2, perm=[1, 0]))
            self.outputs_train = tf.nn.softmax(self.logits_train, name="outputs_train")
            self.outputs_test = tf.nn.softmax(self.logits_test, name="outputs_test")
            self.outputs_level1 = tf.nn.softmax(self.logits_level1, name="outputs_level1")
            self.outputs_level2 = tf.nn.softmax(self.logits_level2, name="outputs_level2")

        with tf.name_scope('Label_pred'):
            self.labels_pred_train = tf.cast(tf.argmax(self.outputs_train, axis=1), tf.int32)
            self.labels_pred_test = tf.cast(tf.argmax(self.outputs_test, axis=1), tf.int32)
            self.labels_pred_level1 = tf.cast(tf.argmax(self.outputs_level1, axis=1), tf.int32)
            self.labels_pred_level2 = tf.cast(tf.argmax(self.outputs_level2, axis=1), tf.int32)

    def add_loss_op(self):
        """Defines the loss"""
        pos_label = tf.ones_like(self.targets)
        neg_label = -tf.ones_like(self.targets)
        IND = tf.where(tf.equal(self.targets, 1), x=pos_label, y=neg_label)
        max_margin_loss = tf.maximum(0.0, 0.1 - tf.multiply(IND, self.outputs_train))
        self.loss = tf.reduce_mean(max_margin_loss)

        # for tensorboard
        tf.compat.v1.summary.scalar("loss", self.loss)

    def build(self):
        self.add_placeholders()
        self.add_left_context_op()
        self.add_right_context_op()
        self.add_context_op()
        self.add_logits_op()
        self.add_loss_op()

        self.add_train_op(self.config.lr_method, self.lr, self.loss,
                          self.config.clip)
        self.initialize_session()

    def predict_batch(self, mention_emb, left_context, right_context, label):
        fd = self.get_feed_dict(mention_emb, left_context, right_context, label,
                                                  dropout=1.0)
        labels_pred = self.sess.run(self.labels_pred_train, feed_dict=fd)
        labels_pred = load_raw_labels(self.config.supertypefile_common, labels_pred)
        return labels_pred

    def dev_data(self):
        self.dev = [self.config.embedding_dev, self.config.left_context_dev,
               self.config.right_context_dev, self.config.label_dev]

    def run_epoch(self, epoch):
        # progbar stuff for logging
        batch_size = self.config.batch_size
        nbatches = math.ceil(self.config.train_mention_length/batch_size) // batch_size
        prog = Progbar(target=nbatches)

        self.dev = (self.config.embedding_dev, self.config.left_context_dev,
                    self.config.right_context_dev, self.config.label_dev)

        dev_men, dev_left, dev_right, dev_y = self.sess.run(self.dev)
        dev_men = np.squeeze(dev_men)
        total_loss = 0
        # iterate over dataset
        for i in range(nbatches):
            train = [self.config.embedding_train, self.config.left_context_train,
                     self.config.right_context_train, self.config.label_train]
            batch_men, batch_left, batch_right, batch_y = self.sess.run(train)
            # (batch, 1, glove_emb) to (batch, glove_emb)
            batch_men = np.squeeze(batch_men)
            fd = self.get_feed_dict(batch_men, batch_left, batch_right, batch_y,
                                       self.config.lr, self.config.dropout)

            _, label, train_loss, summary = self.sess.run(
                [self.train_op, self.labels_pred_train, self.loss, self.merged], feed_dict=fd)
            prog.update(i + 1, [("train loss", train_loss)])
            total_loss += train_loss
            # tensorboard
            if i % 10 == 0:
                self.file_writer.add_summary(summary, epoch * nbatches + i)
        pred_dev = self.predict_batch(dev_men, dev_left, dev_right, dev_y)
        strict_acc, mi_f1, ma_f1, _, _, _, _ = evaluate(pred_dev, dev_y)
        msg = "loss: {:.4f}, strict_acc: {:.4f}, mi_f1: {:.4f}, ma_f1: {:.4f}".format(
            total_loss, strict_acc, mi_f1, ma_f1
        )
        print("\nepoch:", epoch, "| dev evaluation:[", msg, "]")
        return total_loss / float(nbatches)

    def evaluate_all(self):
        test = [self.config.embedding_test, self.config.left_context_test,
                self.config.right_context_test, self.config.label_test]
        mention_emb, leftctxt, rightctxt, label = self.sess.run(test)
        mention_emb = np.squeeze(mention_emb)
        fd = self.get_feed_dict(mention_emb, leftctxt, rightctxt, label, dropout=1.0, is_train="test")
        labels_pred = self.sess.run(self.labels_pred_test, feed_dict=fd)
        labels_pred = load_raw_labels(self.config.supertypefile_common, labels_pred)
        strict_acc, mi_f1, ma_f1, mi_p, mi_r, ma_p, ma_r = evaluate(labels_pred, label)
        msg = "Test Evaluation on Overall---\n"
        msg += "Overall---strict_acc: {:.4f}, mi_f1: {:.4f}, ma_f1: {:.4f}, ".format(strict_acc, mi_f1, ma_f1)
        msg += "mi_p: {:.4f}, mi_r: {:.4f}, ma_p: {:.4f}, ma_r: {:.4f}\n".format(mi_p, mi_r, ma_p, ma_r)
        self.logger.info(msg)

    def evaluate_level1(self):
        level1 = [self.embedding_level1, self.left_context_level1,
                  self.right_context_level1, self.label_level1]
        mention_emb1, leftctxt1, rightctxt1, label1 = self.sess.run(level1)
        mention_emb1 = np.np.squeeze(mention_emb1)
        fd1 = self.get_feed_dict(mention_emb1, leftctxt1, rightctxt1, label1, dropout=1.0, is_train="level1")
        labels_pred1 = self.sess.run(self.labels_pred_level1, feed_dict=fd1)
        strict_acc1, mi_f11, ma_f11, mi_p1, mi_r1, ma_p1, ma_r1 = evaluate(labels_pred1, label1)
        msg = "Test Evaluation on level_1---\n"
        msg += "Level_1---acc: {:.4f}, mi_f1: {:.4f}, ma_f1: {:.4f}, ".format(strict_acc1, mi_f11, ma_f11)
        msg += "mi_p: {:.4f}, mi_r: {:.4f}, ma_p: {:.4f}, ma_r: {:.4f}\n".format(mi_p1, mi_r1, ma_p1, ma_r1)
        self.logger.info(msg)

    def evaluate_level2(self):
        level2 = [self.embedding_level2, self.left_context_level2,
                  self.right_context_level2, self.label_level2]
        mention_emb2, leftctxt2, rightctxt2, label2 = self.sess.run(level2)
        mention_emb2 = np.np.squeeze(mention_emb2)
        fd2 = self.get_feed_dict(mention_emb2, leftctxt2, rightctxt2, label2, dropout=1.0, is_train="level2")
        labels_pred2 = self.sess.run(self.labels_pred_level2, feed_dict=fd2)
        strict_acc2, mi_f12, ma_f12, mi_p2, mi_r2, ma_p2, ma_r2 = evaluate(labels_pred2, label2)
        msg = "Test Evaluation on level_2---\n"
        msg += "Level_2---acc: {:.4f}, mi_f1: {:.4f}, ma_f1: {:.4f}, ".format(strict_acc2, mi_f12, ma_f12)
        msg += "mi_p: {:.4f}, mi_r: {:.4f}, ma_p: {:.4f}, ma_r: {:.4f}\n".format(mi_p2, mi_r2, ma_p2, ma_r2)
        self.logger.info(msg)