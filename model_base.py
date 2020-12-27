import os
import tensorflow as tf
from evaluation import load_raw_labels, evaluate


class BaseModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.sess   = None
        self.saver  = None


    def reinitialize_weights(self, scope_name):
        """Reinitializes the weights of a given layer"""
        variables = tf.contrib.framework.get_variables(scope_name)
        init = tf.variables_initializer(variables)
        self.sess.run(init)


    def add_train_op(self, lr_method, lr, loss, clip=-1):
        """Defines self.train_op that performs an update on a batch

        Args:
            lr_method: (string) sgd method, for example "adam"
            lr: (tf.placeholder) tf.float32, learning rate
            loss: (tensor) tf.float32 loss to minimize
            clip: (python float) clipping of gradient. If < 0, no clipping

        """
        _lr_m = lr_method.lower() # lower to make sure

        with tf.compat.v1.variable_scope("train_step"):
            if _lr_m == 'adam': # sgd method
                optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            elif _lr_m == 'adagrad':
                optimizer = tf.compat.v1.train.AdagradOptimizer(lr)
            elif _lr_m == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(lr)
            elif _lr_m == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(lr)
            else:
                raise NotImplementedError("Unknown method {}".format(_lr_m))

            if clip > 0: # gradient clipping if clip is positive
                grads, vs     = zip(*optimizer.compute_gradients(loss))
                grads, gnorm  = tf.clip_by_global_norm(grads, clip)
                self.train_op = optimizer.apply_gradients(zip(grads, vs))
            else:
                self.train_op = optimizer.minimize(loss)


    def initialize_session(self):
        """Defines self.sess and initialize the variables"""
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        self.logger.info("Initializing tf session")
        self.sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)


    def restore_session(self, dir_model):
        """Reload weights into session

        Args:
            sess: tf.Session()
            dir_model: dir with weights

        """
        self.logger.info("Reloading the latest trained model...")
        self.saver.restore(self.sess, dir_model)


    def save_session(self):
        """Saves session = weights"""
        if not os.path.exists(self.config.dir_model):
            os.makedirs(self.config.dir_model)
        self.saver.save(self.sess, self.config.dir_model)


    def close_session(self):
        """Closes the session"""
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()


    def add_summary(self):
        """Defines variables for Tensorboard

        Args:
            dir_output: (string) where the results are written

        """
        self.merged      = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.dir_output,
                self.sess.graph)


    def train(self):
        """Performs training with early stopping and lr exponential decay

        Args:
            train: dataset that yields tuple of (sentences, tags)

        """
        lowest_loss = 0
        nepoch_no_imprv = 0 # for early stopping
        self.add_summary()  # tensorboard

        for epoch in range(self.config.nepochs):
            self.logger.info("Epoch {:} out of {:}".format(epoch + 1,
                        self.config.nepochs))

            score = self.run_epoch(epoch)
            # # early stopping and saving best parameters
            # if score <= lowest_loss:
            #     nepoch_no_imprv = 0
            #     self.save_session()
            #     lowest_loss = score
            #     self.logger.info("- new lowest loss!")
            # else:
            #     nepoch_no_imprv += 1
            #     if nepoch_no_imprv >= self.config.nepoch_no_imprv:
            #         self.logger.info("- early stopping {} epochs without "\
            #                 "improvement".format(nepoch_no_imprv))
            #         break
            # self.evaluate_all()
        self.save_session()

    def variable_summaries(self, var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            #stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            #tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

