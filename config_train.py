from data_loader import DataLoader
from label_embedding import get_label_embedding
from mention_embedding import sentence_embedding_generator, mention_embedding, mention_bert_context_embedding
from word_character_embedding import load_vocab, get_processing_word,  get_trimmed_glove_vectors
from general_utils import get_dir, read_type_id, get_logger, Compute_Sim
from tfrecords_data_io import decode_from_tfrecords, decode_from_tfrecords2, decode_from_tfrecords3, \
                              decode_from_bertdata
import mention_tokenization as tokenization
from model_net import NETModel
from model_DZET import DZETModel
from model_CtxtZET import CtxtModel
from model_MZET import MZETModel
from model_MZET_attn import Attn_MZETModel
from model_bert_finetuning import bertZET
from model_CtxtMemZET import CtxtMemZET
import tensorflow as tf
import os


class Config():
    def __init__(self):
        self.nwords = None
        self.nchars = None
        self.word_vocab = None
        self.char_vocab = None
        self.embeddings = None
        self.n_label_emb = None
        self.processing_word = None
        self.train, self.test, self.dev = None, None, None
        self.mention_train, self.embedding_train, self.label_train, self.length_train = None, None, None, None
        self.mention_test, self.embedding_test, self.label_test, self.length_test = None, None, None, None
        self.mention_dev, self.embedding_dev, self.label_dev, self.length_dev = None, None, None, None
        self.test_level1_mention_data, self.test_level2_mention_data = None, None
        self.label_emb_train, self.label_emb_test, self.label_emb_level1, self.label_emb_level2 = None, None, None, None
        self.input_ids, self.input_mask, self.segment_ids, self.label_ids = None, None, None, None
        self.left_context_level1, self.right_context_level1 = None, None
        self.left_context_level2, self.right_context_level2 = None, None
        self.left_context_train, self.right_context_train = None, None
        self.left_context_test, self.right_context_test = None, None
        self.left_context_dev, self.right_context_dev = None, None        

    def load(self, level=False):
        self.word_vocab = load_vocab(self.word_vocab_file)
        self.char_vocab = load_vocab(self.char_vocab_file)
        self.nwords = len(self.word_vocab)
        self.nchars = len(self.char_vocab)
        self.embeddings = get_trimmed_glove_vectors(self.glove_trimmed)

        self.processing_word = get_processing_word(self.word_vocab, self.char_vocab, lowercase=True)

        if self.feature_flag == "Ctxt_NET":
            filename_queue_train = tf.train.string_input_producer([self.tfrecord_ctxt_train], num_epochs=None)
            self.train = decode_from_tfrecords3(
                filename_queue_train, self.window_size, self.bert_emb_len, self.timesteps,
                self.label_len_train, is_batch=True, batch=self.batch_size)
            filename_queue_test = tf.train.string_input_producer([self.tfrecord_ctxt_test], num_epochs=None)
            self.test = decode_from_tfrecords3(
                filename_queue_test, self.window_size, self.bert_emb_len, self.timesteps,
                self.label_len_test, is_batch=True, batch=self.test_mention_length)
            self.dev = decode_from_tfrecords3(
                filename_queue_train, self.window_size, self.bert_emb_len, self.timesteps,
                self.label_len_train, is_batch=True, batch=self.batch_size)

            (self.mention_train, self.embedding_train, self.left_context_train,
             self.right_context_train, self.label_train, self.length_train) = self.train

            (self.mention_test, self.embedding_test, self.left_context_test,
             self.right_context_test, self.label_test, self.length_test) = self.test

            (self.mention_dev, self.embedding_dev, self.left_context_dev,
             self.right_context_dev, self.label_dev, self.length_dev) = self.dev

            if level:
                filename_queue_test = tf.train.string_input_producer([self.tfrecord_ctxt_level1], num_epochs=None)
                self.level1 = decode_from_tfrecords3(
                    filename_queue_test, self.window_size, self.bert_emb_len, self.timesteps,
                    self.label_len_level1, is_batch=True, batch=self.label_len_level1)
                filename_queue_test = tf.train.string_input_producer([self.tfrecord_ctxt_level2], num_epochs=None)
                self.level2 = decode_from_tfrecords3(
                    filename_queue_test, self.window_size, self.bert_emb_len, self.timesteps,
                    self.label_len_level2, is_batch=True, batch=self.label_len_level2)

                (self.mention_level1, self.embedding_level1, self.left_context_level1,
                 self.right_context_level1, self.label_level1, self.length_level1) = self.level1

                (self.mention_level2, self.embedding_level2, self.left_context_level2,
                 self.right_context_level2, self.label_level2, self.length_level2) = self.level2

        elif self.feature_flag == "DZET":
            filename_queue_train = tf.train.string_input_producer([self.tfrecord_dzet_train], num_epochs=None)
            self.train = decode_from_tfrecords2(
                filename_queue_train, self.window_size, self.glove_emb_len,
                self.label_len_train, is_batch=True, batch=self.batch_size)
            filename_queue_test = tf.train.string_input_producer([self.tfrecord_dzet_test], num_epochs=None)
            self.test = decode_from_tfrecords2(
                filename_queue_test, self.window_size, self.glove_emb_len,
                self.label_len_test, is_batch=True, batch=self.test_mention_length)
            self.dev = decode_from_tfrecords2(
                filename_queue_train, self.window_size, self.glove_emb_len,
                self.label_len_train, is_batch=True, batch=self.batch_size)

            (self.embedding_train, self.left_context_train,
             self.right_context_train, self.label_train) = self.train

            (self.embedding_test, self.left_context_test,
             self.right_context_test, self.label_test) = self.test

            (self.embedding_dev, self.left_context_dev,
             self.right_context_dev, self.label_dev) = self.dev

            if level:
                filename_queue_test = tf.train.string_input_producer([self.tfrecord_dzet_level1], num_epochs=None)
                self.level1 = decode_from_tfrecords(
                    filename_queue_test, self.window_size, self.glove_emb_len,
                    self.label_len_level1, is_batch=True, batch=self.label_len_level1)
                filename_queue_test = tf.train.string_input_producer([self.tfrecord_dzet_level2], num_epochs=None)
                self.level2 = decode_from_tfrecords(
                    filename_queue_test, self.window_size, self.glove_emb_len,
                    self.label_len_level2, is_batch=True, batch=self.label_len_level2)

                (self.embedding_level1, self.left_context_level1,
                 self.right_context_level1, self.label_level1) = self.level1

                (self.embedding_level2, self.left_context_level2,
                 self.right_context_level2, self.label_level2) = self.level2

        elif self.feature_flag == "NET":
            filename_queue_train = tf.train.string_input_producer([self.tfrecord_train], num_epochs=None)
            self.train = decode_from_tfrecords(
                filename_queue_train, self.timesteps, self.bert_emb_len,
                self.label_len_train, is_batch=True, batch=self.batch_size)
            filename_queue_test = tf.train.string_input_producer([self.tfrecord_test], num_epochs=None)
            self.test = decode_from_tfrecords(
                filename_queue_test, self.timesteps, self.bert_emb_len,
                self.label_len_test, is_batch=True, batch=self.test_mention_length)
            self.dev = decode_from_tfrecords(
                filename_queue_train, self.timesteps, self.bert_emb_len,
                self.label_len_train, is_batch=True, batch=self.batch_size)

            (self.mention_train, self.embedding_train,
             self.label_train, self.length_train) = self.train

            (self.mention_test, self.embedding_test,
             self.label_test, self.length_test) = self.test

            (self.mention_dev, self.embedding_dev,
             self.label_dev, self.length_dev) = self.dev

            if level:
                filename_queue_test = tf.train.string_input_producer([self.tfrecord_level1], num_epochs=None)
                self.level1 = decode_from_tfrecords(
                    filename_queue_test, self.timesteps, self.bert_emb_len,
                    self.label_len_level1, is_batch=True, batch=self.level1_len)
                filename_queue_test = tf.train.string_input_producer([self.tfrecord_level2], num_epochs=None)
                self.level2 = decode_from_tfrecords(
                    filename_queue_test, self.timesteps, self.bert_emb_len,
                    self.label_len_level2, is_batch=True, batch=self.level2_len)

                (self.mention_level1, self.embedding_level1,
                 self.label_level1, self.length_level1) = self.level1

                (self.mention_level2, self.embedding_level2,
                 self.label_level2, self.length_level2) = self.level2

        elif self.feature_flag == "BertTuning":
            filename_queue_train = tf.train.string_input_producer([self.tfrecore_bert_train], num_epochs=None)
            self.train = decode_from_bertdata(filename_queue_train, 
                            self.max_seq_len, self.label_len_train, 
                            is_batch=True, batch=self.batch_size)
            filename_queue_test = tf.train.string_input_producer([self.tfrecore_bert_test], num_epochs=None)
            self.test = decode_from_bertdata(filename_queue_test, 
                            self.max_seq_len, self.label_len_test, 
                            is_batch=True, batch=self.batch_size)
            
            (self.input_ids_train, self.input_mask_train, 
            self.segment_ids_train, self.label_ids_train) = self.train

            (self.input_ids_test, self.input_mask_test, 
            self.segment_ids_test, self.label_ids_test) = self.test

        # load test data with different level types
        # _, test = self.dataloader.load_dataset()
        # test_level1_label2id = read_type_id(self.typefile_level1)
        # test_level2_label2id = read_type_id(self.typefile_level2)
        # test_sentence_generator1 = sentence_embedding_generator(self.sentence_emb_test)
        # self.test_level1_mention_data = mention_embedding(
        #     test, test_level1_label2id, test_sentence_generator1, 0, len(test),
        #     config.full_tokenizer, config.timesteps, config.bert_emb_len, is_train=False)
        # test_sentence_generator2 = sentence_embedding_generator(config.sentence_emb_test)
        # self.test_level2_mention_data = mention_embedding(
        #     test, test_level2_label2id, test_sentence_generator2, 0, len(test),
        #     config.full_tokenizer, config.timesteps, config.bert_emb_len, is_train=False)

        if self.label_flag == "glove":
            self.n_label_emb = self.glove_emb_len
            self.label_emb_train = get_label_embedding(self.glove_label_train)
            self.label_emb_test = get_label_embedding(self.glove_label_test)
            self.label_emb_level1 = get_label_embedding(self.glove_label_test_lv1)
            self.label_emb_level2 = get_label_embedding(self.glove_label_test_lv2)
        elif self.label_flag == "bert":
            self.n_label_emb = self.bert_emb_len
            self.label_emb_train = get_label_embedding(self.bert_label_train)
            self.label_emb_test = get_label_embedding(self.bert_label_test)
            self.label_emb_level1 = get_label_embedding(self.bert_label_test_lv1)
            self.label_emb_level2 = get_label_embedding(self.bert_label_test_lv2)
        elif self.label_flag == "proto":
            self.n_label_emb = self.glove_emb_len
            self.label_emb_train = get_label_embedding(self.proto_label_train)
            self.label_emb_test = get_label_embedding(self.proto_label_test)
            self.label_emb_level1 = get_label_embedding(self.proto_label_test_lv1)
            self.label_emb_level2 = get_label_embedding(self.proto_label_test_lv2)

    def load_similarity(self):
        self.sim_train = Compute_Sim(self.label_emb_train, self.label_emb_train,
                                self.sim_scale)
        self.sim_test = Compute_Sim(self.label_emb_train, self.label_emb_test,
                                self.sim_scale)
        self.sim_level1 = Compute_Sim(self.label_emb_train, self.label_emb_level1,
                        self.sim_scale)
        self.sim_level2 = Compute_Sim(self.label_emb_train, self.label_emb_level2,
                        self.sim_scale)

    # label_flag = "glove"
    label_flag = "bert"
    # label_flag = "proto"

    # feature_flag = "NET"
    # feature_flag = "DZET"
    feature_flag = "Ctxt_NET"
    # feature_flag = "BertTuning"

    # dataname = "Wiki"
    dataname = "OntoNotes"
    # dataname = "BBN"

    dataloader = DataLoader(dataname)
    # embeddings
    dim_word = 300
    dim_char = 100

    # training
    train_embeddings = False
    nepochs = 3
    dropout = 0.8
    batch_size = 64
    lr_method = "adam"
    lr = 0.001
    lr_decay = 0.9
    clip = -1                 # if negative, no clipping
    nepoch_no_imprv = 3
    sim_scale = 3

    # model hyperparameters
    hidden_size_char = 100    # lstm on chars
    hidden_size_lstm_1 = 300  # lstm on word embeddings
    hidden_size_lstm_2 = 300  # lstm on word embeddings
    hidden_size_lstm = 300
    hidden_size_bert = 300    # lstm on bert embeddings
    hidden_size_ctxt = 150    # lstm on contextual embedding
    hidden_size_fc = 100      # fully connected layer
    attention_size = 100      # attention node size
    memory_embedding_size = 300  # memory size

    # bert
    BERT_MODEL = 'uncased_L-12_H-768_A-12'
    BERT_PRETRAINED_DIR = '../bert/' + BERT_MODEL
    VOCAB_FILE = BERT_PRETRAINED_DIR + '/vocab.txt'
    BERT_CONFIG = BERT_PRETRAINED_DIR + '/bert_config.json'
    BERT_CHECKPOINT = BERT_PRETRAINED_DIR + '/bert_model.ckpt'
    SAVE_CHECKPOINTS_STEPS = 2000
    SAVE_SUMMARY_STEPS = 500
    full_tokenizer = tokenization.FullTokenizer(
        vocab_file=VOCAB_FILE, do_lower_case=True)
    basic_tokenizer = tokenization.BasicTokenizer()
    timesteps = 10
    window_size = 10
    bert_emb_len = 768
    glove_emb_len = 300
    label_len_train = 16    # training label hot code length
    label_len_test = 39     # testing label hot code length
    label_len_level1 = 15   # level1 label length
    label_len_level2 = 24   # level2 label length
    max_seq_len = 100       # bert fine tuning input sequence length

    train_mention_length, test_mention_length, level1_len, level2_len = 86078, 12845, 12845, 8918

    ''' 
    file declaration
    '''
    path_log             = get_dir(dataname, "log.txt")
    typefile_train       = get_dir(dataname, "type_train.txt")
    typefile_zsl_train   = get_dir(dataname, "type_zsl_train.txt")
    typefile_test        = get_dir(dataname, "type_test.txt")
    typefile_common      = get_dir(dataname, "type_common.txt")
    typefile_level1      = get_dir(dataname, "type_level1.txt")
    typefile_level2      = get_dir(dataname, "type_level2.txt")
    supertypefile_train  = get_dir(dataname, "supertype_train.txt")
    supertypefile_test   = get_dir(dataname, "supertype_test.txt")
    supertypefile_common = get_dir(dataname, "supertype_common.txt")
    supertypefile_level2 = get_dir(dataname, "supertype_level2.txt")
    prototype            = get_dir(dataname, "prototypes.csv")
    hierarchy_train      = get_dir(dataname, "label_hier_train.txt")
    hierarchy_test       = get_dir(dataname, "label_hier_test.txt")
    glove_label_train    = get_dir(dataname, "label_glove_train.json")
    glove_label_test     = get_dir(dataname, "label_glove_test.json")
    glove_label_test_lv1 = get_dir(dataname, "label_glove_test_lv1.json")
    glove_label_test_lv2 = get_dir(dataname, "label_glove_test_lv2.json")
    bert_label_train     = get_dir(dataname, "label_bert_train.json")
    bert_label_test      = get_dir(dataname, "label_bert_test.json")
    bert_label_test_lv1  = get_dir(dataname, "label_bert_test_lv1.json")
    bert_label_test_lv2  = get_dir(dataname, "label_bert_test_lv2.json")
    proto_label_train    = get_dir(dataname, "label_proto_train.json")
    proto_label_test     = get_dir(dataname, "label_proto_test.json")
    proto_label_test_lv1 = get_dir(dataname, "label_proto_test_lv1.json")
    proto_label_test_lv2 = get_dir(dataname, "label_proto_test_lv2.json")
    word_vocab_file      = get_dir(dataname, "words_vocab.txt")
    char_vocab_file      = get_dir(dataname, "chars_vocab.txt")
    sentence_train       = get_dir(dataname, "sentence_train.txt")
    sentence_test        = get_dir(dataname, "sentence_test.txt")
    sentence_emb_train   = get_dir(dataname, "sentence_emb_train.json")
    sentence_emb_test    = get_dir(dataname, "sentence_emb_test.json")
    tfrecord_train       = get_dir(dataname, "train.tf.records")
    tfrecord_test        = get_dir(dataname, "test.tf.records")
    tfrecord_level1      = get_dir(dataname, "level1.tf.records")
    tfrecord_level2      = get_dir(dataname, "level2.tf.records")
    tfrecord_ctxt_train  = get_dir(dataname, "train_ctxt.tf.records")
    tfrecord_ctxt_test   = get_dir(dataname, "test_ctxt.tf.records")
    tfrecord_ctxt_level1 = get_dir(dataname, "level1_ctxt.tf.records")
    tfrecord_ctxt_level2 = get_dir(dataname, "level2_ctxt.tf.records")
    tfrecord_dzet_train  = get_dir(dataname, "train_dzet.tf.records")
    tfrecord_dzet_test   = get_dir(dataname, "test_dzet.tf.records")
    tfrecord_dzet_level1 = get_dir(dataname, "level1_dzet.tf.records")
    tfrecord_dzet_level2 = get_dir(dataname, "level2_dzet.tf.records")
    tfrecore_bert_train  = get_dir(dataname, "bert_output/train.tf_record")
    tfrecore_bert_test   = get_dir(dataname, "bert_output/test.tf_record")
    tsv_train            = get_dir(dataname, "train.tsv")
    tsv_test             = get_dir(dataname, "test.tsv")
    tsv_train_sampling   = get_dir(dataname, "train_sample.tsv")
    dir_output           = get_dir(dataname, "result")
    dir_checkpoint       = get_dir(dataname, "Model/")
    dir_bert_date        = get_dir(dataname, "bert_output")
    glove_file = "../../Glove/glove.6B.300d.txt"
    glove_trimmed = "../../Glove/glove.6B.300d.npz"
    dir_model = "./Model/BBN/4/"

    logger = get_logger(path_log)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # model_name = "NET"
    # model_name = "DZET"
    # model_name = "Ctxt_ZET"
    model_name = "MZETModel"
    # model_name = "MZETModel_attn"

    config = Config()
    config.load(level=True)
    config.logger.info("data is ready.......")
    if model_name == "NET":
        model = NETModel(config)
    elif model_name == "DZET":
        model = DZETModel(config)
    elif model_name == "Ctxt_ZET":
        model = CtxtModel(config)
    elif model_name == "MZETModel":
        config.load_similarity()
        model = MZETModel(config)
    elif model_name == "MZETModel_attn":
        config.load_similarity()
        model = Attn_MZETModel(config)
    model.build()
    model.train()
    # model.restore_session(config.dir_model)
    model.evaluate_all()
    model.evaluate_level1()
    model.evaluate_level2()

    # config.load_similarity()
    # model = CtxtMemZET(config)
    # model.build()
    # model.train()
    # model.evaluate_all()



