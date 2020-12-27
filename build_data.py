from config_train import Config
from label_extract import extract, extract_label_type, write_label, supertype
from label_embedding import hierarchy_embedding, glove_embedding, bert_embedding, \
                            prototype_embedding, get_label_embedding
from mention_embedding import get_mention_list, mention_glove_context_embedding, \
                              mention_bert_context_embedding, mention_embedding, \
                              sentence_embedding_generator, get_label_mention_dataset, \
                              save_tsv_file, save_sentence
from word_character_embedding import get_word_vocabs, get_char_vocab,\
                            UNK, NUM, write_vocab, load_vocab, export_trimmed_glove_vectors
from general_utils import read_type_id, get_glove_vec
from tfrecords_data_io import encode_to_tfrecords, encode_to_tfrecords2, encode_to_tfrecords3
import numpy as np
import math

config = Config()
logger = config.logger


def load_raw_data():
    """ load raw data """
    train, test = config.dataloader.load_dataset()
    logger.info("finish loading raw data.")
    return train, test


def load_glove_data():
    """ load glove embedding vector with embedding size 300"""
    glove_emb_vec = get_glove_vec(config.glove_file)
    logger.info("finish loading glove embedding dictionary")
    return glove_emb_vec


def get_label2id():
    """ get the label2id dictionary for look up into any label"""
    train_label2id = read_type_id(config.typefile_zsl_train)
    test_label2id = read_type_id(config.typefile_common)
    level1_label2id = read_type_id(config.typefile_level1)
    level2_label2id = read_type_id(config.typefile_level2)
    return train_label2id, test_label2id, level1_label2id, level2_label2id


def label_statistics(train, test):
    """ label extraction and related statistics """
    label_train, label_test = extract(train["mentions"]), extract(test["mentions"])
    label_level_train = extract_label_type(set([label_train[x] for x in label_train.keys()]))
    label_level_test = extract_label_type(set([label_test[x] for x in label_test.keys()]))
    coarse_common = set(label_level_train[0]) & set(label_level_test[0])
    fine_common = set(label_level_train[1]) & set(label_level_test[1])
    difference = (set(label_level_train[0]) | set(label_level_test[0]) | set(label_level_train[1]) |
                  set(label_level_test[1])) - set(coarse_common | fine_common)

    # write files for type and hierarchical relationship
    write_label(config.typefile_train, label_level_train)                   # raw train data label
    write_label(config.typefile_zsl_train, [set(label_level_train[0])])     # level-1 label for ZSL
    write_label(config.typefile_test, label_level_test)                     # raw test data label
    write_label(config.typefile_common, [coarse_common, fine_common])       # common label for ZSL test
    write_label(config.typefile_level1, [coarse_common])
    write_label(config.typefile_level2, [fine_common])
    supertype(config.typefile_zsl_train, config.supertypefile_train)
    supertype(config.typefile_test, config.supertypefile_test)
    supertype(config.typefile_common, config.supertypefile_common)

    is_show_statistic = True
    if is_show_statistic:
        print('train data size: ', train.size)
        print('test data size : ', test.size)
        print()
        print("train type:      ", len(label_train), label_train)
        print("level-1 count :  ", len(label_level_train[0]), ",",
              "level-2 count :  ", len(label_level_train[1]))
        print("level-1 :        ", label_level_train[0])
        print("level-2 :        ", label_level_train[1])
        print()
        print("test type:       ", len(label_test), label_test)
        print("level-1 count :  ", len(label_level_test[0]), ",",
              "level-2 count :  ", len(label_level_test[1]))
        print("level-1 :        ", label_level_test[0])
        print("level-2 :        ", label_level_test[1])
        print()
        print("coarse common:   ", len(coarse_common), coarse_common)
        print("fine common:     ", len(fine_common), fine_common)
        print("difference set : ", len(difference), difference)


def label_embedding_data(glove_emb_vec):
    """ make label embedding """
    hierarchy_embedding(config.typefile_zsl_train,
                        config.supertypefile_train, config.hierarchy_train)
    hierarchy_embedding(config.typefile_common,
                        config.supertypefile_common, config.hierarchy_test)

    glove_embedding(config.typefile_zsl_train,
                    config.glove_label_train, glove_emb_vec)
    glove_embedding(config.typefile_common,
                    config.glove_label_test, glove_emb_vec)
    glove_embedding(config.typefile_level1, config.glove_label_test_lv1, glove_emb_vec)
    glove_embedding(config.typefile_level2, config.glove_label_test_lv2, glove_emb_vec)
    emb = get_label_embedding(config.glove_label_test_lv1)
    print(len(emb))
    emb = get_label_embedding(config.glove_label_test_lv2)
    print(len(emb))

    bert_embedding(config.typefile_zsl_train, config.bert_label_train)
    bert_embedding(config.typefile_common, config.bert_label_test)
    label_embedding = get_label_embedding(config.bert_label_test)
    print(np.array(label_embedding).shape)
    bert_embedding(config.typefile_level1, config.bert_label_test_lv1)
    bert_embedding(config.typefile_level2, config.bert_label_test_lv2)
    emb = get_label_embedding(config.bert_label_test_lv1)
    print(len(emb))
    emb = get_label_embedding(config.bert_label_test_lv2)
    print(len(emb))

    prototype_embedding(config.prototype, glove_emb_vec,
                        config.typefile_zsl_train, config.proto_label_train)
    prototype_embedding(config.prototype, glove_emb_vec,
                        config.typefile_common, config.proto_label_test)
    get_label_embedding(config.proto_label_test)
    prototype_embedding(config.prototype, glove_emb_vec,
                        config.typefile_level1, config.proto_label_test_lv1)
    prototype_embedding(config.prototype, glove_emb_vec,
                        config.typefile_level2, config.proto_label_test_lv2)
    emb = get_label_embedding(config.proto_label_test_lv1)
    print(len(emb))
    emb = get_label_embedding(config.proto_label_test_lv2)
    print(len(emb))


def word_character_statistics(para, glove_emb_vec):
    """ extract all words in the mention, and statistic on those words and their characters."""
    mention_list_train = get_mention_list(para["train"], para["train_label2id"], is_train=True)
    mention_list_test = get_mention_list(para["test"], para["test_label2id"], is_train=False)
    mention_list_all = mention_list_train + mention_list_test
    data_word_vocab = get_word_vocabs(mention_list_all)
    glove_word_vocab = set(glove_emb_vec.keys())
    word_vocab = data_word_vocab & glove_word_vocab
    print("{} words cannot be found in Glove. ".format(len(data_word_vocab) -
                                                       len(word_vocab)))
    word_vocab.add(UNK)
    word_vocab.add(NUM)
    char_vocab = get_char_vocab(mention_list_all)
    write_vocab(word_vocab, config.word_vocab_file)
    write_vocab(char_vocab, config.char_vocab_file)

    word_vocab = load_vocab(config.word_vocab_file)
    export_trimmed_glove_vectors(word_vocab, config.glove_file,
                                 config.glove_trimmed, config.dim_word)
    logger.info("finish trimming glove embedding.")


def save_sentence_json(para):
    """save the raw sentence into a json file for bert embedding."""
    train = para["train"]
    save_sentence(train, config.sentence_train)

    test = para["test"]
    save_sentence(test, config.sentence_test)


def para_asignments(para, type, model, is_train=True):
    """ assign the parameters for each different mention embedding methods"""
    model_tf = {
        "NET":[config.tfrecord_train, config.tfrecord_test, config.tfrecord_level1, config.tfrecord_level2],
        "DZET":[config.tfrecord_dzet_train, config.tfrecord_dzet_test, config.tfrecord_dzet_level1, config.tfrecord_dzet_level2],
        "Ctxt":[config.tfrecord_ctxt_train, config.tfrecord_ctxt_test, config.tfrecord_ctxt_level1, config.tfrecord_ctxt_level2]
    }
    tf_file = model_tf[model]
    if is_train:
        data = para["train"]
    else:
        data = para["test"]
    if type == "train":
        label2id = para["train_label2id"]
        tfrecord_file = tf_file[0]
    elif type == "test":
        label2id = para["test_label2id"]
        tfrecord_file = tf_file[1]
    elif type == "level1":
        label2id = para["level1_label2id"]
        tfrecord_file = tf_file[2]
    elif type == "level2":
        label2id = para["level2_label2id"]
        tfrecord_file = tf_file[3]
    return data, label2id, tfrecord_file


def NET_mention_embedding_in_tfrecord(para, type, model, is_train=True):
    """ write into the tfrecord for training, testing, level1, and level2 data by the function 'mention_embedding' """
    data, label2id, tfrecord_file = para_asignments(para, type, model, is_train=is_train)
    if is_train:
        sentence_file = config.sentence_emb_train
    else:
        sentence_file = config.sentence_emb_test

    sentence_generator = sentence_embedding_generator(sentence_file)
    mention_data = mention_embedding(
        data, label2id, sentence_generator, 0, len(data),
        config.full_tokenizer, config.timesteps, config.bert_emb_len, is_train=is_train)
    mention, embedding, label, length = mention_data
    encode_to_tfrecords(mention, embedding, label, length, tfrecord_file)
    return len(mention)


def NET_tfrecord_data(para):
    """ tfrecord data preparation for NET model when using mention bert embedding. """

    train_mention_length = NET_mention_embedding_in_tfrecord(para, type="train", model= "NET", is_train=True)
    logger.info("training tfrecords data convert finishes")

    test_mention_length = NET_mention_embedding_in_tfrecord(para, type="test", model= "NET", is_train=False)
    logger.info("testing tfrecords data convert finishes")
    print("mention count : (train, ", train_mention_length, "),  (test, ", test_mention_length, ")")

    level1_mention_length = NET_mention_embedding_in_tfrecord(para, type="level1", model= "NET", is_train=False)
    logger.info("level1 tfrecords data convert finishes")
    print(level1_mention_length)

    level2_mention_length = NET_mention_embedding_in_tfrecord(para, type="level2", model= "NET", is_train=False)
    logger.info("level2 tfrecords data convert finishes")
    print(level2_mention_length)


def DZET_mention_embedding_in_tfrecord(para, glove_emb_vec, type, model, is_train=True):
    """ write into the tfrecord for training, testing, level1, and level2 data by the function 'mention_embedding' """
    data, label2id, tfrecord_file = para_asignments(para, type, model, is_train=is_train)

    mention_context_data = mention_glove_context_embedding(
        data, label2id, 0, len(data), glove_emb_vec, config.basic_tokenizer, config.window_size, is_train=is_train)
    embedding, left_ctxt, right_ctxt, label= mention_context_data
    encode_to_tfrecords2(embedding, left_ctxt, right_ctxt, label, tfrecord_file)
    return len(embedding)


def DZET_tfrecord_data(para, glove_emb_vec):
    """ tfrecord data preparation for DZET model when using mention contextual glove embedding. """
    # train_mention_length = DZET_mention_embedding_in_tfrecord(para, glove_emb_vec, type="train", model= "DZET", is_train=True)
    # logger.info("training tfrecords data convert finishes")

    # test_mention_length = DZET_mention_embedding_in_tfrecord(para, glove_emb_vec, type="test", model= "DZET", is_train=False)
    # print("mention count : (train, ", train_mention_length, "),  (test, ", test_mention_length, ")")

    level1_mention_length = DZET_mention_embedding_in_tfrecord(para, glove_emb_vec, type="level1", model= "DZET", is_train=False)
    logger.info("level1 tfrecords data convert finishes")
    print(level1_mention_length)

    level2_mention_length = DZET_mention_embedding_in_tfrecord(para, glove_emb_vec, type="level2", model= "DZET", is_train=False)
    logger.info("level2 tfrecords data convert finishes")
    print(level2_mention_length)


def Ctxt_mention_embedding_in_tfrecord(para, type, model, is_train=True):
    """ write into the tfrecord for training, testing, level1, and level2 data by the function 'mention_embedding' """
    data, label2id, tfrecord_file = para_asignments(para, type, model, is_train=is_train)
    if is_train:
        sentence_file = config.sentence_emb_train
    else:
        sentence_file = config.sentence_emb_test


    sentence_generator = sentence_embedding_generator(sentence_file)
    mention_data = mention_bert_context_embedding(
        data, label2id, sentence_generator, 0, len(data), config.full_tokenizer,
        config.bert_emb_len, config.timesteps, config.window_size, is_train=is_train)
    mention, left_ctxt, right_ctxt, label, embedding, length = mention_data
    encode_to_tfrecords3(mention, embedding, left_ctxt, right_ctxt, label, length, tfrecord_file)
    return len(mention)


def Ctxt_NET_tfrecord_data(para):
    """ tfrecord data preparation for NET_context model when using mention contextual bert embedding. """
    train_mention_length = Ctxt_mention_embedding_in_tfrecord(para, type="train", model="Ctxt", is_train=True)
    logger.info("training tfrecords data convert finishes")

    test_mention_length = Ctxt_mention_embedding_in_tfrecord(para, type="test", model="Ctxt", is_train=False)
    print("mention count : (train, ", train_mention_length, "),  (test, ", test_mention_length, ")")

    level1_mention_length = Ctxt_mention_embedding_in_tfrecord(para, type="level1", model="Ctxt", is_train=False)
    logger.info("level1 tfrecords data convert finishes")
    print(level1_mention_length)

    level2_mention_length = Ctxt_mention_embedding_in_tfrecord(para, type="level2", model="Ctxt", is_train=False)
    logger.info("level2 tfrecords data convert finishes")
    print(level2_mention_length)


def get_tsv_data(data, label2id, tsv_file, start, end, is_train=True):
    tsv = get_label_mention_dataset(data, label2id, start, end, is_train)
    label_list, mention_list, context_list = tsv
    save_tsv_file(tsv_file, label_list, mention_list, context_list)


def bert_fine_tuning_data(para):
    data = para["train"]
    label2id = para["train_label2id"]
    ratio = 0.9
    split = math.floor(len(data)*ratio)
    get_tsv_data(data, label2id, config.tsv_train, 0, len(data), is_train=True)

    test = para["test"]
    get_tsv_data(test, para["test_label2id"], config.tsv_test, 0, len(test), is_train=False)


def get_sentence_max_length(data):
    max_len = 1
    for i in range(len(data)):
        tokens = data['tokens'][i]
        if len(tokens) > max_len:
            max_len = len(tokens)
    return max_len


def main():
    """
    This function has to be processed in the first step overall to make a bunch of
    training and testing related files and some statistics release.
    """
    train, test = load_raw_data()
    glove_emb_vec = load_glove_data()
    # label_statistics(train, test)

    train_label2id, test_label2id, level1_label2id, level2_label2id = get_label2id()
    para_data = {
        "train" : train,
        "test"  : test,
        "train_label2id" : train_label2id,
        "test_label2id"  : test_label2id,
        "level1_label2id": level1_label2id,
        "level2_label2id": level2_label2id
    }
    
    word_character_statistics(para_data, glove_emb_vec)
    save_sentence_json(para_data)
    label_embedding_data(glove_emb_vec)
    # NET_tfrecord_data(para_data)
    # DZET_tfrecord_data(para_data, glove_emb_vec)
    Ctxt_NET_tfrecord_data(para_data)
    # bert_fine_tuning_data(para_data)



if __name__ == '__main__':

    main()
