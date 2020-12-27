import numpy as np
from collections import Counter
import tqdm

# shared global variables to be imported from model also
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"


def get_word_vocabs(mention_list):
    """Build vocabulary from an iterable of datasets objects
    Args:
        datasets: a list of dataset objects
    Returns:
        a set of all the words in the dataset
    """
    print("Building word vocab...")
    vocab_words = []
    for men in mention_list:
        for words in men.split(" "):
            # print(words)
            vocab_words.append(words.lower())
    vocab_words = set(vocab_words)
    print("- done. {} tokens".format(len(vocab_words)))
    return vocab_words


def get_char_vocab(mention_list):
    """Build char vocabulary from an iterable of datasets objects
    Args:
        dataset: a iterator yielding tuples (sentence, tags)
    Returns:
        a set of all the characters in the dataset
    """
    print("Building character vocab...")
    vocab_char = set()
    for men in mention_list:
        for char in men:
            # print(char)
            vocab_char.update(char)
    print("- done. {} characters".format(len(vocab_char)))
    return vocab_char


def write_vocab(vocab, filename):
    """
    Writes a vocab to a file
    Writes one word per line.
    Args:
        vocab: iterable that yields word
        filename: path to vocab file
    Returns:
        write a word per line
    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} records".format(len(vocab)))


def load_vocab(filename):
    """Loads vocab from a file
    Args:
        filename: (string) the format of the file must be one word per line.
    Returns:
        d: dict[word] = index
    """
    d = dict()
    with open(filename) as f:
        for idx, word in enumerate(f):
            word = word.strip()
            d[word] = idx
    return d


def export_trimmed_glove_vectors(vocab, glove_filename, trimmed_filename, dim):
    """
    Saves glove vectors in numpy array
    Args:
        vocab: dictionary vocab[word] = index
        glove_filename: a path to a glove file
        trimmed_filename: a path where to store a matrix in npy
        dim: (int) dimension of embeddings
    """
    embeddings = np.zeros([len(vocab), dim])
    with open(glove_filename) as f:
        for line in f:
            line = line.strip().split(' ')
            word = line[0]
            embedding = [float(x) for x in line[1:]]
            if word in vocab:
                word_idx = vocab[word]
                embeddings[word_idx] = np.asarray(embedding)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)


def get_trimmed_glove_vectors(filename):
    """
    Args:
        filename: path to the npz file

    Returns:
        matrix of embeddings (np array)

    """
    with np.load(filename) as data:
        return data["embeddings"]


def get_processing_word(vocab_words=None, vocab_chars=None,
                    lowercase=False, allow_unk=True):
    """Return lambda function that transform a word (string) into list,
    or tuple of (list, id) of int corresponding to the ids of the word and
    its corresponding characters.

    Args:
        vocab: dict[word] = idx

    Returns:
        f("cat") = ([12, 4, 32], 12345)
                 = (list of char ids, word id)

    """
    def f(word):
        # 0. get chars of words
        if vocab_chars is not None:
            char_ids = []
            for char in word:
                # ignore chars out of vocabulary
                if char in vocab_chars:
                    char_ids += [vocab_chars[char]]

        # 1. preprocess word
        if lowercase:
            word = word.lower()
        if word.isdigit():
            word = NUM

        # 2. get id of word
        if vocab_words is not None:
            if word in vocab_words:
                word = vocab_words[word]
            else:
                if allow_unk:
                    word = vocab_words[UNK]
                else:
                    raise Exception("Unknow key is not allowed. Check that "\
                                    "your vocab (tags?) is correct")

        # 3. return tuple char ids, word id
        if vocab_chars is not None:
            return char_ids, word
        else:
            return word

    return f


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with

    Returns:
        a list of list where each sublist has same length
    """
    sequence_padded, sequence_length = [], []
    for seq in sequences:
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1, timestep=10):
    """
    Args:
        sequences: a generator of list or tuple
        pad_tok: the char to pad with
        nlevels: "depth" of padding, for the case where we have characters ids

    Returns:
        a list of list where each sublist has same length

    """
    if nlevels == 1:
        max_length = max(map(lambda x: len(x), sequences))
        sequence_padded, sequence_length = _pad_sequences(sequences,
                                                          pad_tok, max_length)


    elif nlevels == 2:
        max_length_word = max([max(map(lambda x: len(x), seq))
                               for seq in sequences])
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded,
                                            [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0,
                                            max_length_sentence)

    return sequence_padded, sequence_length


def get_batch_word_char_ids(batch_mention, processing_word, is_decode=True):
    # encode the byte string which decoded from tfrecord for each mention
    if is_decode:
        men = [str(x, encoding="utf-8") for x in batch_mention]
    else:
        men = batch_mention

    char_ids = []
    word_ids = []
    for item in men:
        item_processed = list(map(
            lambda x: processing_word(x),
            [x for x in item.split(" ")]))
        word_ids.append([x[1] for x in item_processed])
        char_ids.append([x[0] for x in item_processed])
    return char_ids, word_ids