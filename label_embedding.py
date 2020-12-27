# use bert-as-service to encode label type:
# 1> demo test:
# How to install and use the tool bert-as-service, please see https://github.com/hanxiao/bert-as-service#install
# After installing the tool then use the demo to test whether it works well.
#     >> from bert_serving.client import BertClient
#     >> bc = BertClient()
#     >> results = bc.encode(['First do it', 'then do it right', 'then do it better'])
#     >> print(results, '\n shape: \t', results.shape)
#
# 2> run code in this script:
# First of all, please start the bert-as service to run the code below.
#     >>  bert-serving-start -model_dir bert/uncased_L-12_H-768_A-12/ -num_worker=4

from bert_serving.client import BertClient
import collections
import numpy as np
import json
from general_utils import read_type_id, read_id_type

class TypeHierarchy:
    """
    Provide funtions to find the all child nodes in the hierarchy structure,
    and get the path from parent to a specific child node.
    """
    def __init__(self, file_name):
        self._type_hierarchy = {}                               # {child: parents}
        self._subtype_mapping = collections.defaultdict(list)   # {parents: [child1, child2,...]}
        self._root = set()                                      # root types (on 1-level)

        with open(file_name) as f:
            for line in f:
                t = line.strip('\r\n').split('\t')
                self._type_hierarchy[int(t[0])] = int(t[1])
                self._subtype_mapping[int(t[1])].append(int(t[0]))
                self._root.add(int(t[0]))

    def get_type_path(self, label):
        """
        get the path from parent to the child, and denote by type id:
        example: [6, 12]
        """
        if label in self._type_hierarchy:  # label has super type
            path = [label]
            while label in self._type_hierarchy:
                path.append(self._type_hierarchy[label])
                label = self._type_hierarchy[label]
            path.reverse()
            return path
        else:  # label is the root type
            return [label]

    def get_subtypes(self, label):
        """
        get all children generated by this parent node.
        example : get_subtypes(6)
        return  : [4, 25, 12, 29, 35, 45, 24, 44]
        """
        if label in self._subtype_mapping:
            return self._subtype_mapping[label]
        else:
            return None


def get_type_name(type):
    """
    extract type name only, lowercase them, recover the phrase.
    example:
        "/PLANT"          -> "plant"
        "/CONTACT_INFO"   -> "contact info"
    """
    splite = type.split('/')
    splite = [x.replace('_',' ').lower() for x in splite if len(x)>0]
    return splite


def save_embedding(embedding_dic, indir):
    """ save embedding dictionary into json file"""
    with open(indir, 'w') as f:
        json.dump(embedding_dic, f)


def load_embedding(indir):
    """from json file load the embedding dictionary."""
    with open(indir, 'r') as f:
        return json.load(f)


def make_hier_vec(tier,vocab):
    """
    embedding the hierarchy type in one hot manner, which means:
    each row represents a type, the same as a column. But for a
    single row vector in the embedding matrix, it is encoded by
    1 or 0, when the node id is in the parent-child path, then
    the corresponded column will be set to be 1, otherwise 0.
    """
    vecs = np.zeros([len(vocab), len(vocab)], dtype=np.float32)
    for key in vocab.keys():
        value = vocab[key]
        path = tier.get_type_path(value)
        vecs[value][path] =1.0
    return vecs


def save_hierarchy(mat, dct, fn):
    """
    save the embedding matrix into a file, and the format is:
    matrix shape:   type_len * type_len
    example :
                    [/PLANT 1.0 0.0 0.0 0.0 0.0 0.0 ...]
    """
    type_len, embedding_len = mat.shape
    with open(fn, 'w') as out:
        for i in range(type_len):
            text = " ".join(map(str, mat[i]))
            out.write("%s %s\n" % (dct[i], text))


def hierarchy_embedding(type_dir, supertype_dir, save_dir):
    """
    make hierarchy embeddings for each label and save the embedding
    information into 'label_hier' files.
    """
    tier = TypeHierarchy(supertype_dir)
    label2id = read_type_id(type_dir)
    id2label = read_id_type(type_dir)
    embedding = make_hier_vec(tier, label2id)
    # print(embedding[12])    # path: [6, 12]. only position 6, and 12 are set to be 1, the rest are 0.
    save_hierarchy(embedding, id2label, save_dir)


def make_glove_vec(vocab, glove_emb):
    """
    Using bert to encode the label type. For hierarchy structure label,
    combine the bert embedding for the parent and the child into the final
    hierarchy type embedding vector.

    return: a dictionary about embedding for each type.
    """
    vecs = {}
    for key in vocab.keys():
        value = vocab[key]
        trans_value = get_type_name(value)
        temp = []
        for word in trans_value:
            for single in word.split(" "):
                if single in glove_emb.keys():
                    temp.append(glove_emb[single])
                else:
                    print(single, " was ont found in Glove vocabulary.")
                    temp.append(np.random.rand(300))
        vecs[value] = (np.sum(np.array(temp), axis=0)/float(len(temp))).tolist()
    return vecs


def glove_embedding(type_dir, embed_dir, word_to_vec):
    """
    make glove embeddings for each label and save the embedding
    information into 'label_glove' files.
    """
    id2label = read_id_type(type_dir)
    glove_dic = make_glove_vec(id2label, word_to_vec)
    save_embedding(glove_dic, embed_dir)


def make_bert_vec(vocab):
    """
    Using bert to encode the label type. For hierarchy structure label,
    combine the bert embedding for the parent and the child into the final
    hierarchy type embedding vector.

    return: a dictionary about embedding for each type.
    """
    vecs = {}
    bc = BertClient()
    for id, value in vocab.items():
        trans_value = get_type_name(value)
        vecs[value] = np.sum(bc.encode(trans_value), axis=0)/float(len(trans_value))
        vecs[value] = vecs[value].tolist()
    return vecs


def bert_embedding(type_dir, embed_dir):
    """
    make bert embeddings for each label and save the embedding
    information into 'label_glove' files.
    """
    id2label = read_id_type(type_dir)
    embedding_dic = make_bert_vec(id2label)
    save_embedding(embedding_dic, embed_dir)


def make_prototype_vec(protofile, glove_emb_vec, vocab):
    """
    using glove to embedding the words for each type, then calculate the average
    for each type as this type's proto embedding.
    """
    prototype_dic = {}
    vecs = {}
    with open(protofile, 'r') as f:
        for line in f:
            seg = line.replace("\n", "").split("\t")
            value = seg[1:len(seg)]
            emb_word = []
            for item in value:
                word = item.split("_")
                emb_word += [glove_emb_vec[x] for x in word if x in glove_emb_vec.keys()]
            prototype_dic[seg[0]] = np.sum(np.array(emb_word), axis=0)/float(len(emb_word))
    for id, value in vocab.items():
        vecs[value] = prototype_dic[value].tolist()
    return vecs


def prototype_embedding(protofile, glove_emb_vec, typefile, embed_dir):
    """
    make proto type embedding, and save it in "label_proto" files.
    """
    id2label = read_id_type(typefile)
    embedding_dic = make_prototype_vec(protofile, glove_emb_vec, id2label)
    save_embedding(embedding_dic, embed_dir)


def get_label_embedding(label_emb_dir):
    """
    return label embedding which comes from the file "label_emb_dir"
    """
    label_embedding = load_embedding(label_emb_dir)
    label_matrix = []
    for key in label_embedding.keys():
        label_matrix.append(label_embedding[key])
    return label_matrix
