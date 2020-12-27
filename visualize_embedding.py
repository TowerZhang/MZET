import sys, os

import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from label_bert_embedding import load_bert_embedding
# emb_length = 0
# logs_path = './graph/visual/'
# NAME_TO_VISUALISE_VARIABLE = "label_embedding"
#
# path_for_mnist_sprites =  os.path.join(logs_path,'mnistdigits.png')
# path_for_mnist_metadata =  os.path.join(logs_path,'metadata.tsv')
#
# embedding_var = tf.Variable(emb_length, name=NAME_TO_VISUALISE_VARIABLE)
# summary_writer = tf.summary.FileWriter(logs_path)
#
# config = projector.ProjectorConfig()
# embedding = config.embeddings.add()
# embedding.tensor_name = embedding_var.name
#
# # Specify where you find the metadata
# embedding.metadata_path = path_for_mnist_metadata
#
# # Specify where you find the sprite (we will create this later)
# embedding.sprite.image_path = path_for_mnist_sprites #'mnistdigits.png'
# embedding.sprite.single_image_dim.extend([28,28])
#
# # Say that you want to visualise the embeddings
# projector.visualize_embeddings(summary_writer, config)
#
#
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# saver.save(sess, os.path.join(logs_path, "model.ckpt"), 1)

tsv_file_path = "./graph/visual/metadata.tsv"
glove_vect_file_path = "./graph/visual/vectdata_glove.tsv"
bert_vect_file_path = "./graph/visual/vectdata_bert.tsv"

data = load_bert_embedding("Data/BBN/intermediate/label_glove_train.json")
with open(glove_vect_file_path, 'w') as file_vectdata:
    for word in data.keys():
        temp = "\t".join([str(x) for x in data[word]])
        file_vectdata.write(temp+'\n')

data = load_bert_embedding("Data/BBN/intermediate/label_bert_train.json")
with open(bert_vect_file_path, 'w') as file_vectdata:
    for word in data.keys():
        temp = "\t".join([str(x) for x in data[word][0]])
        file_vectdata.write(temp+'\n')

with open(tsv_file_path, "w") as file_metadata:
    for word in data.keys():
        file_metadata.write(word+'\n')
