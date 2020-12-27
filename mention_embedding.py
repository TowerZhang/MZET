import numpy as np
import json




def get_mention_surface(tokens, start, end):
    """
    using the position of the mention to extract the mention string form the tokens.
    example: 'Eddie_Bauer'
    """
    concat = ""
    for i in range(start, end):
        concat += tokens[i] + "_"
    if len(concat) > 0:
        concat = concat[:-1]
    return concat


def get_mention_surface_raw(tokens, start, end):
    """
    using the position of the mention to extract the mention string form the tokens.
    example: 'Eddie Bauer'
    """
    concat = ""
    for i in range(start, end):
        concat += tokens[i] + " "
    if len(concat) > 0:
        concat = concat[:-1]
    return concat


def get_token_string(tokens, mentions):
    """
    get the whole sentence string
    """
    concat_str = ""
    start_end = {}  # store the mention position: {{start1: end1},{start2: end2},{start3: end3},...]}
    for men in mentions:
        start = int(men['start'])
        end = int(men['end'])
        start_end.update({start: end})
    # concatenating the tokens with combined mentions, i.e. multi words in a mention will be connected by underline.
    i = 0
    while i < len(tokens):
        if i in start_end.keys():   # start
            start = i
            end = start_end[i]
            concat_str += get_mention_surface_raw(tokens, start, end)
            i = end - 1             # end
        else:
            concat_str += tokens[i]
        i += 1
        concat_str += " "
    if len(concat_str) > 0:
        concat_str = concat_str[:-1]
    return concat_str


def index_mention_in_token_str(tokens, mentions, label2id, tokenizer):
    """
    Do bert tokenization on each concatenating sentence, and trace the position
    of each mention after tokenization.
    Return the position dictionary {start: (end, label_onehot_code)} for each mention.
    """
    concat_str = ""
    start_end = {}  # store the mention position: {{start1: (index1, end1)},{start2: (index2, end2)},...]}
    for idx in range(len(mentions)):
        start = int(mentions[idx]['start'])
        end = int(mentions[idx]['end'])
        start_end.update({start: (idx, end)})   # 👈index is for tracking the mention["labels"]
    i = 0

    # store the index for each mention in the tokenized sentence, {start: (end, label_onehot_code)}
    start_end_token_men = {}
    while i < len(tokens):
        if i in start_end.keys():   # start
            start = i
            end = start_end[i][1]
            mention_str = get_mention_surface_raw(tokens, start, end)
            mention_idx = start_end[i][0]
            men_label_code = mention_label_hot_encode(mentions[mention_idx]["labels"], label2id)
            # print(tokenizer.tokenize(mention_str), mentions[mention_idx]["labels"])

            temp_start = len(tokenizer.tokenize(concat_str))
            temp_end = len(tokenizer.tokenize(mention_str))

            # temp_start+1, considering the [CLS] mark.
            start_end_token_men.update({temp_start+1: (temp_start + temp_end+1, men_label_code)})
            concat_str += mention_str
            i = end - 1             # end
        else:
            concat_str += tokens[i]
        i += 1
        concat_str += " "
    if len(concat_str) > 0:
        concat_str = concat_str[:-1]
    return start_end_token_men


def mention_label_hot_encode(label_list, label2id):
    """
    get the mention labels one hot embedding, especially for multi-labels.
    when encoding multi-labels in one hot vector for single mention, the
    order of label follows by the label list dictionary in type_XX.txt.
    """
    embedding_len = len(label2id)
    men_label_code = np.zeros(embedding_len)
    label_id_list = [label2id[x] for x in label_list if x in label2id.keys()]
    if len(label_id_list) == 0:
        print("0000")
    men_label_code[label_id_list] = 1.0
    return men_label_code


def mention_label_dict(tokens, mentions):
    """
    get the dictionary with mention as the key and the labels as the value.
    """
    mention_label = {}  # construct the dictionary for mention-label pair, "mention-[label1, label2,...]"
    for men in mentions:
        start = int(men['start'])
        end = int(men['end'])
        temp = get_mention_surface_raw(tokens, start, end)
        if len(temp) > 0:
            mention_label[temp] = men['labels']
    return mention_label


def filter_noisy_label(mention_label, label2id):
    """
    filter the mention that only has the labels does not appear in training label.
    """
    new_mention_label = {}
    for men in mention_label.keys():
        label_list = mention_label[men]
        update_label_list = [x for x in label_list if x in label2id.keys()]
        if len(update_label_list) > 0:
            new_mention_label.update({men: update_label_list})
        # else:
        #     print(label_list)
    return new_mention_label


def get_mention_context(start, end, sentence_embedding, emb_len, window_size=5):
    """
    build the left and right contextual embedding for mentions, and fill with zero
    vector that has left or right neighbors less than window size.
    """
    left_context = []
    right_context = []
    token_length = len(sentence_embedding)
    sentence_key = list(sentence_embedding.keys())
    if start <= window_size:                     # left token size is less than 10
        for i in range(window_size-start):
            left_context.append([0. for x in range(emb_len)])
        for i in range(start):
            left_context.append(sentence_embedding[sentence_key[i]])
    if token_length - end <= window_size:        # right token size is less than 10
        for i in range(end+1, token_length):
            right_context.append(sentence_embedding[sentence_key[i]])
        for i in range(window_size+1-token_length+end):
            right_context.append([0. for x in range(emb_len)])
    if len(left_context) == 0:          # left token size is greater than 10
        for i in range(start-window_size, start):
            left_context.append(sentence_embedding[sentence_key[i]])
    if len(right_context) == 0:         # right token size is greater than 10
        for i in range(end + 1, end + window_size + 1):
            right_context.append(sentence_embedding[sentence_key[i]])
    return left_context, right_context


def get_mention_bert_emb(sentence_embedding, mention_label, tokenizer, timesteps, emb_len):
    """
    get the sequence of word embedding for single mention, and return the embedding list and the length.
    """
    men_embedding_list = []  # contains mention embedding
    men_length = []          # contains mention length
    sentence_key = list(sentence_embedding.keys())
    for men in mention_label.keys():
        men_token_embedding = []
        men_token = tokenizer.tokenize(men)
        for x in men_token:
            if x not in sentence_key:
                # print("---found one not in bert tokenized sequence.")
                # print(x)
                # print(len(sentence_key), sentence_key)
                # men_token_embedding.append(np.random.rand(emb_len))
                pass
            else:
                men_token_embedding.append(sentence_embedding[x])
        length = len(men_token)
        if length < timesteps:
            for i in range(timesteps-length):
                men_token_embedding.append([.0 for x in range(emb_len)])
        elif length > timesteps:
            men_token_embedding = men_token_embedding[:timesteps]
            length = timesteps
        men_embedding_list.append(men_token_embedding)
        men_length.append(length)
    return men_embedding_list, men_length


def mention_label_encode(sentence_embedding, mention_label, label2id,
                         tokenizer, timesteps, emb_len):
    """
    extract the mention embedding from the sentence embedding, and find the label encode for
    each mention embedding to construct the training example.
    """
    men_label_code = []      # contains label one hot code for the mention
    for men in mention_label.keys():
        men_label_code.append(mention_label_hot_encode(mention_label[men], label2id))
    mention_list = [x for x in mention_label.keys()]
    men_embedding_list, men_length = get_mention_bert_emb(sentence_embedding, mention_label,
                                                          tokenizer, timesteps, emb_len)
    return mention_list, men_embedding_list, men_label_code, men_length


def mention_context_label_encode(sentence_embedding, mention_label, label2id,
                         tokenizer, emb_len, window_size):
    """
    extract the mention left and right neighbors embedding from the sentence embedding,
    compute the average word embedding for each mention, and find the label encode for
    each mention embedding to construct the training example.
    """
    mention_list = []            # contains mention raw words
    left_context_list = []       # contains mention left neighbors embedding
    right_context_list = []      # contains mention right neighbors embedding
    men_label_code = []          # contains label one hot code for the mention in the example
    sentence_key = list(sentence_embedding.keys())   
    for men in mention_label.keys():
        men_token = tokenizer.tokenize(men)
        if (men_token[0] not in sentence_key) or ( men_token[-1] not in sentence_key):
            print(men_token[0], "not found in")
            print(sentence_key)
        else:
            start = sentence_key.index(men_token[0])
            end = sentence_key.index(men_token[-1])
            left_context, right_context = get_mention_context(
                start, end, sentence_embedding, emb_len, window_size)
            left_context_list.append(left_context)
            right_context_list.append(right_context)
            men_label_code.append(mention_label_hot_encode(mention_label[men], label2id))
            mention_list = [x for x in mention_label.keys()]
    return mention_list,  left_context_list,  right_context_list, men_label_code


def get_glove_avg_emb(mention_label, glove_emb_vec):
    """
    get the average word glove embedding as the whole mention embedding.
    """
    glove_avg_emb = []  # contains mention embedding
    emb_len = len(glove_emb_vec["a"])
    for men in mention_label.keys():
        word_seq = men.split(" ")
        emb_word = []
        emb_word += [glove_emb_vec[x.lower()] for x in word_seq
                     if x.lower() in glove_emb_vec.keys()]
        if len(emb_word) == 0:
            emb_word.append([0. for x in range(emb_len)])
        glove_avg_emb.append(list(np.sum(np.array(emb_word), axis=0) / float(len(emb_word))))
    return glove_avg_emb


def save_sentence(data, savefile):
    """
    extract the each token list in the raw data, and then save the whole sentence
    after concatenate those tokens into a file."""
    # _, mention_list, left_context, right_context = get_label_mention_dataset(
    #     data, label2id, index_start, index_end, is_train=True)
    # with open(savefile, "w") as f:
    #     for i in range(len(mention_list)):
    #         left_str = " ".join(left_context[i]) 
    #         right_str = " ".join(right_context[i]) 
    #         f.write(left_str+" "+mention_list[i]+" "+right_str+"\n")
    #         if i < len(mention_list)-1:
    #             f.write("\n")
    #     f.close()

    with open(savefile, "w") as f:
        for i in range(len(data)):
            tokens = data['tokens'][i]
            mention = data['mentions'][i]
            f.write(get_token_string(tokens, mention))
            if i < len(data)-1:
                f.write("\n")
        f.close()



def sentence_embedding_generator(indir):
    """
    load the sentence embedding for an existed sentence embedding file in a generator manner.
    """
    with open(indir, 'rb') as f:
        for line in f.readlines():
            dic = json.loads(line)
            sentence = {}
            for key, value in dic.items():
                if key == "features":
                    for item in value:
                        sentence[item["token"]] = item["layers"][0]["values"]
            yield sentence


def get_sentence_glove_dic(glove_emb_vec, sentence, tokenizer):
    """
    build a dictionary for the sentence with glove embedding
    """
    seg = tokenizer.tokenize(sentence)
    glove_emb_len = len(glove_emb_vec["a"])
    sentence_emb = list(map(lambda x: glove_emb_vec[x]
                       if x in glove_emb_vec.keys()
                       else np.random.rand(glove_emb_len), seg))
    return dict(zip(seg, sentence_emb))


def mention_embedding(data, label2id, sentence_generator, batch_start, batch_end,
                        tokenizer, timesteps, emb_len, is_train=True):
    """
    Do mention embedding to provide the mention embedding list and label code for training models.
    Return the mention embedding and label one-hot code
    """
    mention_list = []
    mention_embedding_list = []
    label_code_list = []
    mention_length_list = []
    for i in range(batch_start, batch_end):
        tokens = data['tokens'][i]
        mentions = data['mentions'][i]
        sentence = next(sentence_generator)

        # get the raw mention and label pair, and only keep the mention has labels in label2id.keys()
        mention_label = mention_label_dict(tokens, mentions)
        if not is_train:
            mention_label = filter_noisy_label(mention_label, label2id)
        raw_mention, bert_embedding, label_code, word_length = mention_label_encode(
            sentence, mention_label, label2id, tokenizer, timesteps, emb_len)
        # for i in mention_label.keys():
        #     print(mention_label[i])
        # for i in range(len(label_code)):
        #     print(set(np.where(label_code[i] == 1)[0]))
        mention_list += raw_mention
        label_code_list += label_code
        mention_embedding_list += bert_embedding
        mention_length_list += word_length
    return mention_list, mention_embedding_list, label_code_list, mention_length_list


def mention_glove_context_embedding(data, label2id, batch_start, batch_end,
                         glove_emb_vec, tokenizer, window_size, is_train=True):
    """
    Do mention embedding to provide the mention embedding list and label code for training models.
    Return the mention embedding, contextual embedding and label one-hot code
    """
    mention_embedding_list = []
    left_context_list = []       # contains mention left neighbors embedding
    right_context_list = []      # contains mention right neighbors embedding
    label_code_list = []
    emb_len = len(glove_emb_vec["a"])
    for i in range(batch_start, batch_end):
        tokens = data['tokens'][i]
        mentions = data['mentions'][i]
        sentence = get_token_string(tokens, mentions)
        sentence = get_sentence_glove_dic(glove_emb_vec, sentence, tokenizer)
        # get the raw mention and label pair, and only keep the mention has labels in label2id.keys()
        mention_label = mention_label_dict(tokens, mentions)
        if not is_train:
            mention_label = filter_noisy_label(mention_label, label2id)
        input_example = mention_context_label_encode(sentence, mention_label, label2id, tokenizer, emb_len, window_size)
        _, left_context, right_context, label_code = input_example
        avg_embedding = get_glove_avg_emb(mention_label, glove_emb_vec)
        label_code_list += label_code
        mention_embedding_list += avg_embedding
        left_context_list += left_context
        right_context_list += right_context
    return mention_embedding_list, left_context_list, right_context_list, label_code_list


def mention_bert_context_embedding(data, label2id, sentence_generator, batch_start, batch_end,
                        tokenizer, emb_len, timesteps, window_size, is_train=True):
    """
    Do mention embedding to provide the mention embedding list and label code for training models.
    Return the mention embedding, contextual embedding and label one-hot code
    """
    mention_list = []
    mention_embedding_list = []
    left_context_list = []       # contains mention left neighbors embedding
    right_context_list = []      # contains mention right neighbors embedding
    label_code_list = []
    mention_len_list = []
    for i in range(batch_start, batch_end):
        tokens = data['tokens'][i]
        mentions = data['mentions'][i]
        sentence = next(sentence_generator)
        # get the raw mention and label pair, and only keep the mention has labels in label2id.keys()
        mention_label = mention_label_dict(tokens, mentions)
        if not is_train:
            mention_label = filter_noisy_label(mention_label, label2id)
        bert_embedding, word_length = get_mention_bert_emb(sentence, mention_label, tokenizer, timesteps, emb_len)
        input_example = mention_context_label_encode(
            sentence, mention_label, label2id, tokenizer, emb_len, window_size)
        raw_mention, left_context, right_context, label_code = input_example
        mention_list += raw_mention
        label_code_list += label_code
        left_context_list += left_context
        right_context_list += right_context
        mention_embedding_list += bert_embedding
        mention_len_list += word_length
    return mention_list, left_context_list, right_context_list, label_code_list, mention_embedding_list,\
           mention_len_list


def get_mention_list(data, label2id, is_train=True):
    """
    Only get the mention list form the data, which is used for statistics on mention data.
    """
    mention_list = []
    for i in range(len(data)):
        tokens = data['tokens'][i]
        mentions = data['mentions'][i]

        # get the raw mention and label pair, and remove the mention in test set that only has the label
        # which dose not appear in train set, to keep the label type consistence on train and test set.
        mention_label = mention_label_dict(tokens, mentions)
        if not is_train:
            mention_label = filter_noisy_label(mention_label, label2id)

        mention_list += [x for x in mention_label.keys()]
    return mention_list


def get_context_raw_string(sentence, start, end, side_window=20):
    """
    return raw string for mention context. extract the leftside and rightside words
    in fix-sized window to be the context.
    """
    split_sentence = sentence.split()
    max_len = len(split_sentence)
    temp = split_sentence
    del temp[start: end]
    context_sentence = ' '.join([str(elem) for elem in temp])
    left_context, right_context = None, None
    if len(temp) <= 2 * side_window:
        left_context = split_sentence[:start]
        right_context = split_sentence[end:]
    else:
        if start >= side_window and end + side_window < max_len:
            left_context = ' '.join([str(elem) for elem in split_sentence[start-side_window: start]])
            right_context = ' '.join([str(elem) for elem in split_sentence[end: end + side_window]])
            context_sentence = left_context + " " + right_context
        elif start < side_window and end + side_window < max_len:
            left_context = ' '.join([str(elem) for elem in split_sentence[0: start]])
            right_context = ' '.join([str(elem) for elem in split_sentence[end: end + side_window]])
            context_sentence = left_context + " " + right_context
        elif start >= side_window and end + side_window >= max_len:
            left_context = ' '.join([str(elem) for elem in split_sentence[start - side_window: start]])
            right_context = ' '.join([str(elem) for elem in split_sentence[end: max_len]])
            context_sentence = left_context + " " + right_context
        else:
            print("ERROR", len(temp), start, end, sentence)
    return left_context, right_context


def get_label_mention_dataset(data, label2id, index_start, index_end, is_train=True):
    """
    Construct train, dev, test dataset in the formal of (label_id, mention, context),
    where label_id is unique id for each label, mention is a set of words for a mention,
    context is the rest part in the sentence except the mention words.
    """
    mention_list = []
    left_context_list = []
    right_context_list = []
    label_list = []
    for i in range(index_start, index_end):
        tokens = data["tokens"][i]
        mentions = data["mentions"][i]
        sentence = get_token_string(tokens, mentions)
        mention_label = mention_label_dict(tokens, mentions)
        if not is_train:
            mention_label = filter_noisy_label(mention_label, label2id)
        for men in mentions:
            start = int(men['start'])
            end = int(men['end'])
            mention_str = get_mention_surface_raw(tokens, start, end)
            if mention_str in mention_label.keys():
                left_context, right_context = get_context_raw_string(sentence, start, end)
                label_one_hot = mention_label_hot_encode(mention_label[mention_str], label2id)
                mention_list.append(mention_str)
                left_context_list.append(left_context)
                right_context_list.append(right_context)
                label_list.append(label_one_hot)
    return label_list, mention_list, left_context_list, right_context_list


def save_tsv_file(file_name, label_list, mention_list, context_list):
    with open(file_name, "w") as f:
        for i in range(len(label_list)):
            label_str = ','.join([str(int(item)) for item in label_list[i]])
            f.write(str(i)+"\t"+label_str+"\t"+str(mention_list[i])+"\t"+context_list[i]+"\n")
        f.close()







'''

python bert/extract_features.py \
  --input_file=./BERT_ZSL/Data/Wiki/intermediate/sentence_train.txt \
  --output_file=./BERT_ZSL/Data/Wiki/intermediate/sentence_emb_train.json \
  --vocab_file=./bert/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=./bert/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=./bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=100 \
  --batch_size=4



python bert/extract_features.py \
  --input_file=./BERT_ZSL/Data/Wiki/intermediate/sentence_test.txt \
  --output_file=./BERT_ZSL/Data/Wiki/intermediate/sentence_emb_test.json \
  --vocab_file=./bert/uncased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=./bert/uncased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=./bert/uncased_L-12_H-768_A-12/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=100 \
  --batch_size=8



python bert/extract_features.py \
  --input_file=./BERT_ZSL/Data/BBN/intermediate/sentence_train.txt \
  --output_file=./BERT_ZSL/Data/BBN/intermediate/sentence_emb_large_train.json \
  --vocab_file=./bert/wwm_cased_L-24_H-1024_A-16/vocab.txt \
  --bert_config_file=./bert/wwm_cased_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=./bert/wwm_cased_L-24_H-1024_A-16/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=200 \
  --batch_size=8

python bert/extract_features.py \
  --input_file=./BERT_ZSL/Data/BBN/intermediate/sentence_test.txt \
  --output_file=./BERT_ZSL/Data/BBN/intermediate/sentence_emb_large_test.json \
  --vocab_file=./bert/wwm_cased_L-24_H-1024_A-16/vocab.txt \
  --bert_config_file=./bert/wwm_cased_L-24_H-1024_A-16/bert_config.json \
  --init_checkpoint=./bert/wwm_cased_L-24_H-1024_A-16/bert_model.ckpt \
  --layers=-1 \
  --max_seq_length=400 \
  --batch_size=8

'''