__author__ = 'tzhang'

import numpy as np


def extract(mention):
    """
    extract whole label set form the dataset and return the
    label dictionary
    """
    label_dic = []
    for i in range(len(mention)):
        label_dic += [label for men in mention[i] for label in men['labels']]
    label_dic = set(label_dic)
    label_dic = dict(zip(range(len(label_dic)), label_dic))
    return label_dic


def extract_label_type(label_set):
    """
    reconstruct the list with separated label level,
    which means list[0] contains all level-1 labels,
    while list[1] contains level-2 labels.
    """
    label_level_list = []
    label_level_list.append([x for x in label_set if x.count('/') == 1])
    label_level_list.append([x for x in label_set if x.count('/') == 2])
    return label_level_list


def write_label(file_name, label_list):
    """
    write level-separated label list into files.
    """
    t = open(file_name, 'w')
    id = 0
    for level_list in label_list:
        for label in level_list:
            t.write(label + '\t' + str(id) + '\n')
            id += 1
    t.close()


def supertype(type_dir, supertype_dir):
    """
    generate the child -> parents pair type to indicate the hierarchy structure.
    example (type.txt):
                /ORGANIZATION	6
                /ORGANIZATION/CORPORATION	4
    results in supertype.txt:
                4   6
    """
    with open(type_dir, 'r') as f, open(supertype_dir, 'w') as g:
        mm = {}
        for line in f:
            seg = line.strip('\r\n').split('\t')
            mm[seg[0]] = seg[1]

        for key1 in mm:
            for key2 in mm:
                if key1 != key2:
                    seg1 = key1[1:].split('/')
                    seg2 = key2[1:].split('/')
                    if len(seg1) == len(seg2) + 1:
                        flag = True
                        for k in range(len(seg2)):
                            if seg1[k] != seg2[k]:
                                flag = False
                                break
                        if flag:
                            g.write(mm[key1] + '\t' + mm[key2] + '\n')





