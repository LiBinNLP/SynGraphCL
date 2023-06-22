#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2023/3/8 15:18
# @Software: PyCharm
import copy
import random
from nltk.corpus import wordnet as wn

from data_read import load_conllu_data
import numpy as np

REPLACE_TAG = ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # [NNP, NNPS]
REPLACE_POS = ['NOUN', 'VERB', 'ADJ', 'ADV']
POS_TO_TAGS = {'NOUN': ['NN', 'NNS', 'NNP'],
               'ADJ': ['JJ', 'JJR', 'JJS'],
               'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
               'ADV': ['RB', 'RBR', 'RBS']}


# the ratio of perturbation
edge_remove_ratio = 0.5
edge_label_change_ratio = 1-edge_remove_ratio
edge_add_ratio = 0.5


def load_dict(path):
    lines = []
    with open(path, encoding='utf-8') as fin:
        for line in fin.readlines():
            line = line.strip('\n')
            lines.append(line)
        fin.close()

    return lines

def sample_from_dict(name, dict):
    new_entity = ''
    while new_entity == name or len(new_entity.split()) != len(name.split()):
        new_entity = random.sample(dict, 1)[0]
    dict.remove(new_entity)
    return new_entity


def get_antonym_words(token):
    lemma = token.lemma_
    text = token.text
    tag = token.tag_
    pos = token.pos_
    word_antonym = set()
    if pos not in REPLACE_POS:
        return list(word_antonym)

    synsets = wn.synsets(text, pos=eval("wn."+pos))
    for synset in synsets:
        for synlemma in synset.lemmas():
            for antonym in synlemma.antonyms():
                word = antonym.name()
                #word = wnl.lemmatize(word, pos=eval("wn."+pos))
                if word.lower() != text.lower() and word.lower() != lemma.lower():
                    # inflt = getInflection(word, tag=tag)
                    # word = inflt[0] if len(inflt) else word
                    word = word.replace('_', ' ')
                    word_antonym.add(word)

    return list(word_antonym)


def write_conllu(fw, instance):
    sent_id = instance['sent_id']
    words = instance['words']
    values = instance['values']
    gen_from = instance['gen_from']

    fw.write('# format = sdp\n')
    fw.write('# sent_id = ' + sent_id + '\n')
    fw.write('# gen_from = ' + gen_from + '\n')
    fw.write('# text = ' + ' '.join(words) + '\n')
    for value in values:
        fw.write('	'.join(value) + '\n')
    fw.write('\n')


def edges_to_values(values, edges):
    '''
    Convert edges of semantic dependency parsing graph to conllu format
    :param values:
    :param edges:
    :return:
    '''
    # add a ROOT node in values
    values.insert(0, ['0', 'ROOT', 'root', 'ROOT', 'ROOT', '_', None, None, None, '_'])

    # set None to the head, label and dependencies of each token
    for value in values:
        value[6] = ''
        value[7] = ''
        value[8] = ''

    # assign the head, label and dependencies for each token
    for edge in edges:
        head_idx = int(edge[0])
        dep_idx = int(edge[1])
        label = edge[2]

        if values[dep_idx][6] == '':
            values[dep_idx][6] = str(head_idx)
        if values[dep_idx][7] == '':
            values[dep_idx][7] = label
        if values[dep_idx][8] == '':
            values[dep_idx][8] = '{}:{}'.format(head_idx, label)
        else:
            values[dep_idx][8] = values[dep_idx][8] + '|{}:{}'.format(head_idx, label)
    return values

def rep_synonymous_word(instance):
    sent_id = instance['sent_id']
    words = instance['words']
    values = instance['values']
    change_flag = instance['change_flag']

    # replace original words with synonymous words
    for idx in range(1, len(words)):
        if not change_flag[idx]:
            word = words[idx]
            pos = values[idx - 1][3]
            syn_word = get_synonymous_words(sent, word, pos)
            if syn_word != None and '_' not in syn_word:
                words[idx] = syn_word
                values[idx - 1][1] = syn_word
                change_flag[idx] = True
                instance['sent_id'] = 'Gen Positive-' + sent_id


def add_erroneous_edge(instance, label_set):
    '''
    This is one of methods for generating negative samples.
    The idea is that remove all correct dependency edges and randomly add erroneous
    dependency edges for node pairs that have no dependency relations.
    :param instance:
    :param label_set:
    :return: instance
    '''
    sent_id = instance['sent_id']
    words = instance['words']
    values = instance['values']
    edges = instance['edges']
    num_nodes = len(words)
    num_edges = pow(num_nodes, 2)
    num_labels = len(label_set)
    num_sampled_edges = int(num_edges * edge_add_ratio)
    temp_edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            temp_edges.append([i, j])
    matrix = np.array(temp_edges)

    sampled_edges = matrix[np.random.choice(matrix.shape[0], num_sampled_edges, replace=False), :]
    sampled_edges = sampled_edges.tolist()
    for edge in edges:
        head_idx = edge[0]
        dep_idx = edge[1]
        label = edge[2]
        if [int(head_idx), int(dep_idx)] in sampled_edges:
            sampled_edges.remove([int(head_idx), int(dep_idx)])

    labels = list(label_set)
    sampled_labels = np.random.choice(num_labels, len(sampled_edges))
    new_edges = []
    for edge, label in zip(sampled_edges, sampled_labels):
        new_edges.append([edge[0], edge[1], labels[label]])

    new_values = edges_to_values(values, new_edges)
    gen_from = 'neg.add_edge'
    instance.update({'gen_from': gen_from})
    instance['sent_id'] = sent_id
    instance['values'] = new_values
    instance['edges'] = new_edges

    return instance


def change_dep_token(instance):
    '''
    This is one of methods for generating negative samples.
    The idea is that change the dependency token id randomly for each dependency edge
    :param instance:
    :return: instance
    '''
    sent_id = instance['sent_id']
    words = instance['words']
    values = instance['values']
    edges = instance['edges']
    num_nodes = len(words)
    sampled_dep_idx = np.random.choice(range(num_nodes), len(edges))
    for i in range(len(edges)):
        edge = edges[i]
        dep_id = edge[1]
        sampled_dep_id = sampled_dep_idx[i]
        if dep_id != sampled_dep_id:
            edge[1] = sampled_dep_id
        else:
            edge[1] = sampled_dep_id + 1

    new_values = edges_to_values(values, edges)

    gen_from = 'neg.change_dep_token'
    instance.update({'gen_from': gen_from})
    instance['sent_id'] = sent_id
    instance['values'] = new_values
    instance['edges'] = edges

    return instance


def exchange_head_dep(instance):
    '''
    This is one of methods for generating negative samples.
    The idea is that exchange the head and dependency token id
    :param instance:
    :return: instance
    '''
    sent_id = instance['sent_id']
    values = instance['values']
    edges = instance['edges']
    for i in range(len(edges)):
        edge = edges[i]
        head_id = edge[0]
        dep_id = edge[1]
        edge[0] = dep_id
        edge[1] = head_id

    new_values = edges_to_values(values, edges)
    gen_from = 'neg.exchange_head_dep'
    instance.update({'gen_from': gen_from})
    instance['sent_id'] = sent_id
    instance['values'] = new_values
    instance['edges'] = edges

    return instance

def change_edge_label(instance, label_dict):
    '''
    This is one of methods for generating negative samples.
    The idea is that randomly change the label of each dependency edge
    :param label_dict:
    :param instance:
    :return: instance
    '''
    sent_id = instance['sent_id']
    values = instance['values']
    edges = instance['edges']
    labels = list(label_dict)
    sampled_labels = np.random.choice(labels, len(edges))
    for i in range(len(edges)):
        edge = edges[i]
        label = edge[2]
        sampled_label = sampled_labels[i]
        if label != sampled_label:
            edge[2] = sampled_label
        else:
            edge[2] = sampled_labels[i-1]

    new_values = edges_to_values(values, edges)
    gen_from = 'neg.change_label'
    instance.update({'gen_from': gen_from})
    instance['sent_id'] = sent_id
    instance['values'] = new_values
    instance['edges'] = edges

    return instance

if __name__ == '__main__':

    # read conllu data
    all_instances, all_seq_len, labels = load_conllu_data('./data/contrastive/original.conllu')

    # randomly sample instances from total dataset
    total_num = len(all_instances)
    type_set = set()
    # generate new instances with rule-based approach
    with open('./data/contrastive/negative.conllu', encoding='utf-8', mode='w+') as fw:
        for instance in all_instances:
            sent_id = instance['sent_id']
            words = instance['words']

            values = instance['values']
            sent = ' '.join(words[1:])

            instance1 = add_erroneous_edge(copy.deepcopy(instance), labels)
            write_conllu(fw, instance1)

            instance2 = change_dep_token(copy.deepcopy(instance))
            write_conllu(fw, instance2)

            instance3 = exchange_head_dep(copy.deepcopy(instance))
            write_conllu(fw, instance3)

            # instance4 = change_edge_label(copy.deepcopy(instance), labels)
            # write_conllu(fw, instance4)

    fw.close()