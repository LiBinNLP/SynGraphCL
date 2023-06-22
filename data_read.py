#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2021/11/8 21:38
# @File : data_utils.py
# @Software: PyCharm
import os
import unicodedata

import numpy as np
import torch


def write_conllu_data(path, instances):
    '''
    写入conllu data
    :param path:
    :param instances:
    :return:
    '''
    with open(path, 'w+') as fw:
        for instance in instances:
            sent_id = instance['sent_id']
            words = instance['words']
            values = instance['values']

            fw.write('# format = sdp\n')
            fw.write('# sent_id = ' + sent_id + '\n')
            fw.write('# text = ' + ' '.join(words) + '\n')
            for value in values:
                fw.write('	'.join(value) + '\n')
            fw.write('\n')
        fw.close()


def load_conllu_data(file_path):
    """
    加载conllu格式的数据
    :param file_path:
    :return:
    """
    all_instances = []
    all_seq_len = []
    labels = set()
    edges = []
    with open(file_path, 'r') as fp:
        values = []
        sent_id = 0
        for line in fp:
            if 'sent_id' in line:
                sent_id = line.split('=')[1].replace(' ', '').replace('\n', '')
            elif not line.startswith("#") and line != '\n':
                value = line.replace('\n', '').split('\t')
                values.append(value)
                pheads = value[8].split('|')
                for phead in pheads:
                    if ':' in phead:
                        id_rels = phead.split(':')
                        rels = id_rels[1]
                        labels.add(rels)
                        edges.append([id_rels[0], value[0], rels])
            elif line == '\n':
                word_list = ['ROOT']
                words = [value[1] for value in values]
                word_list.extend(words)
                all_seq_len.append(len(word_list))
                if sent_id != 0:
                    all_instances.append({'sent_id': sent_id, 'words': word_list, 'values': values, 'edges':edges})
                else:
                    sent_id += 1
                    all_instances.append({'sent_id': sent_id, 'words': word_list, 'values': values, 'edges':edges})
                values = []
                edges = []

    return all_instances, all_seq_len, labels




def load_sdp_data(gold_file_name, init_file_name):
    """
    加载Semantic Dependency Parsing数据.
    :param gold_file_name:
    :param init_file_name:
    :param data_split:
    :param seed:
    :return:
    """
    all_instances_g, all_seq_len_g, labels_g = load_conllu_data(gold_file_name)
    all_instances_p, all_seq_len_p, labels_p = load_conllu_data(init_file_name)

    all_instances = []
    all_seq_len = all_seq_len_g
    # 将所有边的类型放在一个集合中去重
    labels_set = labels_g.union(labels_p)

    labels = list()
    # 如果两个节点之间没有边，则将其表示为NoRel，idx=0
    labels.extend(list(labels_set))

    for gold_instance, pred_instance in zip(all_instances_g, all_instances_p):
        all_instances.append({'sent_id': gold_instance['sent_id'], 'words': gold_instance['words'], 'values': gold_instance['values'], 'gold':gold_instance['edges'], 'pred':pred_instance['edges']})

    # 最长的句子
    print('[ Max seq length: {} ]'.format(np.max(all_seq_len)))
    # 最短的句子
    print('[ Min seq length: {} ]'.format(np.min(all_seq_len)))
    # 平均句子长度
    print('[ Mean seq length: {} ]'.format(int(np.mean(all_seq_len))))


    return all_instances, labels




def load_conllx_data(file_path):
    """
    加载conllx格式依存句法解析的数据

    :param file_path:
    :return:
    """
    instances = []
    labels = set()
    with open(file_path, 'r') as fp:
        values = []
        sent_id = 0
        for line in fp:
            if 'sent_id' in line:
                sent_id = line.split('=')[1].replace(' ', '').replace('\n', '')
            elif not line.startswith("#") and line != '\n':
                value = line.replace('\n', '').split('\t')
                values.append(value)
                rel = value[7]
                if rel not in labels:
                    labels.add(rel)
            elif line == '\n':
                word_list = ['ROOT']
                words = [value[1] for value in values]
                word_list.extend(words)
                if sent_id != 0:
                    instances.append({'sent_id': sent_id, 'words': word_list, 'values': values})
                else:
                    sent_id += 1
                    instances.append({'sent_id': sent_id, 'words': word_list, 'values': values})
                values = []

    return instances, labels



def vectorize_input(batch, config, training=True, device=None):
    # Check there is at least one valid example in batch (containing targets):
    if not batch:
        return None

    # Relevant parameters:
    batch_size = len(batch.sent1_word)

    context = torch.LongTensor(batch.sent1_word)
    context_lens = torch.LongTensor(batch.sent1_length)
    if config['task_type'] == 'regression':
        targets = torch.Tensor(batch.labels)
    elif config['task_type'] == 'classification':
        targets = torch.stack(batch.gold, 0)
        sources = torch.stack(batch.pred, 0)
        init_adj = torch.Tensor(batch.init_adj)
    else:
        raise ValueError('Unknwon task_type: {}'.format(config['task_type']))

    if batch.has_sent2:
        context2 = torch.LongTensor(batch.sent2_word)
        context2_lens = torch.LongTensor(batch.sent2_length)

    with torch.set_grad_enabled(training):
        example = {'batch_size': batch_size,
                   'context': context.to(device) if device else context,
                   'context_lens': context_lens.to(device) if device else context_lens,
                   'targets': targets.to(device) if device else targets,
                   'sources': sources.to(device) if device else sources,
                   'init_adj': init_adj.to(device) if device else init_adj,
                   }

        if batch.has_sent2:
            example['context2'] = context2.to(device) if device else context2
            example['context2_lens'] = context2_lens.to(device) if device else context2_lens
        return example


def prepare_datasets(gold_file_name, pred_file_name):
    # 加载文本数据集
    all_instances, labels = load_sdp_data(gold_file_name, pred_file_name)
    print('# of examples: {}'.format(len(all_instances)))
    return all_instances, labels

def ispunct(token: str) -> bool:
    return all(unicodedata.category(char).startswith('P') for char in token)