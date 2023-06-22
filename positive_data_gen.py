#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2023/3/8 15:18
# @Software: PyCharm
import random
from nltk.corpus import wordnet as wn
import numpy as np
from data_read import load_conllu_data
from nlp_util import sem_calc, nlp_util

REPLACE_TAG = ['NN', 'NNS', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'] # [NNP, NNPS]
REPLACE_POS = ['NOUN', 'VERB', 'ADJ', 'ADV']
POS_TO_TAGS = {'NOUN': ['NN', 'NNS', 'NNP'],
               'ADJ': ['JJ', 'JJR', 'JJS'],
               'VERB': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
               'ADV': ['RB', 'RBR', 'RBS']}

sem_calc_util = sem_calc()
nlp_util = nlp_util()

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
    try_count = 0
    while True:
        new_entity = random.sample(dict, 1)[0]
        try_count += 1
        if try_count >= 10:
            new_entity = None
            break
        if '-' not in new_entity and new_entity != name and len(new_entity.split()) == len(name.split()):
            break
    return new_entity

def get_synonymous_words(sent, word, pos, lemma):
    normal_pos = None
    for (key, value) in POS_TO_TAGS.items():
        if pos in value:
            normal_pos = key
            break

    if normal_pos == None:
        return None

    synwords = wn.synsets(word, pos=eval('wn.'+normal_pos))

    max_sim = 0
    max_index = 0
    syn_lemma_list = []
    for idx in range(0, len(synwords)):
        synword = synwords[idx]
        syn_lemmas = wn.synset(synword._name).lemma_names()
        for syn_lemma in syn_lemmas:
            if word.lower() != syn_lemma.lower() and syn_lemma != lemma and '_' not in syn_lemma:
                syn_lemma_list.append(syn_lemma)
                # definition = synword.definition()
                # # 计算该词所在句子与同义词的相似度，选择最相似的同义词返回
                # sim = sem_calc_util.calc_sem_similarity(sent, definition)
                # if sim > max_sim:
                #     max_index = idx
                #     max_sim = sim

    # if max_sim == 0:
    #     return None
    # else:
    #     syn_lemmas = wn.synset(synwords[max_index]._name).lemma_names()
    #     syn_lemma = np.random.choice(syn_lemmas, 1)
    if len(syn_lemma_list) > 0:
        syn_lemma = np.random.choice(syn_lemma_list, 1)[0]
    else:
        syn_lemma = None
    return syn_lemma


def write_conllu(fw, instance):
    sent_id = instance['sent_id']
    words = instance['words']
    values = instance['values']
    try:
        gen_from = instance['gen_from']
    except:
        gen_from = None

    fw.write('# format = sdp\n')
    fw.write('# sent_id = ' + sent_id + '\n')
    if gen_from is not None:
        fw.write('# gen_from = ' + gen_from + '\n')
    fw.write('# text = ' + ' '.join(words) + '\n')
    for value in values:
        fw.write('	'.join(value) + '\n')
    fw.write('\n')

def rep_same_entity(named_entities, instance):
    sent_id = instance['sent_id']
    words = instance['words']
    values = instance['values']
    change_flag = instance['change_flag']

    for entity in named_entities:
        name = entity.text
        type = entity.type

        new_entity = None
        if type == 'PERSON':
            new_entity = sample_from_dict(name, person_dict)
        elif type == 'GPE':
            new_entity = sample_from_dict(name, GPE_dict)
        elif type == 'ORG':
            new_entity = sample_from_dict(name, organization_dict)
        elif type == 'LOC':
            new_entity = sample_from_dict(name, location_dict)
        elif type == 'FAC':
            new_entity = sample_from_dict(name, facility_dict)
        elif type == 'NORP':
            new_entity = sample_from_dict(name, nationality_dict)
        elif type == 'WORK_OF_ART':
            new_entity = sample_from_dict(name, artwork_dict)
        elif type == 'LANGUAGE':
            new_entity = sample_from_dict(name, language_dict)
        elif type == 'ORDINAL':
            new_entity = sample_from_dict(name, ordinal_dict)
        elif type == 'CARDINAL':
            new_entity = sample_from_dict(name, cardinal_dict)
        elif type == 'PRODUCT':
            new_entity = sample_from_dict(name, manmade_dict)
        elif type == 'EVENT':
            new_entity = sample_from_dict(name, event_dict)
        elif type == 'LAW':
            new_entity = sample_from_dict(name, law_dict)
        elif type == 'PERCENT':
            pass
        elif type == 'TIME':
            pass
        elif type == 'DATE':
            pass
        elif type == 'MONEY':
            pass
        else:
            print(type, name)

        if new_entity != None and new_entity != '':
            if len(name.split()) == 1:
                first_token = name
            else:
                first_token = name.split()[0]
            if first_token not in words:
                continue
            idx = words.index(first_token)
            if idx != -1:
                for token in new_entity.split():
                    # words[idx] = token
                    values[idx - 1][1] = token
                    values[idx - 1][2] = token.lower()
                    change_flag[idx] = True
                    gen_from = 'pos.same_entity'
                    instance.update({'gen_from': gen_from})
                    idx += 1


def rep_paraphrase(instance, constituency):
    sent_id = instance['sent_id']
    words = instance['words']
    values = instance['values']
    change_flag = instance['change_flag']


def rep_synonymous_word(instance):
    '''
    replace the original word with the synonymous word from the wordnet
    :param instance:
    :return:
    '''
    words = instance['words']
    values = instance['values']
    change_flag = instance['change_flag']

    # replace original words with synonymous words
    for idx in range(1, len(words)):
        if not change_flag[idx]:
            word = words[idx]
            pos = values[idx - 1][4]
            lemma = values[idx - 1][2]
            syn_word = get_synonymous_words(sent, word, pos, lemma)
            if syn_word != None and '_' not in syn_word:
                words[idx] = syn_word
                values[idx - 1][1] = syn_word
                values[idx - 1][2] = syn_word.lower()
                change_flag[idx] = True
                gen_from = 'pos.synonymous_word'
                instance.update({'gen_from': gen_from})


if __name__ == '__main__':
    # read word dictionary
    dict_path = './data/dict/'
    person_dict = load_dict(dict_path + 'Person.txt')
    GPE_dict = load_dict(dict_path + 'GPE.txt')
    location_dict = load_dict(dict_path + 'Location.txt')
    organization_dict = load_dict(dict_path + 'Organization.txt')
    corporation_dict = load_dict(dict_path + 'Corporation.txt')
    facility_dict = load_dict(dict_path + 'Facility.txt')
    nationality_dict = load_dict(dict_path + 'Facility.txt')
    cardinal_dict = load_dict(dict_path + 'Cardinal.txt')
    ordinal_dict = load_dict(dict_path + 'Ordinal.txt')
    language_dict = load_dict(dict_path + 'Language.txt')
    manmade_dict = load_dict(dict_path + 'ManMade.txt')
    artwork_dict = load_dict(dict_path + 'ArtWork.txt')
    law_dict = load_dict(dict_path + 'Law.txt')
    event_dict = load_dict(dict_path + 'Event.txt')

    # read conllu data
    all_instances, all_seq_len, labels = load_conllu_data('./data/contrastive/original.conllu')

    # generate new instances with rule-based approach
    with open('./data/contrastive/positive.conllu', encoding='utf-8', mode='w+') as fw:
        for instance in all_instances:
            sent_id = instance['sent_id']
            print('the sent {} is being processing'.format(sent_id))
            words = instance['words']
            change_flag = [False for _ in range(len(words))]
            instance.update({'change_flag': change_flag})
            values = instance['values']
            sent = ' '.join(words[1:])
            # print('nlp_util.pipeline')
            named_entities, constituency = nlp_util.pipline(sent)
            for entity in named_entities:
                type = entity.type
                text = entity.text
            # replace original named entity with same-type named entity
            rep_same_entity(named_entities, instance)
            # replace original words with synonymous words
            rep_synonymous_word(instance)
            if 'gen_from' in instance:
                write_conllu(fw, instance)
    fw.close()