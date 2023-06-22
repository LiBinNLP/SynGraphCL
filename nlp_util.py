#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2023/3/13 16:04
# @Author : libin
# @File : sem_util.py
# @Software: PyCharm

import numpy as np
import stanza
from scipy import spatial
import gensim
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import pandas as pd


class sem_calc():

    def __init__(self):
        #load word2vec model, here GoogleNews is used
        self.model = gensim.models.KeyedVectors.load_word2vec_format('/mnt/sda1_hd/atur/libin/projects/wordvec/english/GoogleNews-vectors-negative300.bin', binary=True)
        self.index2word_set = set(self.model.index_to_key)

    def avg_feature_vector(self, sentence, model, num_features, index2word_set):
        words = sentence.split()
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in index2word_set:
                n_words += 1
                feature_vec = np.add(feature_vec, model[word])
        if (n_words > 0):
            feature_vec = np.divide(feature_vec, n_words)
        return feature_vec

    def calc_sem_similarity(self, sent1, sent2):
        s1_afv = self.avg_feature_vector(sent1, model=self.model, num_features=300, index2word_set=self.index2word_set)
        s2_afv = self.avg_feature_vector(sent2, model=self.model, num_features=300, index2word_set=self.index2word_set)
        sim = 1 - spatial.distance.cosine(s1_afv, s2_afv)
        return sim

class nlp_util():
    def __init__(self):
        try:
            self.nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', download_method=None,
                                       tokenize_pretokenized=True, tokenize_no_ssplit=True)
        except Exception as e:
            stanza.download('en')

    def pipline(self, sentence):
        doc = self.nlp(sentence)
        entities = doc.sentences[0].entities
        constituency = doc.sentences[0].constituency
        return entities, constituency

    def nltk_ner(self, sentence):
        # 分词
        tokenized_sentence = nltk.word_tokenize(sentence)
        # 标注词性
        tagged_sentence = nltk.tag.pos_tag(tokenized_sentence)
        # 命名实体识别
        ne_tagged_sentence = nltk.ne_chunk(tagged_sentence)
        # extract all named entities
        named_entities = []

        for tagged_tree in ne_tagged_sentence:
            # extract only chunks having NE labels
            if hasattr(tagged_tree, 'label'):
                entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  # get NE name
                entity_type = tagged_tree.label()  # get NE category
                named_entities.append((entity_name, entity_type))
                # get unique named entities
                named_entities = list(set(named_entities))

        # # store named entities in a data frame
        # entity_frame = pd.DataFrame(named_entities, columns=['Entity Name', 'Entity Type'])
        # # display results
        # print(entity_frame)
        return named_entities

    def stanza_test(self):
        doc = self.nlp("Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29.")
        for sentence in doc.sentences:
            print(sentence.entities)
            print(sentence.constituency)

if __name__ == '__main__':
    sent1 = 'Today is sunny'
    sent2 = 'The weather is well'
    # sem_calc_util = sem_calc()
    # sim = sem_calc_util.calc_sem_similarity(sent1, sent2)
    # print(sim)

    # sent3 = 'China is the greatest country'
    # nltk_util = nltk_util()
    # nltk_util.nltk_ner(sent3)

    util = nlp_util()
    util.stanza_test()
