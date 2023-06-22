#!D:/Code/python
# -*- coding: utf-8 -*-
# @Time : 2023/4/11 1:19
# @File : stanza_dep_parse.py
# @Software: PyCharm
'''
The function of this script  is that using stanza to parse a file to another file with dependency parser
'''
import argparse
import os
import stanza
from data_read import load_conllu_data

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def write_conllu(fw, sent_id, words, values):
    fw.write('# format = dep\n')
    fw.write('# sent_id = {}'.format(sent_id) + '\n')
    fw.write('# text = ' + ' '.join(words) + '\n')
    for value in values:
        fw.write('	'.join(value) + '\n')
    fw.write('\n')


def parse(nlp, sentence):
    '''
    use stanza parser to parse a sentence
    :param nlp:
    :param sentence:
    :return:
    '''
    values = []
    tokens = []
    token_id = 0
    assert nlp != None
    doc = nlp(sentence)
    results = doc.sentences
    for result in results:
        dependencies = result.dependencies

        for idx in range(len(dependencies)):
            token_id += 1
            dependency = dependencies[idx]
            word_info = dependency[2]
            token = word_info.text
            tokens.append(token)
            head_idx = word_info.head
            deprel = word_info.deprel
            word_id = word_info.id
            # if one sentence has been parsed into two sentences
            if token_id != word_id and len(results) > 1:
                offset = token_id - word_id
                word_id += offset
                head_idx += offset
            value = [str(word_id), token, word_info.lemma, word_info.upos, word_info.xpos, '_',
                     str(head_idx), deprel, '{}:{}'.format(head_idx, deprel), '_']
            # xpos 细粒度词性 upos 粗粒度词性
            values.append(value)
    return tokens, values

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dependency parsing with stanza parser')
    parser.add_argument('--format', type=str, default='flat', help='format of input file')
    parser.add_argument('--input', type=str, default=r'../data/contrastive/raw_sentence.txt', help='input file path')
    parser.add_argument('--output', type=str, default=r'../data/contrastive/original.conllu', help='output file path')
    args = parser.parse_args()

    nlp = None

    if args.format == 'conllu':
        # sentence in conllu file has been pretokenized
        try:
            nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', download_method=None,
                                  tokenize_pretokenized=True, tokenize_no_ssplit=True)
        except Exception as e:
            stanza.download('en')
            nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', download_method=None,
                                  tokenize_pretokenized=True, tokenize_no_ssplit=True)

        all_instances, all_seq_len, labels = load_conllu_data(args.input)
        with open(args.output, mode='w+') as fw:
            for instance in all_instances:
                try:
                    sent_id = instance['sent_id']
                    words = instance['words']
                    sent = ' '.join(words[1:])
                    tokens, values = parse(nlp, sent)
                    if len(words[1:]) != len(tokens):
                        print(sent_id)
                    instance.update({'values': values})
                    write_conllu(fw, sent_id, words, values)
                except Exception as e:
                    print(e)
                    print('parsing sent_id = {} error'.format(sent_id))
                    pass
    elif args.format == 'flat':
        # sentence in flat file has not been pretokenized
        try:
            nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', download_method=None)
        except Exception as e:
            stanza.download('en')
            nlp = stanza.Pipeline(lang='en', processors='tokenize, pos, lemma, depparse', download_method=None)

        with open(args.input) as fr:
            lines = fr.readlines()
            fr.close()

        sent_id = 0
        with open(args.output, mode='w+') as fw:
            for line in lines:
                sent_id += 1
                sentence = line.strip('\n')
                tokens, values = parse(nlp, sentence)
                write_conllu(fw, sent_id, tokens, values)
    else:
        assert False




