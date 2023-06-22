# -*- coding: utf-8 -*-

import argparse

from supar.cmds.cmd import init
from supar.models.contrastive import ContrastiveGNNSemanticDependencyParser


def main(language, formalism, data_percent, number):
    parser = argparse.ArgumentParser(description='Create Contrastive GNN Semantic Dependency Parser.')
    parser.set_defaults(Parser=ContrastiveGNNSemanticDependencyParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    PROJ_BASE_PATH = '/mnt/SynGraphCL/'

    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'lemma', 'elmo', 'bert'], default=['tag', 'char', 'lemma'], nargs='*', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--pretrained_gnn_path', '-g', default='/mnt/SynGraphCL/output/pretrain/ggnn/model', type=str, help='path of contrastive pretrained gnn model')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'transformer', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default=PROJ_BASE_PATH + 'data/sdp/{}/{}/{}.train.{}.{}.conllu'.format(formalism, language, data_percent, language, formalism.lower()),
                           help='path to train file')
    subparser.add_argument('--syn_train', default=PROJ_BASE_PATH + 'data/syn_dep/{}/syn.{}.conllu'.format(language, language), help='path to train file')
    subparser.add_argument('--dev', default=PROJ_BASE_PATH + 'data/sdp/{}/{}/dev.{}.{}.conllu'.format(formalism, language, language, formalism.lower()), help='path to dev file')
    subparser.add_argument('--syn_dev', default=PROJ_BASE_PATH + 'data/syn_dep/{}/syn.dev.{}.{}.conllu'.format(language, language, formalism.lower()), help='path to dev file')
    subparser.add_argument('--test', default=PROJ_BASE_PATH + 'data/sdp/{}/{}/{}.id.{}.conllu'.format(formalism, language, language, formalism.lower()), help='path to test file')
    subparser.add_argument('--syn_test', default=PROJ_BASE_PATH + 'data/syn_dep/{}/syn.{}.id.conllu'.format(language, language), help='path to test file')

    # subparser.add_argument('--embed', default=False, help='file or embeddings available at `supar.utils.Embedding`')
    subparser.add_argument('--embed', default='/mnt/wordvec/english/glove.6B.100d.txt', help='file or embeddings available at `supar.utils.Embedding`')
    subparser.add_argument('--n-embed-proj', default=125, type=int, help='dimension of projected embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--eval_data', default=PROJ_BASE_PATH + 'data/sdp/{}/{}/{}.id.{}.conllu'.format(formalism, language, language, formalism.lower()), help='path to data file')
    subparser.add_argument('--syn_eval_data', default=PROJ_BASE_PATH + 'data/syn_dep/{}/syn.{}.id.conllu'.format(language, language), help='path to test file')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--pred_data', default=PROJ_BASE_PATH + 'data/syn_dep/stanza_parse.conllu', help='path to dataset')
    subparser.add_argument('--syn_pred_data', default=PROJ_BASE_PATH + 'data/syn_dep/stanza_parse.conllu', help='path to test file')
    subparser.add_argument('--pred', default=PROJ_BASE_PATH + 'data/pred.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')

    parser.add_argument('--path', '-p', default=PROJ_BASE_PATH + '/output/sdp/ggnn/{}/{}/{}/{}/model'.format(number, language, formalism.lower(), data_percent), help='path to model file')
    init(parser)


if __name__ == "__main__":
    # lang_formals = [('en', 'DM'), ('en', 'PAS'), ('en', 'PSD')]
    lang_formals = [('en', 'DM')]
    data_percents = [1, 10]
    for number in [1]:
        for lang_formal in lang_formals:
            language = lang_formal[0]
            formalism = lang_formal[1]
            for data_percent in data_percents:
                main(language, formalism, data_percent, number)
