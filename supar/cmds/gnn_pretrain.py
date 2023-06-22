# -*- coding: utf-8 -*-

import argparse

from supar.cmds.cmd import init
from supar.models.pretrain.pretrain import ContrastiveGNNPretrainer


def main():
    BASE_PATH = '/mnt/sda1_hd/atur/libin/projects/SynGraphCL/'
    parser = argparse.ArgumentParser(description='Contrastive Pretraining GNNs.')
    parser.set_defaults(Parser=ContrastiveGNNPretrainer)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'lemma', 'elmo', 'bert'], default=['tag', 'char', 'lemma'], nargs='*', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--pretrained_gnn_path', '-g', default='/mnt/sda1_hd/atur/libin/projects/SynGraphCL/output/pretrain/ggnn/model', type=str, help='path of contrastive pretrained gnn model')
    subparser.add_argument('--checkpoint', action='store_true', help='whether to load a checkpoint to restore training')
    subparser.add_argument('--encoder', choices=['lstm', 'transformer', 'bert'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--original', default=BASE_PATH + 'data/contrastive/original.conllu', help='path to train file')
    subparser.add_argument('--positive', default=BASE_PATH + 'data/contrastive/positive.conllu', help='path to positive samples file')
    subparser.add_argument('--negative', default=BASE_PATH + 'data/contrastive/negative.conllu', help='path to negative samples file')
    subparser.add_argument('--embed', default='/mnt/sda1_hd/atur/libin/projects/wordvec/english/glove.6B.100d.txt', help='file or embeddings available at `supar.utils.Embedding`')
    subparser.add_argument('--n-embed-proj', default=125, type=int, help='dimension of projected embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which BERT model to use')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    parser.add_argument('--path', '-p', default='/mnt/sda1_hd/atur/libin/projects/FewShotSDP/output/pretrain/ggnn/model', help='path to model file')

    init(parser)


if __name__ == "__main__":
    main()
