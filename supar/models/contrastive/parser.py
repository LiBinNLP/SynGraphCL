# -*- coding: utf-8 -*-
import copy
import os
from typing import Iterable, Union

import torch

from supar.models.contrastive import ContrastiveGNNSemanticDependencyModel
from supar.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, PAD, UNK
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.logging import get_logger
from supar.utils.metric import ChartMetric
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch, CoNLL


logger = get_logger(__name__)


class ContrastiveGNNSemanticDependencyParser(Parser):
    r"""
    The implementation of GNN Semantic Dependency Parser :cite:`li2022`.
    """

    NAME = 'contrastive-gnn-semantic-dependency'
    MODEL = ContrastiveGNNSemanticDependencyModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(
        self,
        train: Union[str, Iterable],
        dev: Union[str, Iterable],
        test: Union[str, Iterable],
        epochs: int = 150,
        patience: int = 30,
        batch_size: int = 4000,
        update_steps: int = 1,
        buckets: int = 32,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().train(**Config().update(locals()))

    def evaluate(
        self,
        eval_data: Union[str, Iterable],
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().evaluate(**Config().update(locals()))

    def predict(
        self,
        pred_data: Union[str, Iterable],
        pred: str = None,
        lang: str = None,
        prob: bool = False,
        batch_size: int = 5000,
        buckets: int = 8,
        workers: int = 0,
        amp: bool = False,
        cache: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        return super().predict(**Config().update(locals()))

    def train_step(self, batch: Batch) -> torch.Tensor:
        words, *feats, labels, syn_labels = batch
        mask = batch.mask
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0

        s_edge, s_label = self.model(words, feats, syn_labels)
        loss = self.model.loss(s_edge, s_label, labels, mask)
        return loss

    @torch.no_grad()
    def eval_step(self, batch: Batch) -> ChartMetric:
        words, *feats, labels, syn_labels = batch
        mask = batch.mask
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0
        adj = torch.where(syn_labels <= 0, 0.0, 1.0)

        s_edge, s_label = self.model(words, feats, adj)
        loss = self.model.loss(s_edge, s_label, labels, mask)
        label_preds = self.model.decode(s_edge, s_label)
        return ChartMetric(loss, label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))

    @torch.no_grad()
    def pred_step(self, batch: Batch) -> Batch:
        words, *feats, syn_labels = batch
        mask, lens = batch.mask, (batch.lens - 1).tolist()
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0

        with torch.autocast(self.device, enabled=self.args.amp):
            s_edge, s_label = self.model(words, feats, syn_labels)
        label_preds = self.model.decode(s_edge, s_label).masked_fill(~mask, -1)
        batch.labels = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None for i in row]
                                               for row in chart[1:i, :i].tolist()])
                        for i, chart in zip(lens, label_preds)]
        if self.args.prob:
            batch.probs = [prob[1:i, :i].cpu() for i, prob in zip(lens, s_edge.softmax(-1).unbind())]
        return batch

    @classmethod
    def build(cls, path, min_freq=7, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (Dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        os.makedirs(os.path.dirname(path) or './', exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.transform.FORM[0].embed).to(parser.device)
            return parser

        logger.info("Building the fields")
        WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
        TAG, CHAR, LEMMA, ELMO, BERT = None, None, None, None, None
        if args.encoder == 'bert':
            t = TransformerTokenizer(args.bert)
            WORD = SubwordField('words', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
            WORD.vocab = t.vocab
        else:
            WORD = Field('words', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=BOS)
            if 'char' in args.feat:
                CHAR = SubwordField('chars', pad=PAD, unk=UNK, bos=BOS, fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=PAD, unk=UNK, bos=BOS, lower=True)
            if 'elmo' in args.feat:
                from allennlp.modules.elmo import batch_to_ids
                ELMO = RawField('elmo')
                ELMO.compose = lambda x: batch_to_ids(x).to(WORD.device)
            if 'bert' in args.feat:
                t = TransformerTokenizer(args.bert)
                BERT = SubwordField('bert', pad=t.pad, unk=t.unk, bos=t.bos, fix_len=args.fix_len, tokenize=t)
                BERT.vocab = t.vocab
        LABEL = ChartField('labels', fn=CoNLL.get_labels)
        SYN_LABEL = ChartField('syn_labels', fn=CoNLL.get_labels)

        if True:
            state = torch.load(args.pretrained_gnn_path, map_location='cpu')
            transform = state['transform']
            transform.PDEPREL = SYN_LABEL
            train = Dataset(transform, args.train, **args)
            transform.PHEAD.build(train)
            WORD = transform.FORM[0]
            CHAR = transform.FORM[1]
            LEMMA = transform.LEMMA
            TAG = transform.POS
            args.update({
                'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
                'n_labels': len(transform.PHEAD.vocab),
                'n_tags': len(transform.POS.vocab) if TAG is not None else None,
                'n_chars': len(CHAR.vocab) if CHAR is not None else None,
                'char_pad_index': CHAR.pad_index if CHAR is not None else None,
                'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
                'bert_pad_index': BERT.pad_index if BERT is not None else None,
                'pad_index': WORD.pad_index,
                'unk_index': WORD.unk_index,
                'bos_index': WORD.bos_index
            })
        else:
            transform = CoNLL(FORM=(WORD, CHAR, ELMO, BERT), LEMMA=LEMMA, POS=TAG, PHEAD=LABEL, PDEPREL=SYN_LABEL)
            train = Dataset(transform, args.train, **args)
            if args.encoder != 'bert':
                WORD.build(train, args.min_freq, (Embedding.load(args.embed) if args.embed else None), lambda x: x / torch.std(x))
                if TAG is not None:
                    TAG.build(train)
                if CHAR is not None:
                    CHAR.build(train)
                if LEMMA is not None:
                    LEMMA.build(train)
            LABEL.build(train)
            args.update({
                'n_words': len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
                'n_labels': len(LABEL.vocab),
                'n_tags': len(TAG.vocab) if TAG is not None else None,
                'n_chars': len(CHAR.vocab) if CHAR is not None else None,
                'char_pad_index': CHAR.pad_index if CHAR is not None else None,
                'n_lemmas': len(LEMMA.vocab) if LEMMA is not None else None,
                'bert_pad_index': BERT.pad_index if BERT is not None else None,
                'pad_index': WORD.pad_index,
                'unk_index': WORD.unk_index,
                'bos_index': WORD.bos_index
            })
        logger.info(f"{transform}")
        syn_transform = copy.deepcopy(transform)
        cls.read_syn_tree(args=args, syn_transform=syn_transform)

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform, syn_transform)
        parser.model.to(parser.device)
        return parser

    @classmethod
    def read_syn_tree(cls, args, syn_transform):
        syn_train = Dataset(syn_transform, args.syn_train)
        syn_transform.PHEAD.build(syn_train)
        if syn_transform.POS is not None:
            syn_transform.POS.build(syn_train)


