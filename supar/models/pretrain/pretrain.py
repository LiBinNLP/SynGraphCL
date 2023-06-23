# -*- coding: utf-8 -*-
import os
from datetime import timedelta, datetime

import dill
import torch

from supar import Parser
from supar.models.pretrain import ContrastivePretrainingGNN
import torch.distributed as dist
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import BOS, PAD, UNK
from supar.utils.data import ContrastiveDataset
from supar.utils.field import ChartField, Field, RawField, SubwordField
from supar.utils.fn import set_rng_state
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import ChartMetric, Metric
from supar.utils.optim import InverseSquareRootLR, LinearLR
from supar.utils.tokenizer import TransformerTokenizer
from supar.utils.transform import Batch, CoNLL, ContrastiveTransform
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.cuda.amp import GradScaler
from supar.utils.parallel import DistributedDataParallel as DDP, is_master, is_dist

logger = get_logger(__name__)


class ContrastiveGNNPretrainer(Parser):
    r"""
    The implementation of Contrastive GNN Pretraining .
    """
    NAME = 'contrastive-pretrain-gnn'
    MODEL = ContrastivePretrainingGNN

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(
            self,
            epochs: int = 100,
            patience: int = 20,
            batch_size: int = 1500,
            update_steps: int = 1,
            buckets: int = 32,
            workers: int = 0,
            amp: bool = False,
            cache: bool = False,
            verbose: bool = True,
            **kwargs):

        args = self.args.update(locals())

        "------------------------load original, positive and negative samples--------------------"
        original = Dataset(self.transform, args.original, **args).build(batch_size, buckets, True,
                                                                        dist.is_initialized())
        positive = Dataset(self.transform, args.positive, **args).build(batch_size, buckets, False,
                                                                        dist.is_initialized())
        negative = Dataset(self.transform, args.negative, **args).build(batch_size, buckets, False,
                                                                        dist.is_initialized())
        print(self.transform.flattened_fields)
        "change the transform to adapt it for contrastive learning"
        contrastive_transform = self.contrastive_transform_build()
        contrastive_data = ContrastiveDataset(contrastive_transform, original, positive, negative).build(batch_size, buckets,
                                                                                                         True, dist.is_initialized())
        loader, sampler = contrastive_data.loader, contrastive_data.loader.batch_sampler

        if args.encoder == 'lstm':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = ExponentialLR(self.optimizer, args.decay ** (1 / args.decay_steps))
        elif args.encoder == 'transformer':
            self.optimizer = Adam(self.model.parameters(), args.lr, (args.mu, args.nu), args.eps, args.weight_decay)
            self.scheduler = InverseSquareRootLR(self.optimizer, args.warmup_steps)
        else:
            # we found that Huggingface's AdamW is more robust and empirically better than the native implementation
            from transformers import AdamW
            steps = len(original.loader) * epochs // args.update_steps
            self.optimizer = AdamW(
                [{'params': p, 'lr': args.lr * (1 if n.startswith('encoder') else args.lr_rate)}
                 for n, p in self.model.named_parameters()],
                args.lr,
                (args.mu, args.nu),
                args.eps,
                args.weight_decay
            )
            self.scheduler = LinearLR(self.optimizer, int(steps * args.warmup), steps)
        self.scaler = GradScaler(enabled=args.amp)

        if dist.is_initialized():
            self.model = DDP(self.model,
                             device_ids=[args.local_rank],
                             find_unused_parameters=args.get('find_unused_parameters', True),
                             static_graph=args.get('static_graph', False))
            if args.amp:
                from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook
                self.model.register_comm_hook(dist.group.WORLD, fp16_compress_hook)

        self.step, self.epoch, self.best_e, self.patience, self.n_batches = 1, 1, 1, patience, len(loader)
        self.total_steps = self.n_batches * epochs // args.update_steps
        self.best_metric, self.elapsed = Metric(), timedelta()
        self.minimum_loss = 5
        if self.args.checkpoint:
            try:
                self.optimizer.load_state_dict(self.checkpoint_state_dict.pop('optimizer_state_dict'))
                self.scheduler.load_state_dict(self.checkpoint_state_dict.pop('scheduler_state_dict'))
                self.scaler.load_state_dict(self.checkpoint_state_dict.pop('scaler_state_dict'))
                set_rng_state(self.checkpoint_state_dict.pop('rng_state'))
                for k, v in self.checkpoint_state_dict.items():
                    setattr(self, k, v)
                sampler.set_epoch(self.epoch)
            except AttributeError:
                logger.warning("No checkpoint found. Try re-launching the training procedure instead")

        for epoch in range(self.epoch, args.epochs + 1):
            start = datetime.now()
            bar = progress_bar(loader)

            logger.info(f"Epoch {epoch} / {args.epochs}:")
            self.model.train()
            with self.join():
                # we should zero `step` as the number of batches in different processes is not necessarily equal
                self.step = 0
                for batch in bar:
                    with self.sync():
                        with torch.autocast(self.device, enabled=self.args.amp):
                            loss = self.train_step(batch)
                        self.current_loss = loss
                        self.backward(loss)
                    if self.sync_grad:
                        self.clip_grad_norm_(self.model.parameters(), self.args.clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(True)
                    bar.set_postfix_str(f"lr: {self.scheduler.get_last_lr()[0]:.9e} - loss: {loss:.9f}")
                    self.step += 1
                logger.info(f"{bar.postfix}")
            t = datetime.now() - start
            self.epoch += 1
            self.patience -= 1
            self.elapsed += t
            # 判断当前的loss是否变得更小，如果是，则将当前模型保存起来；否则，继续训练
            # 对比学习是子监督学习，没有dev set，只能用loss来评估模型好坏
            if self.current_loss < self.minimum_loss:
                self.best_e, self.patience, self.minimum_loss = epoch, patience, self.current_loss
                if is_master():
                    self.save()
                    # self.model.save()
                    # self.save_checkpoint(args.path)
                logger.info(f"{t}s elapsed (saved)\n")
            else:
                logger.info(f"{t}s elapsed\n")
            if self.patience < 1:
                break
        if is_dist():
            dist.barrier()

        best = self.load(**args)
        # only allow the master device to save models
        if is_master():
            self.save()
            # self.model.save()
            # best.save(args.path)

        logger.info(f"Epoch {self.best_e} saved")
        logger.info(f"{'minimum_loss:':5} {self.minimum_loss}")
        logger.info(f"{self.elapsed}s elapsed, {self.elapsed / epoch}s/epoch")

    def train_step(self, batch: Batch) -> torch.Tensor:
        r"""
        a training step of contrastive pre-training gnn phase
        :param batch: batch data
        :return:
        """
        org_words, org_feats, org_labels, pos_words, pos_feats, pos_labels, neg_words, neg_feats, neg_labels = batch
        mask = batch.mask
        mask = mask.unsqueeze(1) & mask.unsqueeze(2)
        mask[:, 0] = 0
        # encode original samples to get a original embedding
        org_embedding = self.model(org_words, org_feats, org_labels)
        # encode positive samples to get a positive embedding
        pos_embedding = self.model(pos_words, pos_feats, pos_labels)
        # encode negative samples to get multiple negative embeddings
        neg_embeddings = [self.model(neg_word, neg_feat, neg_label)
                          for neg_word, neg_feat, neg_label in zip(neg_words, neg_feats, neg_labels)]
        # minimize infoNCE loss for one original embedding, one positive embedding
        # and multiple negative embeddings to optimize the model
        loss = self.model.loss(org_embedding, pos_embedding, neg_embeddings, mask)

        return loss

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
        transform = CoNLL(FORM=(WORD, CHAR, ELMO, BERT), LEMMA=LEMMA, POS=TAG, PHEAD=LABEL)

        original = Dataset(transform, args.original, **args)
        if args.encoder != 'bert':
            WORD.build(original, args.min_freq, (Embedding.load(args.embed) if args.embed else None),
                       lambda x: x / torch.std(x))
            if TAG is not None:
                TAG.build(original)
            if CHAR is not None:
                CHAR.build(original)
            if LEMMA is not None:
                LEMMA.build(original)
        LABEL.build(original)
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

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(WORD.embed if hasattr(WORD, 'embed') else None)
        logger.info(f"{model}\n")

        parser = cls(args, model, transform)
        parser.model.to(parser.device)
        return parser


    @classmethod
    def contrastive_transform_build(cls):
        r"""
        transform contrastive data into a transform object
        :return:
        """
        contrastive_transform = ContrastiveTransform(Field('org_words'), Field('org_feats'), Field('org_labels'),
                                                     Field('pos_words'), Field('pos_feats'), Field('pos_labels'),
                                                     Field('neg_words'), Field('neg_feats'), Field('neg_labels'))
        return contrastive_transform

    def save(self):
        """
        save the pre-trained gnn encoder
        :return:
        """
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}
        gnn_state_dict = {k: v.cpu() for k, v in self.model.graph_encoder.state_dict().items()}
        pretrained = state_dict.pop('pretrained.weight', None)
        state = {'name': self.NAME,
                 'args': self.model.args,
                 'state_dict': state_dict,
                 'pretrained': pretrained,
                 'transform': self.transform}
        torch.save(state, self.args.pretrained_gnn_path, pickle_module=dill)
        torch.save(self.transform, self.args.pretrained_gnn_path+'.transform', pickle_module=dill)
        torch.save(gnn_state_dict, self.args.pretrained_gnn_path+'.graph_encoder', pickle_module=dill)
        torch.save(pretrained, self.args.pretrained_gnn_path+'.pretrained', pickle_module=dill)
