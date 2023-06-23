# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import queue
import shutil
import tempfile
import threading
from contextlib import contextmanager
from typing import Dict, Iterable, List, Union

import pathos.multiprocessing as mp
import torch
import torch.distributed as dist
from supar.utils.common import INF
from supar.utils.fn import binarize, debinarize, kmeans
from supar.utils.logging import get_logger, progress_bar
from supar.utils.parallel import is_dist, is_master
from supar.utils.transform import Batch, Transform
from torch.distributions.utils import lazy_property

logger = get_logger(__name__)


class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self,
                 transform: Transform,
                 original_data: Dataset = None,
                 positive_data: Dataset = None,
                 negative_data: Dataset = None
                 ):
        self.original_data = original_data
        self.positive_data = positive_data
        self.negative_data = negative_data
        self.transform = transform
        self.data = []

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_original={len(self.original_data)}"
        s += f"n_positive={len(self.positive_data)}"
        s += f"n_negative={len(self.negative_data)}"
        if hasattr(self, 'loader'):
            s += f", n_batches={len(self.loader)}"
        if hasattr(self, 'buckets'):
            s += f", n_buckets={len(self.buckets)}"
        s += ")"
        return s

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @lazy_property
    def sizes(self):
        return [s.size for s in self.data]

    def build(
        self,
        batch_size: int,
        n_buckets: int = 1,
        shuffle: bool = False,
        distributed: bool = False,
        n_workers: int = 0,
        pin_memory: bool = False,
    ) -> ContrastiveDataset:
        '''
        create contrastive dataset
        :param batch_size:
        :param n_buckets:
        :param shuffle:
        :param distributed:
        :param n_workers:
        :param pin_memory: whether need to lock data in CUDA GPU memory
        :return:
        '''
        # load positive data. One original sample has one positive sample
        syn_positive_dict = {}
        for sentence in self.positive_data.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            syn_positive_dict.update({sent_id: sentence})

        # load negative data. One original sample may has multiple negative samples
        syn_negative_dict = {}
        for sentence in self.negative_data.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            if sent_id in syn_negative_dict:
                syn_negative_dict[sent_id].append(sentence)
            else:
                syn_negative_dict.update({sent_id: [sentence]})

        for sentence in self.original_data.sentences:
            annotations = sentence.annotations
            sent_id = annotations[-2]
            # rename original words, feats, labels
            sentence.fields.update({'org_words': sentence.fields['words']})
            sentence.fields.update({'org_labels': sentence.fields['labels']})
            org_feats = []
            if 'chars' in sentence.fields:
                org_feats.append(sentence.fields['chars'])
            if 'lemmas' in sentence.fields:
                org_feats.append(sentence.fields['lemmas'])
            if 'tags' in sentence.fields:
                org_feats.append(sentence.fields['tags'])
            sentence.fields.update({'org_feats': org_feats})

            # merge features of positive sample into fields.feats
            if sent_id not in syn_positive_dict or sent_id not in syn_negative_dict:
                continue
            positive = syn_positive_dict[sent_id]
            negatives = syn_negative_dict[sent_id]
            pos_words = positive.fields['words']
            pos_labels = positive.fields['labels']
            sentence.fields.update({'pos_words': pos_words})
            sentence.fields.update({'pos_labels': pos_labels})
            pos_feats = []
            if 'chars' in positive.fields:
                pos_feats.append(positive.fields['chars'])
            if 'lemmas' in positive.fields:
                pos_feats.append(positive.fields['lemmas'])
            if 'tags' in positive.fields:
                pos_feats.append(positive.fields['tags'])
            sentence.fields.update({'pos_feats': pos_feats})

            # merge features of original sample into fields.feats
            neg_words = [negative.fields['words'] for negative in negatives]
            neg_labels = [negative.fields['labels'] for negative in negatives]
            sentence.fields.update({'neg_words': neg_words})
            sentence.fields.update({'neg_labels': neg_labels})
            neg_feats = []
            for negative in negatives:
                neg_feat = []
                if 'chars' in negative.fields:
                    neg_feat.append(negative.fields['chars'])
                if 'lemmas' in negative.fields:
                    neg_feat.append(negative.fields['lemmas'])
                if 'tags' in negative.fields:
                    neg_feat.append(negative.fields['tags'])
                neg_feats.append(neg_feat)
            sentence.fields.update({'neg_feats': neg_feats})

            self.data.append(sentence)

        # NOTE: the final bucket count is roughly equal to n_buckets
        self.buckets = dict(zip(*kmeans(self.sizes, n_buckets)))
        self.loader = DataLoader(transform=self.transform,
                                 dataset=self,
                                 batch_sampler=Sampler(self.buckets, batch_size, shuffle, distributed),
                                 num_workers=n_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=pin_memory)
        return self

class Dataset(torch.utils.data.Dataset):
    r"""
    Dataset that is compatible with :class:`torch.utils.data.Dataset`, serving as a wrapper for manipulating all data fields
    with the operating behaviours defined in :class:`~supar.utils.transform.Transform`.
    The data fields of all the instantiated sentences can be accessed as an attribute of the dataset.

    Args:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform` or its derivations.
            The instance holds a series of loading and processing behaviours with regard to the specific data format.
        data (Union[str, Iterable]):
            A filename or a list of instances that will be passed into :meth:`transform.load`.
        cache (bool):
            If ``True``, tries to use the previously cached binarized data for fast loading.
            In this way, sentences are loaded on-the-fly according to the meta data.
            If ``False``, all sentences will be directly loaded into the memory.
            Default: ``False``.
        binarize (bool):
            If ``True``, binarizes the dataset once building it. Only works if ``cache=True``. Default: ``False``.
        bin (str):
            Path for saving binarized files, required if ``cache=True``. Default: ``None``.
        max_len (int):
            Sentences exceeding the length will be discarded. Default: ``None``.
        kwargs (Dict):
            Together with `data`, kwargs will be passed into :meth:`transform.load` to control the loading behaviour.

    Attributes:
        transform (Transform):
            An instance of :class:`~supar.utils.transform.Transform`.
        sentences (List[Sentence]):
            A list of sentences loaded from the data.
            Each sentence includes fields obeying the data format defined in ``transform``.
            If ``cache=True``, each is a pointer to the sentence stored in the cache file.
    """

    def __init__(
        self,
        transform: Transform,
        data: Union[str, Iterable],
        cache: bool = False,
        binarize: bool = False,
        bin: str = None,
        max_len: int = None,
        **kwargs
    ) -> Dataset:
        super(Dataset, self).__init__()

        self.transform = transform
        self.data = data
        self.cache = cache
        self.binarize = binarize
        self.bin = bin
        self.max_len = max_len or INF
        self.kwargs = kwargs

        if cache:
            if not isinstance(data, str) or not os.path.exists(data):
                raise FileNotFoundError("Only files are allowed for binarization, but not found")
            if self.bin is None:
                self.fbin = data + '.pt'
            else:
                os.makedirs(self.bin, exist_ok=True)
                self.fbin = os.path.join(self.bin, os.path.split(data)[1]) + '.pt'
            if not self.binarize and os.path.exists(self.fbin):
                try:
                    self.sentences = debinarize(self.fbin, meta=True)['sentences']
                except Exception:
                    raise RuntimeError(f"Error found while debinarizing {self.fbin}, which may have been corrupted. "
                                       "Try re-binarizing it first")
        else:
            self.sentences = list(transform.load(data, **kwargs))

    def __repr__(self):
        s = f"{self.__class__.__name__}("
        s += f"n_sentences={len(self.sentences)}"
        if hasattr(self, 'loader'):
            s += f", n_batches={len(self.loader)}"
        if hasattr(self, 'buckets'):
            s += f", n_buckets={len(self.buckets)}"
        if self.cache:
            s += f", cache={self.cache}"
        if self.binarize:
            s += f", binarize={self.binarize}"
        if self.max_len < INF:
            s += f", max_len={self.max_len}"
        s += ")"
        return s

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return debinarize(self.fbin, self.sentences[index]) if self.cache else self.sentences[index]

    def __getattr__(self, name):
        if name not in {f.name for f in self.transform.flattened_fields}:
            raise AttributeError
        if self.cache:
            if os.path.exists(self.fbin) and not self.binarize:
                sentences = self
            else:
                sentences = self.transform.load(self.data, **self.kwargs)
            return (getattr(sentence, name) for sentence in sentences)
        return [getattr(sentence, name) for sentence in self.sentences]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    @lazy_property
    def sizes(self):
        if not self.cache:
            return [s.size for s in self.sentences]
        return debinarize(self.fbin, 'sizes')

    def build(
        self,
        batch_size: int,
        n_buckets: int = 1,
        shuffle: bool = False,
        distributed: bool = False,
        n_workers: int = 0,
        pin_memory: bool = True,
        chunk_size: int = 1000,
    ) -> Dataset:
        # numericalize all fields
        if not self.cache:
            self.sentences = [i for i in self.transform(self.sentences) if len(i) < self.max_len]
        else:
            # if not forced to do binarization and the binarized file already exists, directly load the meta file
            if os.path.exists(self.fbin) and not self.binarize:
                self.sentences = debinarize(self.fbin, meta=True)['sentences']
            else:
                @contextmanager
                def cache(sentences):
                    ftemp = tempfile.mkdtemp()
                    fs = os.path.join(ftemp, 'sentences')
                    fb = os.path.join(ftemp, os.path.basename(self.fbin))
                    global global_transform
                    global_transform = self.transform
                    sentences = binarize({'sentences': progress_bar(sentences)}, fs)[1]['sentences']
                    try:
                        yield ((sentences[s:s+chunk_size], fs, f"{fb}.{i}", self.max_len)
                               for i, s in enumerate(range(0, len(sentences), chunk_size)))
                    finally:
                        del global_transform
                        shutil.rmtree(ftemp)

                def numericalize(sentences, fs, fb, max_len):
                    sentences = global_transform((debinarize(fs, sentence) for sentence in sentences))
                    sentences = [i for i in sentences if len(i) < max_len]
                    return binarize({'sentences': sentences, 'sizes': [sentence.size for sentence in sentences]}, fb)[0]

                logger.info(f"Seeking to cache the data to {self.fbin} first")
                # numericalize the fields of each sentence
                if is_master():
                    with cache(self.transform.load(self.data, **self.kwargs)) as chunks, mp.Pool(32) as pool:
                        results = [pool.apply_async(numericalize, chunk) for chunk in chunks]
                        self.sentences = binarize((r.get() for r in results), self.fbin, merge=True)[1]['sentences']
                if is_dist():
                    dist.barrier()
                if not is_master():
                    self.sentences = debinarize(self.fbin, meta=True)['sentences']
        # NOTE: the final bucket count is roughly equal to n_buckets
        self.buckets = dict(zip(*kmeans(self.sizes, n_buckets)))
        self.loader = DataLoader(transform=self.transform,
                                 dataset=self,
                                 batch_sampler=Sampler(self.buckets, batch_size, shuffle, distributed),
                                 num_workers=n_workers,
                                 collate_fn=collate_fn,
                                 pin_memory=pin_memory)
        return self


class Sampler(torch.utils.data.Sampler):
    r"""
    Sampler that supports for bucketization and token-level batchification.

    Args:
        buckets (Dict):
            A dict that maps each centroid to indices of clustered sentences.
            The centroid corresponds to the average length of all sentences in the bucket.
        batch_size (int):
            Token-level batch size. The resulting batch contains roughly the same number of tokens as ``batch_size``.
        shuffle (bool):
            If ``True``, the sampler will shuffle both buckets and samples in each bucket. Default: ``False``.
        distributed (bool):
            If ``True``, the sampler will be used in conjunction with :class:`torch.nn.parallel.DistributedDataParallel`
            that restricts data loading to a subset of the dataset.
            Default: ``False``.
    """

    def __init__(
        self,
        buckets: Dict[float, List],
        batch_size: int,
        shuffle: bool = False,
        distributed: bool = False
    ) -> Sampler:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sizes, self.buckets = zip(*[(size, bucket) for size, bucket in buckets.items()])
        # number of batches in each bucket, clipped by range [1, len(bucket)]
        self.n_batches = [min(len(bucket), max(round(size * len(bucket) / batch_size), 1))
                          for size, bucket in zip(self.sizes, self.buckets)]
        self.rank, self.n_replicas, self.n_samples = 0, 1, sum(self.n_batches)
        if distributed:
            self.rank = dist.get_rank()
            self.n_replicas = dist.get_world_size()
            self.n_samples = sum(self.n_batches) // self.n_replicas + int(self.rank < sum(self.n_batches) % self.n_replicas)
        self.epoch = 1

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        total, batches = 0, []
        # if `shuffle=True`, shuffle both the buckets and samples in each bucket
        # for distributed training, make sure each process generates the same random sequence at each epoch
        range_fn = torch.arange if not self.shuffle else lambda x: torch.randperm(x, generator=g)
        for i, bucket in enumerate(self.buckets):
            split_sizes = [(len(bucket) - j - 1) // self.n_batches[i] + 1 for j in range(self.n_batches[i])]
            # DON'T use `torch.chunk` which may return wrong number of batches
            for batch in range_fn(len(bucket)).split(split_sizes):
                if total % self.n_replicas == self.rank:
                    batches.append([bucket[j] for j in batch.tolist()])
                total += 1
        self.epoch += 1
        return iter(batches[i] for i in range_fn(len(batches)).tolist())

    def __len__(self):
        return self.n_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class DataLoader(torch.utils.data.DataLoader):

    r"""
    A wrapper for native :class:`torch.utils.data.DataLoader` enhanced with a data prefetcher.
    See http://stackoverflow.com/questions/7323664/python-generator-pre-fetch and
    https://github.com/NVIDIA/apex/issues/304.
    """

    def __init__(self, transform, **kwargs):
        super().__init__(**kwargs)

        self.transform = transform

    def __iter__(self):
        return PrefetchGenerator(self.transform, super().__iter__())


class PrefetchGenerator(threading.Thread):

    def __init__(self, transform, loader, prefetch=1):
        threading.Thread.__init__(self)

        self.transform = transform

        self.queue = queue.Queue(prefetch)
        self.loader = loader
        self.daemon = True
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()

        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        if hasattr(self, 'stream'):
            torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.queue.get()
        if batch is None:
            raise StopIteration
        return batch

    def run(self):
        # `torch.cuda.current_device` is thread local
        # see https://github.com/pytorch/pytorch/issues/56588
        if is_dist() and torch.cuda.is_available():
            torch.cuda.set_device(dist.get_rank())
        if hasattr(self, 'stream'):
            with torch.cuda.stream(self.stream):
                for batch in self.loader:
                    self.queue.put(batch.compose(self.transform))
        else:
            for batch in self.loader:
                self.queue.put(batch.compose(self.transform))
        self.queue.put(None)


def collate_fn(x):
    return Batch(x)
