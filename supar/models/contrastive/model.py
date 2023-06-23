# -*- coding: utf-8 -*-
import copy
import os

import torch
import torch.nn as nn

from supar.model import Model
import torch.nn.functional as F

from supar.modules import MLP, Biaffine
from supar.utils import Config, Embedding
from graph4nlp.pytorch.modules.graph_embedding_learning import GCN, GAT, GGNN, GraphSAGE

class ContrastiveGNNSemanticDependencyModel(Model):
    r"""
    The implementation of Contrastive GNN Semantic Dependency Parser.

    Args:
        n_words (int):
            The size of the word vocabulary.
        n_labels (int):
            The number of labels in the treebank.
        n_tags (int):
            The number of POS tags, required if POS tag embeddings are used. Default: ``None``.
        n_chars (int):
            The number of characters, required if character-level representations are used. Default: ``None``.
        n_lemmas (int):
            The number of lemmas, required if lemma embeddings are used. Default: ``None``.
        encoder (str):
            Encoder to use.
            ``'lstm'``: BiLSTM encoder.
            ``'bert'``: BERT-like pretrained language model (for finetuning), e.g., ``'bert-base-cased'``.
            Default: ``'lstm'``.
        feat (List[str]):
            Additional features to use, required if ``encoder='lstm'``.
            ``'tag'``: POS tag embeddings.
            ``'char'``: Character-level representations extracted by CharLSTM.
            ``'lemma'``: Lemma embeddings.
            ``'bert'``: BERT representations, other pretrained language models like RoBERTa are also feasible.
            Default: [ ``'tag'``, ``'char'``, ``'lemma'``].
        n_embed (int):
            The size of word embeddings. Default: 100.
        n_pretrained (int):
            The size of pretrained word representations. Default: 125.
        n_feat_embed (int):
            The size of feature representations. Default: 100.
        n_char_embed (int):
            The size of character embeddings serving as inputs of CharLSTM, required if using CharLSTM. Default: 50.
        n_char_hidden (int):
            The size of hidden states of CharLSTM, required if using CharLSTM. Default: 100.
        char_pad_index (int):
            The index of the padding token in the character vocabulary, required if using CharLSTM. Default: 0.
        elmo (str):
            Name of the pretrained ELMo registered in `ELMoEmbedding.OPTION`. Default: ``'original_5b'``.
        elmo_bos_eos (Tuple[bool]):
            A tuple of two boolean values indicating whether to keep start/end boundaries of elmo outputs.
            Default: ``(True, False)``.
        bert (str):
            Specifies which kind of language model to use, e.g., ``'bert-base-cased'``.
            This is required if ``encoder='bert'`` or using BERT features. The full list can be found in `transformers`_.
            Default: ``None``.
        n_bert_layers (int):
            Specifies how many last layers to use, required if ``encoder='bert'`` or using BERT features.
            The final outputs would be weighted sum of the hidden states of these layers.
            Default: 4.
        mix_dropout (float):
            The dropout ratio of BERT layers, required if ``encoder='bert'`` or using BERT features. Default: .0.
        bert_pooling (str):
            Pooling way to get token embeddings.
            ``first``: take the first subtoken. ``last``: take the last subtoken. ``mean``: take a mean over all.
            Default: ``mean``.
        bert_pad_index (int):
            The index of the padding token in BERT vocabulary, required if ``encoder='bert'`` or using BERT features.
            Default: 0.
        finetune (bool):
            If ``False``, freezes all parameters, required if using pretrained layers. Default: ``False``.
        n_plm_embed (int):
            The size of PLM embeddings. If 0, uses the size of the pretrained embedding model. Default: 0.
        embed_dropout (float):
            The dropout ratio of input embeddings. Default: .2.
        n_encoder_hidden (int):
            The size of encoder hidden states. Default: 1200.
        n_encoder_layers (int):
            The number of encoder layers. Default: 3.
        encoder_dropout (float):
            The dropout ratio of encoder layer. Default: .33.
        n_edge_mlp (int):
            Edge MLP size. Default: 600.
        n_label_mlp  (int):
            Label MLP size. Default: 600.
        edge_mlp_dropout (float):
            The dropout ratio of edge MLP layers. Default: .25.
        label_mlp_dropout (float):
            The dropout ratio of label MLP layers. Default: .33.
        interpolation (int):
            Constant to even out the label/edge loss. Default: .1.
        pad_index (int):
            The index of the padding token in the word vocabulary. Default: 0.
        unk_index (int):
            The index of the unknown token in the word vocabulary. Default: 1.

    .. _transformers:
        https://github.com/huggingface/transformers
    """

    def __init__(self,
                 n_words,
                 n_labels,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 encoder='lstm',
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 elmo='original_5b',
                 elmo_bos_eos=(True, False),
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 finetune=False,
                 n_plm_embed=0,
                 embed_dropout=.2,
                 n_encoder_hidden=1200,
                 n_encoder_layers=3,
                 encoder_dropout=.33,
                 n_edge_mlp=600,
                 n_label_mlp=600,
                 edge_mlp_dropout=.25,
                 label_mlp_dropout=.33,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 gnn_layer=3,
                 n_gat_heads=3,
                 gnn_direction_option=None,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        # super().__init__()
        self.args = Config().update(locals())
        if self.args.gnn == 'GCN':
            self.graph_encoder = GCN(
                num_layers=self.args.gnn_layer, input_size=self.args.n_encoder_hidden, hidden_size=self.args.n_gnn_hidden,
                output_size=self.args.n_gnn_out, feat_drop=self.args.gnn_dropout, direction_option=self.args.gnn_direction_option,
                activation=F.elu, allow_zero_in_degree=True)
        elif self.args.gnn == "GAT":
            self.graph_encoder = GAT(
                num_layers=self.args.gnn_layer, input_size=self.args.n_encoder_hidden, hidden_size=self.args.n_gnn_hidden,
                output_size=self.args.n_gnn_out, feat_drop=self.args.gnn_dropout, attn_drop=self.args.gnn_dropout,
                heads=self.args.n_gat_heads, direction_option=self.args.gnn_direction_option, activation=F.elu,
                allow_zero_in_degree=True)
        elif self.args.gnn == "GraphSage":
            self.graph_encoder = GraphSAGE(
                num_layers=self.args.gnn_layer, input_size=self.args.n_encoder_hidden, hidden_size=self.args.n_gnn_hidden,
                output_size=self.args.n_gnn_out, feat_drop=self.args.gnn_dropout, aggregator_type=self.args.graphsage_aggreagte_type,
                direction_option=self.args.gnn_direction_option, bias=True, norm=None,
                activation=F.relu, use_edge_weight=False
            )
        elif self.args.gnn == "GGNN":
            self.graph_encoder = GGNN(
                num_layers=self.args.gnn_layer, input_size=self.args.n_encoder_hidden, hidden_size=self.args.n_gnn_hidden,
                output_size=self.args.n_gnn_out, feat_drop=self.args.gnn_dropout, direction_option=self.args.gnn_direction_option,
                bias=True, use_edge_weight=False,
            )
        else:
            raise RuntimeError("Unknown gnn type: {}".format(self.args.gnn))

        self.load_pretrained_encoder()
        self.edge_mlp_d = MLP(n_in=self.args.n_encoder_hidden+self.args.n_gnn_out, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.edge_mlp_h = MLP(n_in=self.args.n_encoder_hidden+self.args.n_gnn_out, n_out=n_edge_mlp, dropout=edge_mlp_dropout, activation=False)
        self.label_mlp_d = MLP(n_in=self.args.n_encoder_hidden+self.args.n_gnn_out, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)
        self.label_mlp_h = MLP(n_in=self.args.n_encoder_hidden+self.args.n_gnn_out, n_out=n_label_mlp, dropout=label_mlp_dropout, activation=False)

        self.edge_attn = Biaffine(n_in=n_edge_mlp, n_out=2, bias_x=True, bias_y=True)
        self.label_attn = Biaffine(n_in=n_label_mlp, n_out=n_labels, bias_x=True, bias_y=True)
        self.criterion = nn.CrossEntropyLoss()

    def load_pretrained_encoder(self):
        r"""
        Loads a pretrained model of gnn encoder.

        Args:
            path (str):
                - a string with the shortcut name of a pretrained model defined in ``supar.MODEL``
                  to load from cache or download, e.g., ``'biaffine-dep-en'``.
                - a local path to a pretrained model, e.g., ``./<path>/model``.
        """
        assert os.path.exists(self.args.pretrained_gnn_path+'.graph_encoder')
        state = torch.load(self.args.pretrained_gnn_path+'.graph_encoder', map_location='cpu')
        self.graph_encoder.load_state_dict(state, False)


    def load_pretrained(self, embed=None):
        if embed is not None:
            self.pretrained = nn.Embedding.from_pretrained(embed)
            if embed.shape[1] != self.args.n_pretrained:
                self.embed_proj = nn.Linear(embed.shape[1], self.args.n_pretrained)
        return self

    def forward(self, words, feats=None, adjs=None):
        r"""
        Args:
            words (~torch.LongTensor): ``[batch_size, seq_len]``.
                Word indices.
            feats (List[~torch.LongTensor]):
                A list of feat indices.
                The size is either ``[batch_size, seq_len, fix_len]`` if ``feat`` is ``'char'`` or ``'bert'``,
                or ``[batch_size, seq_len]`` otherwise.
                Default: ``None``.
            adjs (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                adjacency matrix of a batch graphs

        Returns:
            ~torch.Tensor, ~torch.Tensor:
                The first tensor of shape ``[batch_size, seq_len, seq_len, 2]`` holds scores of all possible edges.
                The second of shape ``[batch_size, seq_len, seq_len, n_labels]`` holds
                scores of all possible labels on each edge.
        """
        # encode context embeddings with text encoder
        context_embedding = self.encode(words, feats)

        # convert a batch of syn_labels into a batch_graph object, to construct a large graph for gnn computing
        large_graph = self.batch_graphs_to_large_graph(context_embedding, adjs)
        # encode node embeddings with gnn encoder
        large_graph = self.graph_encoder(large_graph)
        # convert a large graph into a batch of graphs
        node_embedding = self.large_graph_to_batch_graphs(large_graph=large_graph)

        z = torch.cat((context_embedding, node_embedding), -1)

        # z = self.TextGNNEncoder(words, feats, adjs)

        edge_d = self.edge_mlp_d(z)
        edge_h = self.edge_mlp_h(z)
        label_d = self.label_mlp_d(z)
        label_h = self.label_mlp_h(z)

        # [batch_size, seq_len, seq_len, 2]
        s_edge = self.edge_attn(edge_d, edge_h).permute(0, 2, 3, 1)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_d, label_h).permute(0, 2, 3, 1)

        return s_edge, s_label

    def loss(self, s_edge, s_label, labels, mask):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.
            labels (~torch.LongTensor): ``[batch_size, seq_len, seq_len]``.
                The tensor of gold-standard labels.
            mask (~torch.BoolTensor): ``[batch_size, seq_len]``.
                The mask for covering the unpadded tokens.

        Returns:
            ~torch.Tensor:
                The training loss.
        """

        edge_mask = labels.ge(0) & mask
        edge_loss = self.criterion(s_edge[mask], edge_mask[mask].long())
        label_loss = self.criterion(s_label[edge_mask], labels[edge_mask])
        return self.args.interpolation * label_loss + (1 - self.args.interpolation) * edge_loss

    def decode(self, s_edge, s_label):
        r"""
        Args:
            s_edge (~torch.Tensor): ``[batch_size, seq_len, seq_len, 2]``.
                Scores of all possible edges.
            s_label (~torch.Tensor): ``[batch_size, seq_len, seq_len, n_labels]``.
                Scores of all possible labels on each edge.

        Returns:
            ~torch.LongTensor:
                Predicted labels of shape ``[batch_size, seq_len, seq_len]``.
        """

        return s_label.argmax(-1).masked_fill_(s_edge.argmax(-1).lt(1), -1)
