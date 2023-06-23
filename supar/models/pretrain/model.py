# -*- coding: utf-8 -*-
import dill
import torch
import torch.nn as nn
from graph4nlp.pytorch.modules.graph_embedding_learning import GCN, GAT, GraphSAGE, GGNN

from supar.model import Model
import torch.nn.functional as F
from supar.utils import Config
from info_nce import InfoNCE

class ContrastivePretrainingGNN(Model):
    r"""
    The implementation of Contrastive Biaffine Semantic Dependency Parser.

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
    NAME = 'contrastive-gnn-pretrain'

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
                 **kwargs):
        super().__init__(**Config().update(locals()))
        if self.args.gnn == 'GCN':
            self.graph_encoder = GCN(
                num_layers=self.args.gnn_layer, input_size=self.args.n_encoder_hidden,
                hidden_size=self.args.n_gnn_hidden,
                output_size=self.args.n_gnn_out, feat_drop=self.args.gnn_dropout,
                direction_option=self.args.gnn_direction_option,
                activation=F.elu, allow_zero_in_degree=True, use_edge_weight=True)
        elif self.args.gnn == "GAT":
            self.graph_encoder = GAT(
                num_layers=self.args.gnn_layer, input_size=self.args.n_encoder_hidden, hidden_size=self.args.n_gnn_hidden,
                output_size=self.args.n_gnn_out, feat_drop=self.args.gnn_dropout, attn_drop=self.args.gnn_dropout,
                heads=self.args.n_gat_heads, direction_option=self.args.gnn_direction_option, activation=F.elu,
                allow_zero_in_degree=True)
        elif self.args.gnn == "GraphSage":
            self.graph_encoder = GraphSAGE(
                num_layers=self.args.gnn_layer, input_size=self.args.n_encoder_hidden,
                hidden_size=self.args.n_gnn_hidden,
                output_size=self.args.n_gnn_out, feat_drop=self.args.gnn_dropout,
                aggregator_type=self.args.graphsage_aggreagte_type,
                direction_option=self.args.gnn_direction_option, bias=True, norm=None,
                activation=F.relu, use_edge_weight=False
            )
        elif self.args.gnn == "GGNN":
            self.graph_encoder = GGNN(
                num_layers=self.args.gnn_layer, input_size=self.args.n_encoder_hidden,
                hidden_size=self.args.n_gnn_hidden,
                output_size=self.args.n_gnn_out, feat_drop=self.args.gnn_dropout,
                direction_option=self.args.gnn_direction_option,
                bias=True, use_edge_weight=False
            )

        else:
            raise RuntimeError("Unknown gnn type: {}".format(self.args.gnn))
        self.infoNCE = InfoNCE(negative_mode='paired')

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

        return z

    def loss(self, org_node_embedding, pos_node_embedding, neg_node_embeddings, mask):
        r"""
        compute contrastive loss
        :param org_node_embedding:
        :param pos_node_embedding:
        :param neg_node_embeddings:
        :param mask:
        :return:
        """

        org_embedding = torch.mean(org_node_embedding, dim=1)
        pos_embedding = torch.mean(pos_node_embedding, dim=1)
        neg_embedding_list = [torch.mean(neg_node_embedding, dim=1) for neg_node_embedding in neg_node_embeddings]
        neg_embeddings = torch.stack(neg_embedding_list, dim=1)
        loss = self.infoNCE(org_embedding, pos_embedding, neg_embeddings)

        return loss

    # def save(self):
    #     """
    #     save the pre-trained gnn encoder
    #     :return:
    #     """
    #     state_dict = {k: v.cpu() for k, v in self.graph_encoder.state_dict().items()}
    #     torch.save(state_dict, self.args.pretrained_gnn_path, pickle_module=dill)