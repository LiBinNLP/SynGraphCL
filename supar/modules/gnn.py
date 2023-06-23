import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    """https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py"""
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type):
        # valid_aggregator_type = {'mean', 'gcn', 'pool', 'lstm'}
        super(GraphSAGE, self).__init__()
        from dgl.nn.pytorch.conv import SAGEConv

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        # input layer
        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(nhid, nclass, batch_norm=False))


    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)
        return x


def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class RelGAT(nn.Module):
    """
    Relation gat model, use the embedding of the edges to predict attention weight
    """

    def __init__(self, args, dep_rel_num, hidden_size=64, num_layers=2, dropout=0.25):
        super(RelGAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        # gat layer
        # relation embedding, careful initialization?
        self.dep_rel_embed = nn.Embedding(
            dep_rel_num, args.dep_relation_embed_dim)
        nn.init.xavier_uniform_(self.dep_rel_embed.weight)

        # map rel_emb to logits. Naive attention on relations
        layers = [
            nn.Linear(args.dep_relation_embed_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.fcs = nn.Sequential(*layers)

    def forward(self, adj, rel_adj, feature):
        denom = adj.sum(2).unsqueeze(2) + 1
        B, N = adj.size(0), adj.size(1)

        rel_adj_V = self.dep_rel_embed(
            rel_adj.view(B, -1))  # (batch_size, n*n, d)

        # gcn layer
        for l in range(self.num_layers):
            # relation based GAT, attention over relations

            if True:
                rel_adj_logits = self.fcs(rel_adj_V).squeeze(2)  # (batch_size, n*n)
            else:
                rel_adj_logits = self.A[l](rel_adj_V).squeeze(2)  # (batch_size, n*n)

            dmask = adj.view(B, -1)  # (batch_size, n*n)
            rel_adj_logits = F.softmax(
                mask_logits(rel_adj_logits, dmask), dim=1)
            rel_adj_logits = rel_adj_logits.view(
                *rel_adj.size())  # (batch_size, n, n)

            Ax = rel_adj_logits.bmm(feature)
            feature = self.dropout(Ax) if l < self.num_layers - 1 else Ax

        return feature


class GAT(nn.Module):
    """
    GAT module operated on graphs
    """

    def __init__(self, in_dim, hidden_size=64, mem_dim=300, num_layers=2, gat_dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.in_dim = in_dim
        self.dropout = nn.Dropout(gat_dropout)
        self.leakyrelu = nn.LeakyReLU(1e-2)

        # Standard GAT:attention over feature
        a_layers = [
            nn.Linear(2 * mem_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)]
        self.afcs = nn.Sequential(*a_layers)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(num_layers):
            input_dim = self.in_dim if layer == 0 else mem_dim
            self.W.append(nn.Linear(input_dim, mem_dim))
            # self.W.append(GCNLayer(input_dim, mem_dim, batch_norm=False))

    def forward(self, feature, adj):
        B, N = adj.size(0), adj.size(1)
        dmask = adj.view(B, -1)  # (batch_size, n*n)
        # gcn layer
        for l in range(self.num_layers):
            # Standard GAT:attention over feature
            #####################################
            h = self.W[l](feature)  # (B, N, D)
            a_input = torch.cat([h.repeat(1, 1, N).view(
                B, N * N, -1), h.repeat(1, N, 1)], dim=2)  # (B, N*N, 2*D)
            e = self.leakyrelu(self.afcs(a_input)).squeeze(2)  # (B, N*N)
            attention = F.softmax(mask_logits(e, dmask), dim=1)
            attention = attention.view(*adj.size())

            # original gat
            feature = attention.bmm(h)
            feature = self.dropout(feature) if l < self.num_layers - 1 else feature
            #####################################

        return feature