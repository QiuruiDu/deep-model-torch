import torch
from torch import nn


class DCN(nn.Module):
    def __init__(self, emb_dim :int, cross_layer_num :int, hidden_dims :list, dropouts :list, sparse_dim_dict :dict, dense_dim :int):
        """
        :param emb_dim: embedding output dim
        :param cross_layer_num: the number of layer in cross layer
        :param hidden_dims: the output dim of every hidden layer
        :param dropouts: dropout list
        :param sparse_dim_dict: the dimension dict of sparse data
        :param dense_dim: the dimension of dense data
        """
        super(DCN, self).__init__()
        self.emb_dim = emb_dim
        self.cross_layer_num = cross_layer_num
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.sparse_dim_dict = sparse_dim_dict
        self.dense_dim = dense_dim
        self.embedding = None
        # embedding
        self.embedding_layer = nn.ModuleDict()
        for feature_name in self.sparse_dim_dict.keys():
            self.embedding_layer[feature_name] = nn.Embedding(self.sparse_dim_dict[feature_name], self.emb_dim)

        input_dim = len(self.sparse_dim_dict.keys()) * emb_dim + dense_dim
        # cross
        self.cross_layer = nn.ModuleList()
        for _ in range(cross_layer_num):
            self.cross_layer.append(nn.Linear(input_dim, 1))

        # deep
        self.deep_layer = nn.ModuleList()
        hidden_dims = [input_dim] + hidden_dims
        self.deep_layer.append(nn.BatchNorm1d(input_dim))
        for i in range(len(hidden_dims)-1):
            self.deep_layer.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.deep_layer.append(nn.Dropout(self.dropouts[i]))
            self.deep_layer.append(nn.BatchNorm1d(hidden_dims[i+1]))

        # output
        self.predict = nn.Linear(hidden_dims[-1]+len(self.sparse_dim_dict.keys())*emb_dim+dense_dim, 1)

    def get_emb(self, sparse_data):
        """
        get the embedding result of the sparse feature
        :param sparse_data: sparse data
        :return: embedding data
        """
        emb_arr = []
        i = 0
        for feature_name in self.sparse_dim_dict.keys():
            emb_temp = self.embedding_layer[feature_name](sparse_data[:,i])
            emb_arr.append(emb_temp)
        self.embedding = torch.cat(emb_arr, dim = 1)

    def cross_layer_func(self, dense_data):
        """
        get the cross network result
        :param dense_data: dense data
        :return:
        """
        # embedding
        emb = self.embedding
        cross_input = torch.cat([emb, dense_data], dim = 1)
        for i in range(self.cross_layer_num):
            emb_tmp = torch.bmm(torch.transpose(cross_input.unsqueeze(1), 1, 2),
                                cross_input.unsqueeze(1))
            emb_tmp = self.cross_layer[i](emb_tmp)
            emb = emb_tmp.transpose(1, 2).squeeze(1) + cross_input
        return emb

    def deep_layer_func(self, dense_data):
        """
        get the deep network result
        :param dense_data: dense data
        :return: the result of deep layer
        """
        emb = self.embedding
        deep_input = torch.cat([emb, dense_data], dim = 1)
        for layer in self.deep_layer:
            deep_input = layer(deep_input)
        return deep_input

    def forward(self, sparse_data, dense_data):
        """
        the function when training model will use
        :param sparse_data: sparse data input
        :param dense_data: dense data input
        :return:
        """
        self.get_emb(sparse_data)
        cross_result = self.cross_layer_func(dense_data)
        deep_result = self.deep_layer_func(dense_data)
        output = torch.cat([cross_result, deep_result], dim=1)
        output = self.predict(output)
        output = nn.Sigmoid()(output)
        return output
