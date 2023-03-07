"""
    (*^▽^*)
    the implement of wide and deep model
    Author:  Richael
    Date:    2023/03
    Contact: richael0302@163.com
"""

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

class WideAndDeep(nn.Module):
    def __init__(self, sparse_dim_dict :dict, emb_dim :int, dense_dim :int, hidden_dims_list=[16,8], dropout_list=[0.5,0.5]):
        """
        the init function of wideAndDeep
        :param sparse_dim_dict: the dict of sparse feature dimension
        :param emb_dim: embedding output dimension
        :param dense_dim: dense feature dimension
        :param hidden_dims_list: a list of hidden layer output dimension
        :param dropout_list: the list of dropout rate in deep layer
        """
        super(WideAndDeep, self).__init__()
        self.sparse_dim_dict = sparse_dim_dict
        self.emb_dim = emb_dim
        self.dense_dim = dense_dim
        self.hidden_dims_list = hidden_dims_list
        self.dropout_list = dropout_list
        self.embedding = None
        # embedding
        self.embedding_layer = nn.ModuleDict()
        for feature_name in sparse_dim_dict.keys():
            self.embedding_layer[feature_name] = nn.Embedding(num_embeddings=sparse_dim_dict[feature_name], embedding_dim=self.emb_dim)

        # wide
        self.wide_layer = nn.Linear(in_features = len(self.sparse_dim_dict.keys())*self.emb_dim, out_features=1)

        # deep
        self.deep_layer = nn.ModuleList()
        input_dim = len(self.sparse_dim_dict.keys())*self.emb_dim + dense_dim
        hidden_dims_list = [input_dim] + hidden_dims_list
        self.deep_layer.append(nn.BatchNorm1d(num_features=input_dim))
        for i in range(len(hidden_dims_list)-1):
            self.deep_layer.append(nn.Linear(in_features = hidden_dims_list[i], out_features = hidden_dims_list[i+1]))
            self.deep_layer.append(nn.ReLU())
            self.deep_layer.append(nn.Dropout(p=self.dropout_list[i]))
            self.deep_layer.append(nn.BatchNorm1d(num_features=hidden_dims_list[i+1]))
        # concat and predict
        self.predict = nn.Linear(in_features=hidden_dims_list[-1]+1, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def get_emb(self, sparse_feature: TensorDataset):
        """
        get the embedding data based on the sparse input
        :param sparse_feature:
        :return:
        """
        emb_arr = []
        i = 0
        for feature_name in self.sparse_dim_dict.keys():
            temp_emb = self.embedding_layer[feature_name](sparse_feature[:,i])
            emb_arr.append(temp_emb)
            i += 1
        self.embedding = torch.cat(emb_arr, dim = 1)

    def deep_func(self, dense_feature):
        """
        get the result of deep model based on model
        :param dense_feature: dense data input
        :return:
        """
        deep_input = torch.cat([self.embedding, dense_feature], dim = 1)
        for layer in self.deep_layer:
            deep_input = layer(deep_input)
        return deep_input

    def forward(self, sparse_feature, dense_feature):
        """
        the forward function
        :param sparse_feature: sparse data input
        :param dense_feature: dense data input
        :return:
        """
        # get embedding
        self.get_emb(sparse_feature)
        # dense
        deep_result = self.deep_func(dense_feature)
        # wide
        wide_result = self.wide_layer(self.embedding)
        output = torch.cat([wide_result, deep_result], dim=1)
        output = self.predict(output)
        output = self.sigmoid(output)
        return output

