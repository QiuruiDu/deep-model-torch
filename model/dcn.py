import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset

# https://blog.csdn.net/Talantfuck/article/details/124542291
# the implement of deep cross network model

# 编写模型
class DCN(nn.Module):
    def __init__(self, emb_dim :int, cross_layer_num :int, sparse_dim_dict :dict, dense_dim :int, hidden_dims=[16,8], dropouts=[0.5,0.5]):
        """
        :param emb_dim: embedding层要输出的维度
        :param cross_layer_num: 交叉网络的层数
        :param hidden_dims: 隐藏层的每一层输出的维度
        :param dropouts: dropout list, 每一层dropout多少
        :param sparse_dim_dict: 离散特征的维度字典,dict类型
        :param dense_dim: 连续特征的维度，是在外部拼接好的连续特征
        """
        super(DCN, self).__init__()
        self.emb_dim = emb_dim
        self.cross_layer_num = cross_layer_num
        self.hidden_dims = hidden_dims
        self.dropouts = dropouts
        self.sparse_dim_dict = sparse_dim_dict
        self.dense_dim = dense_dim
        self.embedding = None
        # embedding层
        self.embedding_layer = nn.ModuleDict()
        for feature_name in self.sparse_dim_dict.keys():
            self.embedding_layer[feature_name] = nn.Embedding(self.sparse_dim_dict[feature_name], self.emb_dim)

        input_dim = len(self.sparse_dim_dict.keys()) * emb_dim + dense_dim
        # cross层
        self.cross_layer = nn.ModuleList()
        for _ in range(cross_layer_num):
            self.cross_layer.append(nn.Linear(input_dim, 1))

        # deep层
        self.deep_layer = nn.ModuleList()
        hidden_dims = [input_dim] + hidden_dims
        self.deep_layer.append(nn.BatchNorm1d(input_dim))
        for i in range(len(hidden_dims)-1):
            self.deep_layer.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.deep_layer.append(nn.ReLU())
            self.deep_layer.append(nn.Dropout(self.dropouts[i]))
            self.deep_layer.append(nn.BatchNorm1d(hidden_dims[i+1]))
            # 有点问题，没有加激活函数

        # 输出层
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
            i += 1
        self.embedding = torch.cat(emb_arr, dim = 1)
        # print("embedding shape is ", self.embedding.shape)

    def cross_layer_func(self, dense_data):
        """
        get the cross network result
        :param dense_data: dense data
        :return:
        """
        # 对离散特征进行embedding
        emb = self.embedding
        cross_input = torch.cat([emb, dense_data], dim = 1)
        for i in range(self.cross_layer_num):
            emb_tmp = torch.bmm(torch.transpose(cross_input.unsqueeze(1), 1, 2),
                                cross_input.unsqueeze(1))
            # torch.bmm函数：矩阵相乘；torch.transpose函数：tensor矩阵求逆
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
        # print("--- cross result is ",cross_result.shape, ", deep result is ", deep_result.shape)
        output = torch.cat([cross_result, deep_result], dim=1)
        # print("the example of result before predict : ",output[0])
        # print("--- output result shape is ",output.shape)
        output = self.predict(output)
        # print("the example of predict output : ",output[0])
        output = nn.Sigmoid()(output)
        return output
