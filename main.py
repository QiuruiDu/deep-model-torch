import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model.dcn import DCN
from model.wd import WideAndDeep
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report
import argparse

data_path = './data/Telco-Customer-Churn-Clear.csv'


def get_parser():
    parser = argparse.ArgumentParser(description="main function")
    parser.add_argument('--model_type', default='dcn')
    parser.add_argument('--batch_size', default=50)
    parser.add_argument('--embedding_dim', default=4)
    parser.add_argument('--cross_layer_num', default=2)
    parser.add_argument('--early_stopping', default=9)
    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    model_type = args.model_type
    batch_size = int(args.batch_size)
    embedding_dim = int(args.embedding_dim)
    cross_layer_num = int(args.cross_layer_num)
    early_stopping = int(args.early_stopping)
    train_dict, test_dict = get_tensor_data()
    train_dataset = TensorDataset(train_dict['x_sparse'], train_dict['x_dense'], train_dict['y'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sparse_dim_dict = train_dict['sparse_dict']
    dense_dim = len(train_dict['dense_col'])

    if model_type == 'dcn':
        model = DCN(emb_dim=embedding_dim, cross_layer_num=cross_layer_num, sparse_dim_dict=sparse_dim_dict, dense_dim=dense_dim)
    elif model_type == 'wd':
        model = WideAndDeep(sparse_dim_dict=sparse_dim_dict, emb_dim=embedding_dim, dense_dim=dense_dim)
    else:
        print("No this model!")
        return
    model = train(train_loader, early_stopping, model)
    predict(model, test_dict['x_sparse'], test_dict['x_dense'], test_dict['y'])


def get_tensor_data(test_size = 0.15):
    """
    :param test_size: the split size of test data
    :return:train data dict and test data dict
    """
    df = pd.read_csv(data_path)
    sparse_col = [c for c in df.columns if c not in ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges','Churn','customerID']]
    dense_col = ['SeniorCitizen','tenure','MonthlyCharges','TotalCharges']
    for c in sparse_col:
        df[c] = df[c].astype(np.int32)
    for c in dense_col:
        df[c] = df[c].astype(np.float32)
    df['Churn'] = df['Churn'].astype(np.float32)
    sparse_dict = {}
    for col in sparse_col:
        sparse_dict[col] = df[col].value_counts().shape[0]
    y_train = df[['Churn']]
    x_train = df.drop(['Churn','customerID'], axis = 1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = test_size)
    y_train = torch.from_numpy(y_train.values)
    x_train_dense = torch.from_numpy(x_train[dense_col].values)
    x_train_sparse = torch.from_numpy(x_train[sparse_col].values)
    x_test_dense = torch.from_numpy(x_test[dense_col].values)
    x_test_sparse = torch.from_numpy(x_test[sparse_col].values)
    y_test = torch.from_numpy(y_test.values)
    train_dict = {'x_sparse':x_train_sparse,
                  'x_dense':x_train_dense,
                  'y':y_train,
                  'sparse_col':sparse_col,
                  'dense_col':dense_col,
                  'sparse_dict':sparse_dict
                  }
    test_dict = {'x_sparse':x_test_sparse,
                 'x_dense':x_test_dense,
                 'y':y_test
                 }
    return train_dict, test_dict


def train(train_loader: DataLoader, early_stopping: int, model):
    """
    :param train_loader: train data DataLoader
    :param early_stopping: early stopping number
    :param model: model
    :return:
    """
    if torch.cuda.is_available():
        print("cuda is OK")
        model = model.to('cuda')
    else:
        print('cuda is NO')
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.BCELoss()
    logloss = []
    epoch_num = 2
    iter_num = 0
    best_loss = 1e8
    for epoch in range(epoch_num):
        print('starit epoch [{}/{}]'.format(epoch + 1, 5))
        model.train()
        for sparse_feature, dense_feature, label in train_loader:
            iter_num += 1
            if torch.cuda.is_available():
                sparse_feature, dense_feature, label = sparse_feature.to('cuda'), dense_feature.to('cuda'), label.to('cuda')
            pctr = model(sparse_feature, dense_feature)
            loss = criterion(pctr, label)
            iter_loss = loss.item()
            logloss.append(iter_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # early stopping
            if iter_loss < best_loss:
                best_loss = iter_loss
                es = 0
            else:
                es += 1
                print("Counter {} of 6".format(es))

                if es > early_stopping:
                    print("Early stopping with best_loss: ", best_loss, "and loss for this epoch: ", iter_loss, "...")
                    break

            if iter_num % 5 == 0:
                print("epoch {}/{}, total_iter is {}, logloss is {:.2f}".format(epoch+1, epoch_num, iter_num, iter_loss))
                print('pctr sum is : {}/ {}'.format(pctr.sum(),pctr.shape[0]))
        return model


def predict(model, test_sparse, test_dense, y_test = None):
    """

    :param model: model
    :param test_sparse: test sparse data
    :param test_dense: test dense data
    :param y_test: test y data
    :return: prediction result
    """
    prediction = model(test_sparse,test_dense).detach().numpy()
    prediction = prediction.astype(np.int32)
    if y_test is not None:
        y_test = y_test.detach().numpy().astype(np.int32)
        print(classification_report(y_test,prediction))
    return prediction



if __name__ == '__main__':
    main()