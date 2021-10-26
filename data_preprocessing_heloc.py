import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pickle

if __name__ == "__main__" :

    # load heloc dataset
    heloc_data = pd.read_csv('./data/HELOC/HELOC_dataset.csv')

    # drop the noised feature
    heloc_data = heloc_data.drop(['ExternalRiskEstimate'], axis = 1)
    drop_index = heloc_data[heloc_data['MSinceOldestTradeOpen'] == -9].index
    heloc_data = heloc_data.drop(drop_index)

    # extract the label and train data
    heloc_features = heloc_data.columns
    train_label = heloc_features.drop('RiskPerformance')
    heloc_label = heloc_data['RiskPerformance']
    train_data = heloc_data[train_label]

    # train, test split by 0.25 ratio
    x_train, x_test, y_train, y_test = train_test_split(train_data, heloc_label, test_size=0.25)

    # save the original data
    x_train.to_csv('./data/HELOC/HELOC_x_train.csv')
    x_test.to_csv('./data/HELOC/HELOC_x_test.csv')
    y_train.to_csv('./data/HELOC/HELOC_y_train.csv')
    y_test.to_csv('./data/HELOC/HELOC_y_test.csv')


    # scale code
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train_np = np.array(x_train)
    x_test_np = np.array(x_test)
    y_train_np = np.array(y_train)
    y_test_np = np.array(y_test)

    x_train_tensor = torch.tensor(x_train_np, device=device).float()
    x_test_tensor = torch.tensor(x_test_np, device=device).float()
    y_train_tensor = torch.tensor(y_train_np, device=device).long()
    y_test_tensor = torch.tensor(y_test_np, device=device).long()

    max_list = []
    min_list = []

    for i in range(x_train_tensor.shape[1]) :
        x_col_min = x_train_tensor[:, i].min()
        x_col_max = x_train_tensor[:, i].max()
        x_train_tensor[:, i] = (x_train_tensor[:, i] - x_col_min) / (x_col_max - x_col_min)
        x_test_tensor[:, i] = (x_test_tensor[:, i] - x_col_min) / (x_col_max - x_col_min)
        max_list.append(x_col_max)
        min_list.append(x_col_min)

    # save the scaled data
    with open('./data/HELOC/HELOC_x_train_scaled_tensor', 'wb') as f:
        pickle.dump(x_train_tensor, f)
    with open('./data/HELOC/HELOC_x_test_scaled_tensor', 'wb') as f:
        pickle.dump(x_test_tensor, f)
    with open('./data/HELOC/HELOC_y_train_tensor', 'wb') as f:
        pickle.dump(y_train_tensor, f)
    with open('./data/HELOC/HELOC_y_test_tensor', 'wb') as f:
        pickle.dump(y_test_tensor, f)

    # ferification code
    with open('./data/HELOC/HELOC_x_train_scaled_tensor', 'rb') as f:
        data = pickle.load(f)
        print(data.shape)
    with open('./data/HELOC/HELOC_x_test_scaled_tensor', 'rb') as f:
        data = pickle.load(f)
        print(data.shape)
    with open('./data/HELOC/HELOC_y_train_tensor', 'rb') as f:
        data = pickle.load(f)
        print(data.shape)
    with open('./data/HELOC/HELOC_y_test_tensor', 'rb') as f:
        data = pickle.load(f)
        print(data.shape)



