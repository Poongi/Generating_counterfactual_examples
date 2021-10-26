import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import cdist, pdist
import pickle
import torch
from models import clf_model


def evaluate_l2_metric(X_test, cf_list):
    diff = X_test - cf_list
    l2_norm = np.linalg.norm(diff, axis=1, ord=2)
    l2_norm_mean = np.mean(l2_norm)
    l2_norm_std = np.std(l2_norm)
    return l2_norm_mean, l2_norm_std


def evaluate_l1_metric(X_test, cf_list):
    diff = X_test - cf_list
    ls = []
    for i in range(diff.shape[0]):
        number_of_nonzero = (np.abs(diff[i]) >= 0.001).sum()
        ls.append(number_of_nonzero)
    l1_mean = np.mean(ls)
    l1_std = np.std(ls)
    return l1_mean, l1_std

def l1_l2_evaluation(x_test_, cf_):
    cf = cf_[np.where(~np.isnan(cf_[:, 0]))]
    x_test = x_test_[np.where(~np.isnan(cf_[:, 0]))]

    l1_mean, l1_std = evaluate_l1_metric(x_test, cf)
    l2_mean, l2_std = evaluate_l2_metric(x_test, cf)

    return l1_mean, l1_std, l2_mean, l2_std

def calculate_lipschitz_factor(x, x1):
    norm = LA.norm(x - x1)
    return norm


def evaluate_model(clf, X_test, y_test, sklearn = False):
    if sklearn == False:
        clf.eval()
        outputs = clf(X_test)
        _, predicted = torch.max(outputs.data, 1)
        print('test_acc', (predicted == y_test).sum()/y_test.shape[0])
    
    elif sklearn == True:
        clf.eval()
        predicted = np.round(clf.predict(X_test))
        correct = (np.round(predicted) == y_test.cpu().numpy().squeeze()).sum()
        print('test_acc', correct/X_test.shape[0])


def minmax_scaler(input, max_list, min_list):
    max_list_np = np.array(max_list).squeeze()
    min_list_np = np.array(min_list).squeeze()
    rtn = torch.tensor(input)
    for col in range(input.shape[1]):
        x_col_max = max_list[col]
        x_col_min = min_list[col]
        rtn[:, col] = (rtn[:, col] - x_col_min) / (x_col_max - x_col_min)
    return rtn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# original data load
x_train_unscaled_df = pd.read_csv('./data/UCI_creditcard/UCI_x_train.csv')
x_test_unscaled_df = pd.read_csv('./data/UCI_creditcard/UCI_x_test.csv')
x_train_unscaled_np = np.array(x_train_unscaled_df)
x_test_unscaled_np = np.array(x_test_unscaled_df)
x_train_unscaled = torch.from_numpy(x_train_unscaled_np).float().to(device)
x_test_unscaled = torch.from_numpy(x_test_unscaled_np).float().to(device)

# drop the noised data
x_train_unscaled = x_train_unscaled[:, 1:]
x_test_unscaled = x_test_unscaled[:, 1:]

# torch data load
with open('./data/UCI_creditcard/UCI_x_train_scaled_tensor', 'rb') as f:
    x_train = pickle.load(f)
with open('./data/UCI_creditcard/UCI_x_test_scaled_tensor', 'rb') as f:
    x_test = pickle.load(f)
with open('./data/UCI_creditcard/UCI_y_train_tensor', 'rb') as f:
    y_train = pickle.load(f)
with open('./data/UCI_creditcard/UCI_y_test_tensor', 'rb') as f:
    y_test = pickle.load(f)

# model load as clf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf_state_dict = torch.load('models/saved/UCI_MLP.pt')
clf = clf_model.MLP(input_size=23, output_size=2).to(device)
clf.load_state_dict(clf_state_dict)
evaluate_model(clf, x_test, y_test, sklearn=False)


# find min, max for scale

max_list = []
min_list = []

for i in range(x_train_unscaled.shape[1]) :
    x_col_min = x_train_unscaled[:, i].min()
    x_col_max = x_train_unscaled[:, i].max()
    max_list.append(x_col_max)
    min_list.append(x_col_min)

# CF load
with open('./data/UCI_creditcard/UCI_cf.pk', 'rb') as f:
    cf_list = pickle.load(f)

cf = torch.tensor(cf_list).to(device).float()

# verifying cf
_, predicted_cf = torch.max(clf(cf), 1)
_, predicted_x = torch.max(clf(x_test), 1)
print("qualify : ", (predicted_cf + predicted_x).sum(),"=",x_test.shape[0])


# finding instance that properly cross the decision boundary 
idx_diff_class = []
for i in range(x_test.shape[0]):
    if ((predicted_x[i] + predicted_cf[i]) == 1)  :
        idx_diff_class.append(i)

x_test = x_test[idx_diff_class]
cf = cf[idx_diff_class]
predicted_x = predicted_x[idx_diff_class]
predicted_cf = predicted_cf[idx_diff_class]


# copy the loaded data to evaluate on metrics
nbr_experiments = 1000
x = x_test.cpu()[:nbr_experiments]
predicted_test_x = clf(x_test).argmax(axis=1).cpu()
test_x = np.array(x_test.cpu())
test_cf = np.array(cf.cpu())
distance_metric = 'mahalanobis'

lipschitz_list = []
for target in range(nbr_experiments):
    # target = 0
    currentX = x[target].reshape((1,) + x[target].shape).to(device)
    predicted = clf(currentX).argmax(axis=1)
    currentX = np.array(currentX.cpu())
    predicted = predicted.cpu()

    # Input data whose predicted class is same as that of target x
    filteredIdx = np.where(predicted == predicted_test_x)[0]

    '''
    Calculating distances (higher values indicates more similarity)
    '''
    eps = 1e-4 # Possible candidates 1e-2, 1e-3, 1e-4
    dist = []
    for i in range(test_x[filteredIdx].shape[0]):
        if np.sum(currentX == test_x[filteredIdx][i]) == currentX.shape[1]:
            dist.append(-99)
        else:
            nSatisfied = np.sum(np.abs(currentX - test_x[filteredIdx][i]) < eps)
            dist.append(nSatisfied)
    distDescending, distIdx_ = np.sort(dist)[::-1], np.argsort(dist)[::-1]
    distIdx = distIdx_[distDescending > 15] # 2/3 * number of features
    if distIdx.size == 0:
        continue
    '''
    End
    '''
    filteredIdx_under_eps = filteredIdx[distIdx]
    target_cf = test_cf[filteredIdx_under_eps]

    lipschitz_list_for_all_candidates = []
    for under_eps in filteredIdx_under_eps:
        if under_eps == target:
            continue
        norm_x = calculate_lipschitz_factor(currentX, test_x[under_eps])
        norm_cf = calculate_lipschitz_factor(test_cf[target], test_cf[under_eps])
        lipschitz_list_for_all_candidates.append(norm_cf / norm_x)
    lipschitz_list.append(np.max(lipschitz_list_for_all_candidates))

print("Number of instance that pass the decision boundary :", len(idx_diff_class) )
print("Number of instances produced by the distance measure: {}".format(np.sum(lipschitz_list)))
print("Cohenrence mean: {:.2f}".format(np.mean(lipschitz_list)))
print("Cohenrence std: {:.2f}".format(np.std(lipschitz_list)))
print("=====================================================================")

l1_mean, l1_std, l2_mean, l2_std = l1_l2_evaluation(test_x, test_cf)
print("l1_mean: {:.2f} \t\t l2_mean: {:.2f}".format(l1_mean, l2_mean))
print("l1_std: {:.2f} \t\t l2_std: {:.2f}".format(l1_std, l2_std))