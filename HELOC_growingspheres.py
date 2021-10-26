import numpy as np
import pandas as pd
from sklearn import datasets, ensemble
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import growingspheres.growingspheres.counterfactuals as cf
from models import clf_model

import numpy as np
import pandas as pd
from sklearn import datasets, ensemble, tree
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist, pdist
from numpy import linalg as LA
import pickle

def cuda_to_numpy(cuda):
    narray = np.array(cuda.cpu())
    return narray

def evaluate_model(model, x_test, y_test, sklearn = False):
    if sklearn == False:
        model.eval()
        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        print('test_acc', (predicted == y_test).sum() / y_test.shape[0])

    elif sklearn == True:
        model.eval()
        predicted = np.round(model.predict(x_test))
        correct = (np.round(predicted) == y_test).sum()
        print('test_acc', correct / x_test.shape[0])


# data load

with open('./data/HELOC/HELOC_x_train_scaled_tensor', 'rb') as f:
    x_train = pickle.load(f)
with open('./data/HELOC/HELOC_x_test_scaled_tensor', 'rb') as f:
    x_test = pickle.load(f)
with open('./data/HELOC/HELOC_y_train_tensor', 'rb') as f:
    y_train = pickle.load(f)
with open('./data/HELOC/HELOC_y_test_tensor', 'rb') as f:
    y_test = pickle.load(f)

y_test = cuda_to_numpy(y_test)
x_test = cuda_to_numpy(x_test)

# model load as clf
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clf_state_dict = torch.load('models/saved/heloc_MLP.pt')
clf = clf_model.MLP(input_size=22, output_size=2).to(device)
clf.load_state_dict(clf_state_dict)
evaluate_model(clf, x_test, y_test, sklearn=True)

# generate counterfactual examples by growingsphere
cf_list = []
cnt = 0
nbr_experiments = 2468

for obs in x_test[:nbr_experiments]:
    print('=================================================', cnt)
    CF = cf.CounterfactualExplanation(obs, clf.predict, method='GS')
    CF.fit(n_in_layer=2000, first_radius=0.1, dicrease_radius=10, sparse=True, verbose=True)
    cf_list.append(CF.enemy)
    cnt += 1
cf_list = np.array(cf_list)

# evaluate CF

evaluate_model(clf, x_test, y_test, sklearn=True)

print()
print("qualify : ", (clf.predict(cf_list) + clf.predict(x_test[:nbr_experiments])).sum(),"=",nbr_experiments)

with open('./data/HELOC/heloc_cf.pk', 'wb') as f:
    pickle.dump(cf_list, f)

with open('./data/HELOC/heloc_cf.pk', 'rb') as f:
    temp = pickle.load(f)

print("qualify : ", (clf.predict(temp) + clf.predict(x_test[:nbr_experiments])).sum(),"=",nbr_experiments)