import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clf_model
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model, x_test, y_test, sklearn = False):
    if sklearn == False:
        model.eval()
        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)
        print('test_acc', (predicted == y_test).sum() / y_test.shape[0])

    elif sklearn == True:
        model.eval()
        predicted = np.round(model.predect(x_test))
        correct = (np.round(predicted) == y_test).sum().numpy().squeeze().sum()
        print('test_acc', correct / x_test.shape[0])

# hyperparameters

num_epochs = 50
learning_rate = 0.03

# load data 
with open('../data/UCI_creditcard/UCI_x_train_scaled_tensor', 'rb') as f:
    x_train = pickle.load(f)
with open('../data/UCI_creditcard/UCI_x_test_scaled_tensor', 'rb') as f:
    x_test = pickle.load(f)
with open('../data/UCI_creditcard/UCI_y_train_tensor', 'rb') as f:
    y_train = pickle.load(f)
with open('../data/UCI_creditcard/UCI_y_test_tensor', 'rb') as f:
    y_test = pickle.load(f)

model = clf_model.MLP(input_size=23, output_size=2).to(device)
nbr_class1 = y_train.sum().int()
nbr_class0 = y_train.shape[0] - nbr_class1
nbr_total = y_train.shape[0]
imbalanced_ratio = torch.tensor([nbr_class1/nbr_total, nbr_class0/nbr_total])
criterion = nn.CrossEntropyLoss(weight=imbalanced_ratio.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)

    if epoch % 1 == 0:
        print('train_loss', loss.item())
        evaluate_model(model, x_test, y_test, sklearn=False)

    loss.backward()
    optimizer.step()

print()
evaluate_model(model, x_test, y_test, sklearn=False)

torch.save(model.state_dict(), './saved/UCI_MLP.pt')