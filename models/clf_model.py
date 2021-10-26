import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=10, output_size=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.leakyrelu = nn.LeakyReLU()
        self.elu = nn.ELU()

    
    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.relu(self.fc2(output))
        output = self.relu(self.fc3(output))
        output = self.relu(self.fc4(output))
        output = self.fc5(output)
        return output

    def predict(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.from_numpy(x).to(device).float()
        output = self.relu(self.fc1(x))
        output = self.relu(self.fc2(output))
        output = self.relu(self.fc3(output))
        output = self.relu(self.fc4(output))
        output = self.fc5(output)
        _, output = torch.max(output.data, 1)
        output = np.array(output.cpu().detach().numpy())
        return output



