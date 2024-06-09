import torch
import pandas as pd
import numpy as np
import sklearn as learn

df_train = pd.read_csv('mnist_train.csv')
df_test = pd.read_csv('mnist_test.csv')
df_test.head()

y_train = df_train['label']
y_test = df_test['label']

y_train = torch.tensor(y_train.values, dtype = torch.long)
y_test = torch.tensor(y_test.values, dtype = torch.long)

x_train = df_train.drop(columns = ['label'])
x_test = df_test.drop(columns = ['label'])

x_train = torch.tensor(x_train.values, dtype = torch.float)
x_test = torch.tensor(x_test.values, dtype = torch.float)

import torch.nn as nn

torch.manual_seed(13)

vanilla_model = nn.Sequential(
nn.Linear(784, 100),
nn.ReLU(),
nn.Linear(100, 100),
nn.ReLU(),
nn.Linear(100, 100),
nn.ReLU(),
nn.Linear(100, 10),
nn.Softmax()
)

#loss = nn.MSELoss()
loss = nn.CrossEntropyLoss()

import torch.optim as optim

optimizer = optim.Adam(vanilla_model.parameters(),lr=0.001)

import time

num_epochs = 20

for epoch in range(num_epochs):
    start_time = time.time()
    predictions = vanilla_model(x_train)
    MSE = loss(predictions, y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 5 == 0:
        print(MSE)
        print(time.time() - start_time, 'seconds')

for epoch in range(num_epochs):
    start_time = time.time()
    predictions = vanilla_model(x_test)
    MSE = loss(predictions, y_test)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 5 == 0:
        print(MSE)
        print(time.time() - start_time, 'seconds')

num = 184

print('Should be', y_test[num])
print('Is', torch.argmax(vanilla_model(x_test[num])))