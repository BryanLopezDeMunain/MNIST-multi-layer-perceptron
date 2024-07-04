

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

morse_x = pd.read_csv('morse_x.csv', header = None)
morse_x.columns = ['x']
morse_y = pd.read_csv('morse_y.csv')
morse_y.columns = ['y']
normal_x = pd.read_csv('normal_x.csv')
normal_x.columns = ['x']
normal_y = pd.read_csv('normal_y.csv')
normal_y.columns = ['y']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(morse_x.iloc[2:, :].to_numpy(), morse_y.iloc[1:, :].to_numpy())

print(len(x_train))
print(len(x_test))

x_train = torch.tensor(x_train)
y_train = torch.tensor(y_train)
x_test = torch.tensor(x_test)
y_test = torch.tensor(y_test)

import torch.nn as nn

torch.manual_seed(13)

vanilla_model = nn.Sequential(
nn.Linear(1, 100),
nn.ReLU(),
nn.Linear(100, 100),
nn.ReLU(),
nn.Linear(100, 100),
nn.ReLU(),
nn.Linear(100, 1)
)

vanilla_model = vanilla_model.double()

loss = nn.MSELoss()

import torch.optim as optim

optimizer = optim.Adam(vanilla_model.parameters(),lr=0.001)
# May want to change lr later

num_epochs = 400

for epoch in range(num_epochs):
    predictions = vanilla_model(x_train)
    MSE = loss(predictions, y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        print(MSE)

def evaluate_model(model, x_test, y_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        predictions = model(x_test)
        mse = nn.MSELoss()(predictions, y_test).item()
    return predictions, mse

# Evaluate the model
train_predictions, train_mse = evaluate_model(vanilla_model, x_train, y_train)
test_predictions, test_mse = evaluate_model(vanilla_model, x_test, y_test)

print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')

# Plotting the results
plt.figure(figsize=(12, 6))

from IPython.display import Math

# Training data
plt.subplot(1, 2, 1)
plt.scatter(x_train.numpy(), y_train.numpy(), label='Experimental', marker = 'o')
plt.scatter(x_train.numpy(), train_predictions.numpy(), label='Predicted', marker = 'x')
plt.title('Morse Training Data')
plt.ylabel('Normalized level population (%)')
plt.xlabel(r'U ($k_{B}T$) Joules')
plt.legend()
plt.xlim(0,3)
plt.ylim(0,7)
plt.grid(True)

# Test data
plt.subplot(1, 2, 2)
plt.scatter(x_test.numpy(), y_test.numpy(), label='Experimental', marker = 'o')
plt.scatter(x_test.numpy(), test_predictions.numpy(), label='Predicted', marker = 'x')
plt.title('Morse Test Data')
plt.ylabel('Normalized level population (%)')
plt.xlabel(r'U ($k_{B}T$) Joules')
plt.legend()
plt.xlim(0,3)
plt.ylim(0,7)
plt.tight_layout()
plt.grid(True)

plt.show()

vanilla_model(torch.tensor([0.00001], dtype=torch.float64))

# Works badly in the very low-x limit because not trained on much data here (only about 5 data points s.t. x<0.1)

pip install pysr

pred_x = np.linspace(0,3,1000).reshape(-1,1)
pred_y = []
for element in pred_x:
  element_tensor = torch.tensor([element], dtype=torch.float64)
  pred_y = np.append(pred_y, vanilla_model(element_tensor).detach().numpy())
pred_y = pred_y.reshape(-1,1)
plt.plot(pred_x, pred_y)

from pysr import PySRRegressor

model = PySRRegressor(
    niterations=40,  # < Increase me for better results
    ncycles_per_iteration=50,
    # ^ Generations between migrations.
    binary_operators=["+", "*", "/", "-"],
    unary_operators=[
        "exp",
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(pred_x, pred_y)

model.fit(x_train, y_train)

#plt.plot(pred_x, pred_y)

y_sym1 = (np.exp(-0.6028) / np.exp(0.39023 + ((pred_x * (-1.1107 + pred_x)) * pred_x))) / (-0.89903 + np.exp(pred_x))

y_sym2 = 0.085643 * (-4.7122 / (0.88314 - np.exp(pred_x)))

#plt.plot(pred_x, y_sym1)
plt.plot(pred_x, y_sym1, label = 'Symbolic model', color = 'orange')
plt.scatter(morse_x[1:], morse_y, label = 'Experimental', color = 'blue')
plt.legend()
plt.title('Plot of experimental data and symbolic model')
plt.ylabel('Normalized level population (%)')
plt.xlabel(r'U ($k_{B}T$) Joules')
plt.xlim(0,3)
plt.grid(True)

# Try on crazy function but with a lot of data points

def f(x):
  return np.exp(-30*x)

x = np.linspace(0,3,1000).reshape(-1,1)

y = f(x).reshape(-1,1)

plt.plot(x,y)
plt.xlim(0,3)

model.fit(x,y)