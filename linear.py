'''
Making a linear regression ML model for predicting LOGG values of RGB stars

Author: Sarah Odeh (NYU)

See 'my understanding of the workflow step 2' for detailed explanation
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from get_data import train_features, valid_features, test_features
from get_data import train_labels_logg, valid_labels_logg, test_labels_logg

# convert data to tensors
X_train = torch.tensor(train_features, dtype=torch.float32)
X_valid = torch.tensor(valid_features, dtype=torch.float32)
X_test  = torch.tensor(test_features,  dtype=torch.float32)

# unsqeezed to turn into 2D array bc LOGG values are in 1D list
Y_train = torch.tensor(train_labels_logg.astype(np.float32)).unsqueeze(1)
Y_valid = torch.tensor(valid_labels_logg.astype(np.float32)).unsqueeze(1)
Y_test  = torch.tensor(test_labels_logg.astype(np.float32)).unsqueeze(1)

# define the model structure
class LinearRegression(nn.Module):
    def __init__(self, num_pixels):
        super().__init__()
        self.linear = nn.Linear(num_pixels, 1)

    def forward(self, x):
        return self.linear(x)

# create the model
num_pixels = X_train.shape[1]
model = LinearRegression(num_pixels)
#print(model)

# set up loss and optimizer functions
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train
num_epochs = 1000

for epoch in range(num_epochs):
    # forward pass
    prediction = model(X_train)
    loss = loss_fn(prediction, Y_train)
    
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''    
    # print loss every 10 epochs, expect to see decrease
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
'''

# validate
with torch.no_grad():
    train_preds = model(X_train)
    valid_preds = model(X_valid)
    
    train_loss = loss_fn(train_preds, Y_train)
    valid_loss = loss_fn(valid_preds, Y_valid)

print(f"Train Loss: {train_loss.item():.4f}")
print(f"Valid Loss: {valid_loss.item():.4f}")

'''
# early stopping (to decide on epoch range)

train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    train_losses.append(train_loss.item())
    valid_losses.append(valid_loss.item())

plt.plot(train_losses, label='train')
plt.plot(valid_losses, label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
'''

# plotting validation prediction vs true LOGG to prove this works
with torch.no_grad():
    valid_preds = model(X_valid).squeeze().numpy()

true_logg = Y_valid.squeeze().numpy()

plt.figure(figsize=(6,6))
plt.scatter(true_logg, valid_preds, s=5)
plt.plot([0, 3], [0, 3], color='red', label='perfect prediction')
plt.xlabel('True LOGG')
plt.ylabel('Predicted LOGG')
plt.title('Linear Regression: Predicted vs True LOGG')
plt.legend()
plt.savefig('linear_regression.png')

# test (RUN ONLY WHEN READY)
with torch.no_grad():
    test_preds = model(X_test)
    test_loss = loss_fn(test_preds, Y_test)
print(f"Test Loss: {test_loss.item():.4f}")
