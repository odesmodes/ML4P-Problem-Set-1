'''
Making a mlp ML model to predict LOGG values of RGB stars given their spectra

Author: Sarah Odeh (NYU)

For more detailed explanation, see 'my understanding of the workflow step 4'
'''

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from get_data import train_features, valid_features, test_features
from get_data import train_labels_logg, valid_labels_logg, test_labels_logg

# convert to tensors
X_train = torch.tensor(train_features, dtype=torch.float32)
X_valid = torch.tensor(valid_features, dtype=torch.float32)
X_test  = torch.tensor(test_features,  dtype=torch.float32)

Y_train = torch.tensor(train_labels_logg.astype('float32')).unsqueeze(1)
Y_valid = torch.tensor(valid_labels_logg.astype('float32')).unsqueeze(1)
Y_test  = torch.tensor(test_labels_logg.astype('float32')).unsqueeze(1)

# build the model structure

''' For 3 layers hardcoded into the module, just to see what it looks like:
class MLP(nn.Module):
    def__init__(self, num_pixels)
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(num_pixels,256),
            nn.ReLu(),
            nn.Linear(256,128),
            nn.ReLu(),
            nn.Linear(128,64),
            nn.Linear(64,1)
        )

    def forward(self, x):
        return self.network(x)
'''

class MLP(nn.Module):
    def __init__(self, num_pixels, hidden_sizes):
        super().__init__()
        
        layers = [] # empty array
        in_size = num_pixels
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size)) # add  a linear layer that goes from the initial size to the next size
            layers.append(nn.ReLU()) # add ReLu after the layer
            in_size = hidden_size # update the inital layer to be the output of last layer
        
        layers.append(nn.Linear(in_size, 1)) # make the final layer that produces 1 output
        
        self.network = nn.Sequential(*layers) # unpack all the layers from the layers array and pass through them sequentially
    
    def forward(self, x):
        return self.network(x)

# define model
num_pixels = X_train.shape[1]
model = MLP(num_pixels, hidden_sizes=[512,256,128])
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 1000
train_losses = []
valid_losses = []

for epoch in range(num_epochs):
    model.train() 
    prediction = model(X_train)
    loss = loss_fn(prediction, Y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    model.eval()
    train_preds = model(X_train)
    valid_preds = model(X_valid)
    
    train_loss = loss_fn(train_preds, Y_train)
    valid_loss = loss_fn(valid_preds, Y_valid)


print(f"Train Loss: {train_loss.item():.4f}")
print(f"Valid Loss: {valid_loss.item():.4f}")


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
plt.savefig('mlp.png')
