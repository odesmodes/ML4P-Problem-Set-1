'''
Making a KNN ML model for predicting LOGG values of RGB stars

Author: Sarah Odeh (NYU)

See 'my understanding of the workflow step 3' for detailed explanation
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from get_data import train_features, valid_features, test_features
from get_data import train_labels_logg, valid_labels_logg, test_labels_logg

# build model
k = 12
model = KNeighborsRegressor(n_neighbors=k)
model.fit(train_features, train_labels_logg)

# predictions (validation and training)
train_preds = model.predict(train_features)
valid_preds = model.predict(valid_features)

# loss (validation vs training)
train_mse = np.mean((train_preds - train_labels_logg)**2)
valid_mse = np.mean((valid_preds - valid_labels_logg)**2)

print(f"Train MSE: {train_mse:.4f}")
print(f"Valid MSE: {valid_mse:.4f}")

# plotting validation predictions vs true LOGG to prove this works
plt.figure(figsize=(6,6))
plt.scatter(valid_labels_logg, valid_preds, s=5)
plt.plot([0, 3], [0, 3], color='red', label='perfect prediction')
plt.xlabel('True LOGG')
plt.ylabel('Predicted LOGG')
plt.title(f'KNN (k={k}): Predicted vs True LOGG')
plt.legend()
plt.savefig('knn.png')

# test (ONLY WHEN READY)
test_preds = model.predict(test_features)
test_mse = np.mean((test_preds - test_labels_logg)**2)
print(f"Test MSE: {test_mse:.4f}")
