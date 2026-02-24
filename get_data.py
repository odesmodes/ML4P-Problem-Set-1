#!/usr/bin/env python
# coding: utf-8

# # PS1 Data
# Authors
# - David W Hogg (NYU)
# - Clark Miyamoto (NYU)
# 
# Requirements:
# - `astropy`

# # Get the Labels

# In[2]:


import numpy as np
from astropy.io import fits # You might need to pip install this
import pylab as plt # only needed for verification

# In[4]:


path_labels = "./labels.fits"
allstar = fits.open(path_labels)

# In[5]:


# the labels are in an enormous table in element [1] of this FITS file
labels = allstar[1].data
plt.scatter(labels['TEFF'], labels['LOGG'], s=1)
plt.xlim(6000, 3500)
plt.ylim(5, 0)

# In[6]:


# make a reasonable red-giant-branch sample
RGB = True
RGB = np.logical_and(RGB, labels['TEFF'] > 3500.)
RGB = np.logical_and(RGB, labels['TEFF'] < 5400.)
RGB = np.logical_and(RGB, labels['LOGG'] < 3.0)
RGB = np.logical_and(RGB, labels['LOGG'] > 0.0)
RGB = np.logical_and(RGB, labels['H'] < 10.5)
print(np.sum(RGB))

# In[7]:


# make a plot that an astronomer likes to see
RGB_labels = labels[RGB]
plt.scatter(RGB_labels['TEFF'], RGB_labels['LOGG'], c=RGB_labels['FE_H'], s=1)
plt.xlim(5400, 3500)
plt.xlabel("effective temperature")
plt.ylim(3., 0.)
plt.ylabel("log10 surface gravity")
plt.colorbar(label="metallicity")

# In[8]:


# make train, validation, and test data sets
rng = np.random.default_rng(17)
N_RGB = len(RGB_labels)
N_train, N_valid, N_test = 1024, 256, 512
I = rng.permutation(N_RGB)
I_train = I[0:N_train]
I_valid = I[N_train:N_train+N_valid]
I_test = I[N_train+N_valid:N_train+N_valid+N_test]

train_labels = RGB_labels[I_train]
valid_labels = RGB_labels[I_valid]
test_labels = RGB_labels[I_test]
print(len(train_labels), len(valid_labels), len(test_labels))

# Here how you get `LOGG`

# In[9]:


train_labels_logg = train_labels['LOGG']
print(train_labels_logg.shape) # (num_spectra, 1)

# # Get the Features

# In[10]:


import numpy as np
from matplotlib import pyplot as plt

# In[11]:


train_features = np.load('./train_features.npy')
valid_features = np.load('./valid_features.npy')
test_features = np.load('./test_features.npy')
for i in range(10):
    plt.plot(train_features[i] + i)

# In[14]:


print(train_features.shape) # (num_spectra, num_pixels)

'''
# ____
# 
# # If you want to download more data
# 
# Run the `Get Labels` section, and then run this!
# 
# Adjust `N_train`, `N_valid`, and `N_test`.

# In[ ]:


# make train, validation, and test data sets
rng = np.random.default_rng(17)
N_RGB = len(RGB_labels)
N_train, N_valid, N_test = 1024, 256, 512 # <- Adjust these for more data! 
I = rng.permutation(N_RGB)
I_train = I[0:N_train]
I_valid = I[N_train:N_train+N_valid]
I_test = I[N_train+N_valid:N_train+N_valid+N_test]
train_labels = RGB_labels[I_train]
valid_labels = RGB_labels[I_valid]
test_labels = RGB_labels[I_test]
print(len(train_labels), len(valid_labels), len(test_labels))

# In[ ]:


base_url = "https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/"
def get_features(labels):
    features = None
    for l in labels:
        url = base_url + l['TELESCOPE'] + "/" + l['FIELD'] + "/aspcapStar-dr17-" + l['APOGEE_ID'] + ".fits"
        x = fits.open(url)[1].data
        if features is None:
            features = x[None, :]
        else:
            features = np.concatenate((features, x[None, :]), axis=0)
    return features

# In[ ]:


# Downloads data from SDSS
# This will take a while, the first time you run it. Don't be alarmed.
train_features_downloaded = get_features(train_labels)
valid_features_downloaded = get_features(valid_labels)
test_features_downloaded = get_features(test_labels)
print(train_features_downloaded.shape, valid_features_downloaded.shape, test_features_downloaded.shape)
'''
