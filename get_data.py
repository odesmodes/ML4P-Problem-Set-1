'''
PS1 Data:
Authors
- David W Hogg (NYU)
- Clark Miyamoto (NYU)
- Sarah Odeh (NYU)

See 'my understanding of the workflow, step 1' for explanations
'''

import numpy as np
from astropy.io import fits 

# get labels
path_labels = "./labels.fits"
allstar = fits.open(path_labels)
labels = allstar[1].data

# make a reasonable red-giant-branch sample
RGB = True
RGB = np.logical_and(RGB, labels['TEFF'] > 3500.)
RGB = np.logical_and(RGB, labels['TEFF'] < 5400.)
RGB = np.logical_and(RGB, labels['LOGG'] < 3.0)
RGB = np.logical_and(RGB, labels['LOGG'] > 0.0)
RGB = np.logical_and(RGB, labels['H'] < 10.5)
RGB_labels = labels[RGB]

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

# we only want LOGG
train_labels_logg = train_labels['LOGG']
valid_labels_logg = valid_labels['LOGG']
test_labels_logg = test_labels['LOGG']

# get the features

train_features = np.load('./train_features.npy')
valid_features = np.load('./valid_features.npy')
test_features = np.load('./test_features.npy')


# If you want to download more data then run the code under # get labels (comment out everything else up till here, and uncomment the following: 

'''
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

# function to get the features
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

# download the data- this will take a while, the first time you run it. Don't be alarmed.
train_features_downloaded = get_features(train_labels)
valid_features_downloaded = get_features(valid_labels)
test_features_downloaded = get_features(test_labels)
print(train_features_downloaded.shape, valid_features_downloaded.shape, test_features_downloaded.shape)

# don't forget to include a location for the data to save: 

'''
