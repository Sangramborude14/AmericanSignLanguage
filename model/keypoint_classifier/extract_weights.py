import zipfile
import h5py
import numpy as np

keras_file = 'model/keypoint_classifier/keypoint_classifier.keras'
npz_file = 'model/keypoint_classifier/keypoint_classifier_weights.npz'

with zipfile.ZipFile(keras_file) as z:
    with h5py.File(z.open('model.weights.h5'), 'r') as f:
        # BatchNormalization
        bn_vars = f['layers\\batch_normalization']['vars']
        gamma = bn_vars['0'][:]
        beta = bn_vars['1'][:]
        moving_mean = bn_vars['2'][:]
        moving_variance = bn_vars['3'][:]
        
        # Dense layers
        w0 = f['layers\\dense']['vars']['0'][:]
        b0 = f['layers\\dense']['vars']['1'][:]
        
        w1 = f['layers\\dense_1']['vars']['0'][:]
        b1 = f['layers\\dense_1']['vars']['1'][:]
        
        w2 = f['layers\\dense_2']['vars']['0'][:]
        b2 = f['layers\\dense_2']['vars']['1'][:]
        
        w3 = f['layers\\dense_3']['vars']['0'][:]
        b3 = f['layers\\dense_3']['vars']['1'][:]

np.savez(
    npz_file,
    bn_gamma=gamma,
    bn_beta=beta,
    bn_mean=moving_mean,
    bn_var=moving_variance,
    w0=w0, b0=b0,
    w1=w1, b1=b1,
    w2=w2, b2=b2,
    w3=w3, b3=b3
)
print("Successfully extracted weights to", npz_file)
