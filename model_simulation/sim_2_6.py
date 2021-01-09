import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec
#import os
from scipy.stats import norm
import pickle

n = 50000
w1 = 1.0
w2 = 2.7
w3 = 3.1
w4 = -2.0
w5 = -1.0
w6 = -5.6


X = []
likelihood = []
for i in range(0,n):
    z1 = np.random.normal(0.0, 1.0, 1).astype(np.float32)
    z2 = np.random.normal(1.0, 2.0, 1).astype(np.float32)
    #print(z1.shape)
    mean = 0
    sigma = 0.01 ** 0.5
    noise = np.random.normal(mean, sigma, 6).astype(np.float32)
    #print(noise.shape)
    xi = np.squeeze(np.array([w1*np.sin(z1), w2*np.cos(z2), w3*z1**2, w4*z2, w5*(z1**3), w6*(z2+z2**2) ]))
    #print(xi.shape)
    xi = np.add(xi, noise)
    print(xi.shape)
    X.append(xi)
    J1 = [w1*np.cos(z1),    0.0,               2*w3*z1,    0.0,          3*w5*z1**2,      0.0 ]
    J2 = [0.0,              -w2*np.sin(z2),    0.0,          w4,         0.0,             w6*(1+2*z2) ]
    llk = np.log(norm.pdf(z1, 0.0, 1.0)) + np.log(norm.pdf(z2, 1.0, 2.0))
    llk = llk - 0.5 * np.log(np.sum(np.array(J1)**2)*np.sum(np.array(J2)**2))
    print(llk)
    likelihood.append(llk)

data = {'likelihood':likelihood, 'X':X}
pickle.dump(data, open( "data_2_6_train.p", "wb" ))




X = []
likelihood = []
for i in range(0,n):
    z1 = np.random.normal(0.0, 1.0, 1).astype(np.float32)
    z2 = np.random.normal(1.0, 2.0, 1).astype(np.float32)
    #print(z1.shape)
    mean = 0
    sigma = 0.01 ** 0.5
    noise = np.random.normal(mean, sigma, 6).astype(np.float32)
    #print(noise.shape)
    xi = np.squeeze(np.array([w1*np.sin(z1), w2*np.cos(z2), w3*z1**2, w4*z2, w5*(z1**3), w6*(z2+z2**2) ]))
    #print(xi.shape)
    xi = np.add(xi, noise)
    print(xi.shape)
    X.append(xi)
    J1 = [w1*np.cos(z1),    0.0,               2*w3*z1,    0.0,          3*w5*z1**2,      0.0 ]
    J2 = [0.0,              -w2*np.sin(z2),    0.0,          w4,         0.0,             w6*(1+2*z2) ]
    llk = np.log(norm.pdf(z1, 0.0, 1.0)) + np.log(norm.pdf(z2, 1.0, 2.0))
    llk = llk - 0.5 * np.log(np.sum(np.array(J1)**2)*np.sum(np.array(J2)**2))
    print(llk)
    likelihood.append(llk)

data = {'likelihood':likelihood, 'X':X}
pickle.dump(data, open( "data_2_6_test.p", "wb" ))







