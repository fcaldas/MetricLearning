# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:04:18 2015

@author: fcaldas
"""
import numpy as np
import io;
import numpy as np;
import matplotlib as pl;
from scipy import io
from sklearn import metrics;
import matplotlib.pyplot as plt
import bisect;
from numba import jit

def generate_pairs(label, n_pairs, positive_ratio, random_state=41):
    """Generate a set of pair indices
    
    Parameters
    ----------
     X : array, shape (n_samples, n_features)
        Data matrix
    label : array, shape (n_samples, 1)
        Label vector
    n_pairs : int
        Number of pairs to generate
    positive_ratio : float
        Positive to negative ratio for pairs
    random_state : int
        Random seed for reproducibility
        
    Output
    ------
    pairs_idx : array, shape (n_pairs, 2)
        The indices for the set of pairs
    label_pairs : array, shape (n_pairs, 1)
        The pair labels (+1 or -1)
    """
    rng = np.random.RandomState(random_state)
    n_samples = label.shape[0]
    pairs_idx = np.zeros((n_pairs, 2), dtype=int)
    pairs_idx[:, 0] = np.random.randint(0, n_samples, n_pairs)
    rand_vec = rng.rand(n_pairs)
    for i in range(n_pairs):
        if rand_vec[i] <= positive_ratio:
            idx_same = np.where(label == label[pairs_idx[i, 0]])[0]
            while idx_same.shape[0] == 1:
                pairs_idx[i, 0] = rng.randint(0,n_samples)
                idx_same = np.where(label == label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_same.shape[0])
            pairs_idx[i, 1] = idx_same[idx2] 
            while pairs_idx[i, 1] == pairs_idx[i, 0]:
                idx2 = rng.randint(idx_same.shape[0])
                pairs_idx[i, 1] = idx_same[idx2] 
        else:
            idx_diff = np.where(label != label[pairs_idx[i, 0]])[0]
            idx2 = rng.randint(idx_diff.shape[0])
            pairs_idx[i, 1] = idx_diff[idx2]
    pairs_label = 2.0 * (label[pairs_idx[:, 0]] == label[pairs_idx[:, 1]]) - 1.0
    return pairs_idx, pairs_label

@jit
def update(X_i, X_j, A, y, u, l, gamma):
    diff = X_i - X_j
    d = np.dot(diff, np.dot(A , diff))
    if (d >u and y == 1) or (d < l and y == -1):
        target = u * (y == 1) + l * (y == -1)
        _y = ( (gamma * d * target - 1) + np.sqrt((gamma * d * target - 1) ** 2 + 4 * gamma * d * d) )/(2 * gamma * d)
        return A - ((gamma * (_y - target)) / (1 + gamma * (_y - target) * d)) * np.outer(np.dot(A, diff), np.dot(A, diff))
    else :
        return A

@jit    
def A_dist_pairs(X , A, pairs):
    n_pairs = pairs.shape[0]
    dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
    for i in range(n_pairs):
        diff = X[pairs[i, 0], :] - X[pairs[i, 1], :]
        dist[i] = np.dot(diff , np.dot(A , diff))
    return np.sqrt(dist)

def convergeA( easy, X, y , u , l ,posRatio = .1, nbatch = 100, 
               batch_size = 5000, A = np.eye(1500), gamma = 0.01, saveEach = 5):
    print(" Starting convergence...")
    
    for i in range(0, nbatch):
        idx, pairs_label = generate_pairs(y, batch_size, posRatio, random_state = i + 1)
        score = 1
        dist = A_dist_pairs(X, A, idx)
        fpr, tpr, thresholds = metrics.roc_curve(pairs_label, -dist)
        
        if easy:            
            score = 1.0 - tpr[bisect.bisect(fpr, 0.001) - 1]
        else:
            sc_idx = (np.abs(fpr + tpr - 1.)).argmin()
            score = (fpr[sc_idx]+(1-tpr[sc_idx]))/2
            
        for t in range (batch_size):
            diff = X[idx[t, 0], :] - X[idx[t, 1], :];
            dist = np.dot(diff, np.dot(A , diff))
            if (dist >u and pairs_label[t] == 1) or (dist < l and pairs_label[t] == -1):
                if(pairs_label[t] == 1):
                    target = u
                else:
                    target = l
                ybar = ( (gamma * dist * target - 1) + np.sqrt((gamma * dist * target - 1) ** 2 + 4 * gamma * dist * dist) )/(2 * gamma * dist)
                A -= ((gamma * (ybar - target)) / (1 + gamma * (ybar - target) * dist)) * np.outer(np.dot(A, diff), np.dot(A, diff))
            else:
                continue;
#            A = update(X[idx[t, 0], :], X[idx[t, 1], :], A, pairs_label[t], u, l, gamma)
               
            
        print("On iteration %d - score = %.3f"%(i, score))
        if(i%saveEach == 0):
            
            if(easy):
                print("Batch number : %d saving A%d.txt"%(i,i));
                np.savetxt("A%d.txt"%i,A);
            else:
                print("Batch number : %d saving B%d.txt"%(i,i));
                np.savetxt("B%d.txt"%i,A);
    if(easy):
        np.savetxt("Afinal.txt",A);
    else:
        np.savetxt("Bfinal.txt",A);
    return A
    
def plotM(A):
    plt.figure(1)
    plt.imshow(A, interpolation='none')
    plt.grid(True)
    plt.show();
    
