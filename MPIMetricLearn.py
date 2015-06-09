# -*- coding: utf-8 -*-
"""
Implementation of a Distributed Stochastic Gradient Descent
for Metric Learning

Created on Mon Jun  8 16:49:50 2015

@author: fcaldas@enst.fr
"""

from numba import jit
import numpy as np,numpy.linalg
from mpi4py import MPI

@jit
def distanceA(X, pairs, A):
    n_pairs = pairs.shape[0]
    dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
    for a in range(0, n_pairs):
        dist[a] = distA(X[pairs[a,0], :], X[pairs[a,1], :], A)
    return dist

@jit
def psd_proj(M):
    """ projection de la matrice M sur le cone des matrices semi-definies
    positives"""
    # calcule des valeurs et vecteurs propres
    eigenval, eigenvec = np.linalg.eigh(M)
    # on trouve les valeurs propres negatives ou tres proches de 0
    ind_pos = eigenval > 1e-10
    # on reconstruit la matrice en ignorant ces dernieres
    M = np.dot(eigenvec[:, ind_pos] * eigenval[ind_pos][np.newaxis, :],
               eigenvec[:, ind_pos].T)
    return M


class MPIMetricLearn:
    rank = 0;
    anysource = 0;
    comm = 0;
    size = 0;
    A = np.zeros([1500,1500]);
        
    def __init__(self):
        self.anysource = MPI.ANY_SOURCE 	# New name for MPI Global variable saying one can receive from anysource (see below)
        self.comm = MPI.COMM_WORLD		# New name for MPI Global Environment
        self.size = comm.Get_size()		# The size of the world i.e. the number of active agents
        self.rank = comm.Get_rank()		# MY agent number (this variable DIFFERS from one agent to another)

        
    @jit
    def distA(self, X1, X2, A):
        return np.dot(np.dot((X1 - X2).transpose(), A), X1-X2)

    @jit
    def predictA(self, X, pairs, A):
        n_pairs = pairs.shape[0]
        dist = np.ones((n_pairs,), dtype=np.dtype("float32"))
        for a in range(0, n_pairs):
            dist[a] = distA(X[pairs[a,0], :], X[pairs[a,1], :], A)
        return dist

    #The gradient is the sum of the N gradients
    def syncA(self):
        comm.Barrier();
        Ad = np.zeros(self.A.shape);
        comm.Reduce(self.A, Ad, MPI.SUM, root=0);
        comm.Bcast(Ad, root = 0);
        self.A = Ad #/ self.size;

    def saveA(self):
        if(self.rank == 0):
            np.savetxt("A.txt",self.A)
        comm.Barrier();
        
    #calculate the gradient of g(A)
    #dont pass all pairs to this function
    #to use an stochastic approach you have to change the pairs
    #sent to this function on each iteration
    @jit
    def calcGradgA(self, A, X, pairs, indices):
        A0 = np.zeros([X.shape[1],X.shape[1]]);    
        for i in range(0,indices.shape[0]):
            vec = np.atleast_2d(X[pairs[indices[i]][0]] - X[pairs[indices[i]][1]])
            if((2* np.sqrt(np.dot(np.dot(vec.T[:,0], A), vec[0,:] ) )) != 0):
                A0 += 1.0/(2* np.sqrt(np.dot(np.dot(vec.T[:,0], A), vec[0,:] ) )) * np.dot(vec.T, vec);
        return A0
    
    @jit
    def descent(self, X, pair, labels, A0 = np.diag(np.ones(X.shape[1]))):
        alpha = 0.01
        norm = np.linalg.norm(A0);    
        iteration = 0
        indicesD = np.where(labels == -1)[0]
        indicesS = np.where(labels == +1)[0]
        normAntes = 1;
        while(np.abs(norm - normAntes) > 0.02):
            normAntes = norm
            #in the future use only a subset of X
            A1 = A0 + alpha * calcGradgA(A0, X , pairs, indicesD)
            A1 = A1 - alpha/2 * calcGradgA(A0, X , pairs, indicesS)
            # projections!        
            A1 = psd_proj(A1)
            #set new alpha
            #sum of positive matrices is still positive
            self.A = A1
            self.syncA();
            
            norm = np.linalg.norm(A1 - A0)        
            A0 = A1;
            iteration += 1
            if(self.rank == 0):
                print "Iter = ", iteration, " norm = ", np.abs(norm - normAntes);
        return A0
    