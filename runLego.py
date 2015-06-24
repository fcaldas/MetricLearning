# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:49:17 2015

@author: fcaldas
"""
from scipy import io
from lego_functions import *
import bisect;
plt.style.use('ggplot')

facile = True
submit = True

if(facile):
    X = io.loadmat('/tsi/datachallenge/data_train_facile.mat')
else:
    X = io.loadmat('/tsi/datachallenge/data_train_difficile.mat')

y = X['label']
X = X['X']
u = 7
l = 10

#A = np.loadtxt("Astart.txt")
A = convergeA(facile, X, y , u , l ,posRatio = .14, nbatch = 100, batch_size = 20000, gamma = 0.08, saveEach = 5);

idx, pairs_label = generate_pairs(y, 2000, .1)

dist = A_dist_pairs(X, A, idx )
#dist = calculateDistances(X, idx, A)

fpr, tpr, thresholds = metrics.roc_curve(pairs_label, -dist)

#
#plt.clf()
#plt.plot(fpr, tpr, label='ROC curve')
#plt.plot([0, 1], [0, 1], 'k--')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.0])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.show()

##cross validation
if(facile):
    score_facile = 1.0 - tpr[bisect.bisect(fpr, 0.001) - 1]
    print "Score = " , score_facile
else:
    idx = (np.abs(fpr + tpr - 1.)).argmin()
    print "Score = " , (fpr[idx]+(1-tpr[idx]))/2


if(submit):
    if(facile):
        test_facile = io.loadmat('/tsi/datachallenge/data_test_facile.mat')
        dist_test = calculateDistances(test_facile['X'], test_facile['pairs'], A)
        np.savetxt('soumission_facile.txt', dist_test, fmt='%.5f')
    else:
        test_facile = io.loadmat('/tsi/datachallenge/data_test_difficile.mat')
        dist_test = calculateDistances(test_facile['X'], test_facile['pairs'], A)
        np.savetxt('soumission_facile.txt', dist_test, fmt='%.5f')        