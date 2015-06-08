# -*- coding: utf-8 -*-
"""
Find computers that are up on at Télécom ParisTeh network
Created on Mon Jun  8 13:36:53 2015

@author: fcaldas
"""
import numpy as np
import os

def getMachines():
    minRoom = 124
    maxRoom = 128
    minMachine = 1
    maxMachine = 20
    machineName = "c%02d-%02d";
    mList = [];
    for room in range(minRoom, maxRoom + 1):
        for machine in range(minMachine, maxMachine + 1):
            machine = machineName%(room,machine);
            live = os.system("ping -c 1 " + machine)
            if(live == 0):
                mList = mList + [machine];
    return mList;        
    
if(__name__ == '__main__'):
    l = getMachines()
    l = "\n".join(l)
    f = open("hosts", "w");
    print l
    f.write(l)
        
    
