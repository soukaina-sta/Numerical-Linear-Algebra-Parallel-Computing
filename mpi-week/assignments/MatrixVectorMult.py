#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:20:22 2020

@author: kissami
"""
import numpy as np
from scipy.sparse import lil_matrix
from numpy.random import rand, seed
from numba import njit
from mpi4py import MPI


''' This program compute parallel csc matrix vector multiplication using mpi '''

COMM = MPI.COMM_WORLD
nbOfproc = COMM.Get_size()
RANK = COMM.Get_rank()

seed(42)

def matrixVectorMult(A, b, x):
    
    row, col = A.shape
    for i in range(row):
        a = A[i]
        for j in range(col):
            x[i] += a[j] * b[j]

    return x

########################initialize matrix A and vector b ######################
#matrix sizes
SIZE = 1000
Local_size = SIZE // nbOfproc

# counts = block of each proc
counts = [Local_size] * nbOfproc

if RANK == 0:
    A = lil_matrix((SIZE, SIZE))
    A[0, :100] = rand(100)
    A[1, 100:200] = A[0, :100]
    A.setdiag(rand(SIZE))
    A = A.toarray()
    b = rand(SIZE)
else :
    A = None
    b = None



#########Send b to all procs and scatter A (each proc has its own local matrix#####
LocalMatrix = np.zeros((Local_size, Local_size)) 
Localb = np.zeros(Local_size)
# Scatter the matrix A
displacements = [sum(counts[:i]) for i in range(nbOfproc)]
COMM.Scatterv([A,counts,displacements,MPI.DOUBLE],LocalMatrix,root =0)

#####################Compute A*b locally#######################################
LocalX = np.zeros(Local_size)

start = MPI.Wtime()
matrixVectorMult(LocalMatrix, b, LocalX)
stop = MPI.Wtime()
if RANK == 0:
    print("CPU time of parallel multiplication is ", (stop - start)*1000)

##################Gather te results ###########################################
# sendcouns = local size of result
sendcounts = [Local_size] * nbOfproc
sendcounts[-1] += SIZE * nbOfproc
if RANK == 0: 
     X = np.zeros(SIZE)
else :
     X = None

# Gather the result into X
COMM.Gatherv(LocalX,[X,sendcounts,displacements, MPI.DOUBLE],root = 0)

##################Print the results ###########################################

if RANK == 0 :
    X_ = A.dot(b)
    print("The result of A*b using dot is :", np.max(X_ - X))
    print("The result of A*b using parallel version is :", X)
    
