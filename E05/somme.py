import numpy as np
def somma_n(n):
    somma = 0
    for i in range(n+1):
        somma=somma+i
    print(somma)
    return somma
    

def somma_sqrtn(n):
    somma = 0
    for i in range(n+1):
        somma=somma+i**(1/2)
    print(somma)
    return somma
    

def somma_prod(n):
    somma = 0
    prod=1
    for i in range(n+1):
        somma=somma+i
    for i in range (1,n+1):
        prod=prod*i
    print(somma)
    print(prod)
    return somma, prod 

def sommatoria(n, a):
    somma=0
    for i in range(n+1):
        somma=somma+i**a
    print(somma)
    return somma

