import sys
import numpy as np
import pulp as pl

import itertools

def ip_kozos(prob, P):
    '''
    max és sima feladat közös része
    '''
    n = len(P[0])
    V = [i for i in range(1, n+1)]
    x = pl.LpVariable.dicts("x", (V,V), cat = "Binary") #minden a,b veresenyzőpárra egy változó
    
    #egyenlőtlenségek
    for a in V:
        for b in V:
            if a != b:
                prob += x[a][b] + x[b][a] == 1 #egyértelmű sorrend 

    for a in V:
        for b in V:
            for c in V:
                if a != b and b != c and c != a:
                    prob += x[a][c] - x[a][b] - x[b][c] >= -1 #tranzitivitás
    return(n, V, x)

def megold(prob, V, x):
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    k = [1 for i in V]
    for i in V:
        for j in V:
            if i != j:
                k[i-1] += int(pl.value(x[j][i])) #hányan vannak előtte
    return(k, pl.value(prob.objective))

def sulyok(P):
    '''
      minden u,v versenyzőre kiszámolja hányszor van előbb u mint v
    '''
    n = len(P[0])
    w = np.zeros([n+1, n+1])
    for u in range(1, n+1):
        for v in range(1,n+1):
            w[u][v] = 0
            for p in P:
                if p[u-1] < p[v-1]: #ha adott bírónál u előbb mint v növelem eggyel
                    w[u][v] += 1
    return(w)

def ip(P):
    '''
      sima távolság szerinti IP
    '''
    w = sulyok(P)
    prob = pl.LpProblem(name = "min_ip",
                        sense = pl.LpMinimize)
    
    (n, V, x) = ip_kozos(prob, P)

    #célfüggvény
    prob += pl.lpSum([pl.lpSum([x[a][b]*w[b][a] for a in V]) for b in V])

    return(megold(prob, V, x))

def ip_max(P):
    '''
      max. min. szerinti IP
    '''
    prob = pl.LpProblem(name = "minmax_ip",
                        sense = pl.LpMinimize)
    
    k = len(P)
    (n, V, x) = ip_kozos(prob, P)
    y = pl.LpVariable("y", cat = "Integer") #max távolság változója

    #egyenlőtelnségek
    for i in range(k):
        prob += pl.lpSum([pl.lpSum([x[a][b]*(P[i][a-1] > P[i][b-1]) for a in V]) for b in V]) <= y

    #célfüggvény
    prob += y
    return(megold(prob, V, x))


P = [[int(j) for j in sys.argv[i][1:-1].split(',')] for i in range(1,len(sys.argv))]

print("szumma feladatra optimális permutáció: "+str(ip(P)[0]))
print("optimális távolság: "+str(ip(P)[1]))
print("max feladatra optimális permutáció: "+str(ip_max(P)[0]))
print("optimális távolság :"+str(ip_max(P)[1]))
