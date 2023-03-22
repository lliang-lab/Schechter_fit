 #!/usr/bin/env python3

import numpy as np
from numpy import unravel_index
import csv

nbin = 100
box_size = 15 # Mpc
h = 0.7

r = 2.5
interval = 0.01

def main():

    a = read_array()
    print('I found %d galaxies in the list.' % len(a))
    b = lum_to_muv(a)

    hist, bin_edges = np.histogram(b, range=(b.min(), -16), bins=20)
    bin_width = bin_edges[1] - bin_edges[0]
    x = bin_edges + 0.5 * bin_width
    x = x[:-1]
    y = hist #/ (box_size / h) ** 3 / bin_width
    print(x) # UV magnitude bins

    pdf = [[[0 for k in range(nbin)] for j in range(nbin)] for i in range(nbin)]
    pdf = np.array(pdf)

    max = -1.
    best_values = [0.,0.,0.]
    for k in range(nbin):
        phi = phi_min + interval * k
        for j in range(nbin):
            M = M_min + interval * j
            for i in range(nbin):
                alpha = alpha_min + interval * i
                s  = schechter_func(x, alpha, M, phi)
                s *= (box_size / h) ** 3 * bin_width # expected number of galaxies in the bin
                pdf[k][j][i] = likelihood(s, y)
                if likelihood(s, y) > max:
                    max = likelihood(s, y)
                    best_values = [alpha, M, phi]

    print(best_values)

def schechter_func(x, alpha, M, phi):

    a = 0.4 * np.log(10) * np.power(10, phi)
    b = np.power(10, 0.4 * (alpha + 1) * (M - x))
    c = np.exp(-np.power(10, 0.4 * (M - x)))

    return a * b * c

def likelihood(s, y):

    p = 1.
    for i in range(len(s)):
        p *= poisson(s[i], y[i])
    return p

def poisson(r, n):
    # r - expected number of galaxies
    # n - observed number of galaxies
    z = np.power(r, n) * np.exp(-r) / np.math.factorial(n)
    return z

def lum_to_muv(a):

    b = -26.83 - 2.5 * np.log10(np.power(10, a) / 1.22e13);
    return b

def read_array():

    with open('FB15N2048_bpass_dz4_104_LUV.csv', 'r') as file:
        csvreader = csv.reader(file)
        a = []
        count = 0
        for row in csvreader:
            if count > 0:
                a.append(float(row[1]))
            count += 1
        return a

main()
