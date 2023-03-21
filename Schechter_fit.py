 #!/usr/bin/env python3

import numpy as np
import csv

M_guess = -21
phi_guess = -3.
alpha_guess = -1.5

coeff = 0.01

box_size = 15 # Mpc
h = 0.7

def main():

    a = read_array()
    print('I found %d galaxies in the list.' % len(a))
    b = lum_to_muv(a)

    hist, bin_edges = np.histogram(b, range=(b.min(), -16), bins=20)
    bin_width = bin_edges[1] - bin_edges[0]
    x = bin_edges + 0.5 * bin_width
    x = x[:-1]
    y = hist / (box_size / h) ** 3 / bin_width
    print(x)
    print(y)

    alpha_best, M_best, phi_best = find_best_fit(x, y)
    print('The derived best-fit parameter values: alpha = %g, M = %g and phi=%g' % (alpha_best, M_best, phi_best))

def lum_to_muv(a):

    b = -26.83 - 2.5 * np.log10(np.power(10, a) / 1.22e13);
    return b

def find_best_fit(x, y):

    alpha = alpha_guess
    M = M_guess
    phi = phi_guess

    chi = 1.
    chi_new = loss_func(x, y, alpha, M, phi)

    while chi_new < chi :
        print("Chi = %g, alpha = %g, M = %g, phi=%g" % (chi_new, alpha, M, phi))
        alpha -= coeff * diff_loss_alpha(x, y, alpha, M, phi)
        M -= coeff * diff_loss_M(x, y, alpha, M, phi)
        phi -= coeff * diff_loss_phi(x, y, alpha, M, phi)
        chi = chi_new
        chi_new = loss_func(x, y, alpha, M, phi)
        print(schechter_func(-20, alpha, M, phi))

    return alpha, M, phi

def schechter_func(x, alpha, M, phi):

    a = 0.4 * np.log(10) * np.power(10, phi)
    b = np.power(10, 0.4 * (alpha + 1) * (M - x))
    c = np.exp(-np.power(10, 0.4 * (M - x)))

    return a * b * c

def loss_func(x , y, alpha, M, phi):

    z = np.sum((schechter_func(x, alpha, M, phi) - y) ** 2)
    z /= (1. * len(x))

    return z

def diff_sch_alpha(x, alpha, M, phi):

    z = np.log(10) * 0.4 * (M - x) * schechter_func(x, alpha, M, phi)
    return z

def diff_sch_M(x, alpha, M, phi):

    z = 0.4 * np.log(10) * ((alpha + 1) - np.power(10, 0.4 * M)) * schechter_func(x, alpha, M, phi)
    return z

def diff_sch_phi(x, alpha, M, phi):

    z = schechter_func(x, alpha, M, phi) * np.log(10)
    return z

def diff_loss_alpha(x, y, alpha, M, phi):

    z = 2. * (schechter_func(x, alpha, M, phi) - y) * diff_sch_alpha(x, alpha, M, phi) + y ** 2
    z = np.sum(z) / (1. * len(x))
    return z

def diff_loss_phi(x, y, alpha, M, phi):

    z = 2. * (schechter_func(x, alpha, M, phi) - y) * diff_sch_phi(x, alpha, M, phi) + y ** 2
    z = np.sum(z) / (1. * len(x))
    return z

def diff_loss_M(x, y, alpha, M, phi):

    z = 2. * (schechter_func(x, alpha, M, phi) - y) * diff_sch_M(x, alpha, M, phi) + y ** 2
    z = np.sum(z) / (1. * len(x))
    return z

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
