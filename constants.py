import math
import numpy as np


SZ = np.array([[1, 0], [0, -1]])
SX = np.array([[0, 1], [1, 0]])
SY = np.array([[0, -1j], [1j, 0]])
ID = np.eye(2)
J = (np.kron(SY, SX) - np.kron(SX, SY)) / 2

TROTTER_1ST_ORDER = (1, 1)
TROTTER_2ND_ORDER = (1/2, 1, 1/2)

g = 0.75  # arbitrary parameter
TROTTER_3RD_ORDER = (
    1 - g,
    1 / (2 * g),
    g,
    1 - 1 / (2 * g)
)

TROTTER_4TH_ORDER = (
        (3 + 1j * math.sqrt(3)) / 12,
        (3 + 1j * math.sqrt(3)) / 6,
        1 / 2,
        (3 - 1j * math.sqrt(3)) / 6,
        (3 - 1j * math.sqrt(3)) / 12
)

TROTTER_4TH_ORDER_ = (
        (3 - 1j * math.sqrt(3)) / 12,
        (3 - 1j * math.sqrt(3)) / 6,
        1 / 2,
        (3 + 1j * math.sqrt(3)) / 6,
        (3 + 1j * math.sqrt(3)) / 12
)

"""
def trotter_coefficients_1st_order():
    a1 = 1
    b1 = 1
    return a1, b1


def trotter_coefficients_2nd_order():
    a1 = 1 / 2
    b1 = 1
    a2 = 1 / 2
    return a1, b1, a2


def trotter_coefficients_3rd_order(g=0.75):
    a1 = 1 - g
    b1 = 1 / (2 * g)
    a2 = g
    b2 = 1 - 1 / (2 * g)
    return a1, b1, a2, b2


def trotter_coefficients_4th_order():
    a1 = (3 + 1j * math.sqrt(3)) / 12  # a1 = (3 - 1j * math.sqrt(3)) / 12
    b1 = (3 + 1j * math.sqrt(3)) / 6  # b1 = (3 - 1j * math.sqrt(3)) / 6
    a2 = 1 / 2
    b2 = (3 - 1j * math.sqrt(3)) / 6  # b2 = (3 + 1j * math.sqrt(3)) / 6
    a3 = (3 - 1j * math.sqrt(3)) / 12  # a3 = (3 + 1j * math.sqrt(3)) / 12
    return a1, b1, a2, b2, a3
"""
