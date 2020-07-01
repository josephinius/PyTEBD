"""

itebd.py

I use the same approach as in the C/C++ version:

https://github.com/josephinius/iTEBD/blob/master/main.cpp

- [..] TODO: implement adaptive tau for imaginary time evolution
- [..] TODO: create nice file output
- [ ] TODO: implement a fourth order Suzuki-Trotter expansion
- [ ] TODO: reproduce Fig 6 from https://arxiv.org/pdf/cond-mat/0605597.pdf
- [ ] TODO: add file description

Lapack_driver : {‘gesdd’, ‘gesvd’}, optional
Whether to use the more efficient divide-and-conquer approach ('gesdd')
or general rectangular approach ('gesvd') to compute the SVD.
MATLAB and Octave use the 'gesvd' approach. Default is 'gesdd'.


numpy constants:

https://numpy.org/devdocs/reference/constants.html

matrix exponentiation:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html


"""


import math
import numpy as np
from scipy import linalg
import constants
import time


EPS = 1.E-32
# EPS = 1.E-16

D = 2  # physical degrees of freedom

# SX = np.array([[0, 1], [1, 0]])
# SY = np.array([[0, -1j], [1j, 0]])
# SZ = np.array([[1, 0], [0, -1]])


def create_hamiltonian_bulk(g, bias=0):
    """Returns bulk hamiltonian for the wire junction problem"""
    v = - 2 * math.cos(math.pi / (2 * g))
    # return - 2 * (np.kron(SX, SX) + np.kron(SY, SY)) + v * np.kron(SZ, SZ)
    # + 1.E-10 * (np.kron(SZ, id) + np.kron(id, SZ)) / 2
    # id = np.eye(2)
    hopping = - (np.kron(constants.SX, constants.SX) + np.kron(constants.SY, constants.SY)) / 2
    interaction = v * np.kron(constants.SZ, constants.SZ) / 4
    potential = (bias / 2) * (np.kron(constants.SZ, constants.ID) + np.kron(constants.ID, constants.SZ)) / 4
    return hopping + interaction + potential


def ham_ising_test(h):
    # id = np.eye(2)
    return - np.kron(constants.SX, constants.SX) - h * (np.kron(constants.SZ, constants.ID) + np.kron(constants.ID, constants.SZ)) / 2
    # - 1.E-10 * (np.kron(SX, id) + np.kron(id, SX)) / 2


def create_hamiltonian_ising(h):
    """Returns quantum Ising model hamiltonian"""

    """
    hamiltonian = np.zeros((D * D, D * D))
    hamiltonian[0][0] = h
    hamiltonian[0][3] = 1.
    hamiltonian[1][2] = 1.
    hamiltonian[2][1] = 1.
    hamiltonian[3][0] = 1.
    hamiltonian[3][3] = -h
    """

    hamiltonian = np.array(
        [
            [h, 0, 0, 1.],
            [0, 0, 1., 0],
            [0, 1., 0, 0],
            [1., 0, 0, -h]
         ]
    )

    return hamiltonian


def create_hamiltonian_heisenberg(h):
    """Returns Heisenberg model hamiltonian"""

    """
    hamiltonian = np.zeros((D * D, D * D))
    hamiltonian[0][0] = 1. + h
    hamiltonian[1][1] = - 1.
    hamiltonian[1][2] = 2.
    hamiltonian[2][1] = 2.
    hamiltonian[2][2] = - 1.
    hamiltonian[3][3] = 1. - h
    """

    hamiltonian = np.array(
        [
            [1+h, 0,  0,   0],
            [0,  -1,  2,   0],
            [0,   2, -1,   0],
            [0,   0,  0, 1-h]
        ]
    )

    return hamiltonian


def create_u_gate_ising(h, tau):
    """ Returns u_gate for non-unitary evolution (in imaginary time) for quantum Ising model.
    Here u_gate is obtained by the first order Suzuki-Trotter expansion.
    """
    u_gate = np.zeros((D * D, D * D))
    s = math.sqrt(1 + h * h)
    u_gate[0][0] = (math.exp(s * tau) * (-h + s) - math.exp(-s * tau) * (-h - s)) / (2 * s)
    u_gate[0][3] = - math.sinh(s * tau) / s  # == (math.exp(-s * tau) - math.exp(s * tau)) / (2 * s);
    u_gate[1][1] = math.cosh(tau)
    u_gate[1][2] = - math.sinh(tau)
    u_gate[2][1] = - math.sinh(tau)
    u_gate[2][2] = math.cosh(tau)
    u_gate[3][0] = - math.sinh(s * tau) / s  # == (math.exp(-s * tau) - math.exp(s * tau)) / (2 * s)
    u_gate[3][3] = (math.exp(s * tau) * (h + s) - math.exp(-s * tau) * (h - s)) / (2 * s)
    return u_gate


def create_u_gate_heisenberg(h, tau):
    """ Returns u_gate for non-unitary evolution (in imaginary time) for Heisenberg model
        Here u_gate is obtained by the first order Suzuki-Trotter expansion.
"""
    u_gate = np.zeros((D * D, D * D))
    u_gate[0][0] = math.exp(-tau * (1 + h))
    u_gate[1][1] = math.exp(tau) * math.cosh(2 * tau)
    u_gate[1][2] = - math.exp(tau) * math.sinh(2 * tau)
    u_gate[2][1] = - math.exp(tau) * math.sinh(2 * tau)
    u_gate[2][2] = math.exp(tau) * math.cosh(2 * tau)
    u_gate[3][3] = math.exp(-tau * (1 - h))
    return u_gate


def initialize_gamma_ising():
    xi = 1
    gamma_a = np.zeros((D, xi, xi))
    gamma_a[0][0][0] = 1.
    # gamma_a[1][0][0] = 1.
    gamma_b = np.zeros((D, xi, xi))
    gamma_b[0][0][0] = 1.
    return gamma_a, gamma_b


def initialize_gamma_heisenberg():
    xi = 1
    gamma_a = np.zeros((D, xi, xi))
    gamma_a[0][0][0] = 1.
    gamma_a[1][0][0] = 1.
    gamma_b = np.zeros((D, xi, xi))
    gamma_b[0][0][0] = 1.
    # gamma_b[1][0][0] = 1.
    return gamma_a, gamma_b


def psi_expansion_slow(gamma_a, gamma_b, lambda_a, lambda_b):
    """Returns psi. This function can be used for debugging."""
    print("# Using incredibly slow psi expansion function...")
    xi, xi_p = lambda_b.shape[0], lambda_a.shape[0]
    psi = np.zeros((xi, D, D, xi))
    for a in range(xi):
        for i in range(D):
            for j in range(D):
                for c in range(xi):
                    psi[a][i][j][c] = \
                        sum(
                        lambda_b[a] * gamma_a[i][a][b] * lambda_a[b] * gamma_b[j][b][c] * lambda_b[c]
                        for b in range(xi_p)
                    )
    return psi


def trotter_coefficients_1st_order():
    a1 = 1
    b1 = 1
    return a1, b1


def trotter_coefficients_2nd_order():
    a1 = 1 / 2
    b1 = 1
    a2 = 1 / 2
    return a1, b1, a2


def trotter_coefficients_3rd_order(g=0.75):  # g is an arbitrary parameter
    a1 = 1 - g
    b1 = 1 / (2 * g)
    a2 = g
    b2 = 1 - 1 / (2 * g)
    return a1, b1, a2, b2


def trotter_coefficients_4th_order():  # TODO: just store those as constants in a separate file
    a1 = (3 + 1j * math.sqrt(3)) / 12  # a1 = (3 - 1j * math.sqrt(3)) / 12
    b1 = (3 + 1j * math.sqrt(3)) / 6  # b1 = (3 - 1j * math.sqrt(3)) / 6
    a2 = 1 / 2
    b2 = (3 - 1j * math.sqrt(3)) / 6  # b2 = (3 + 1j * math.sqrt(3)) / 6
    a3 = (3 - 1j * math.sqrt(3)) / 12  # a3 = (3 + 1j * math.sqrt(3)) / 12
    return a1, b1, a2, b2, a3


def save_array(np_array, file_name):
    np.save(file_name, np_array)


def load_array(file_name):
    return np.load(file_name)


def scan_time(mps, number_of_steps, trotter_order=4, sampling=1, file_name='data.txt'):

    # print file header
    f = open(file_name, 'w')
    f.write('# Model: %s\n' % mps.model)
    f.write('# Evolution mode: %s\n' % mps.evol_mode)
    f.write('# Order of the Trotter expansion: %d\n' % trotter_order)
    f.write('# D=%d, tau=%.10E, field=%.3E\n' % (mps.dim, mps.tau, mps.field))
    f.write('# time\t\tenergy\t\t\ts_z\t\t\ts_x\t\t\tentropy\n')
    f.close()

    mps.update_hamiltonian(mps.field)  # update the Hamiltonian
    gates = mps.create_gates(trotter_order)

    # TODO: create a neat "measurements" function
    mps.psi_expansion()
    energy = mps.energy_calculation(mps.hamiltonian)
    entropy_a, entropy_b = mps.entropy_calculation()
    mag_z = mps.magnetization_z_calculation()
    mag_x = mps.magnetization_x_calculation()

    t = 0
    f = open('data.txt', 'a')
    f.write('%.6f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n'
            % (t, np.real(energy), np.real(mag_z), np.real(mag_x), entropy_a, entropy_b, mps.tau))
    # TODO: write to the file also the order of the Trotter expansion
    f.close()

    energy = 0
    energy_mem = 1
    counter = 0

    mag_z = 0
    mag_z_mem = 1

    for iter_id in range(number_of_steps):

        """
        if counter > 256 and abs(energy_mem - energy) < 1.E-12 and abs(mag_z_mem - mag_z) < 1.E-14:  # adaptive tau
            if mps.tau > 1.E-32:
                mps.tau /= 2
                gates = mps.create_gates(trotter_order)
                # print('new tau', self.tau)
            counter = 0
        counter += 1
        """

        # mps.evolve_4th_order_1step(gates)
        mps.evolve_1step(gates)

        if iter_id % sampling == 0:
            mps.psi_expansion()
            energy_mem = energy
            energy = mps.energy_calculation(mps.hamiltonian)
            entropy_a, entropy_b = mps.entropy_calculation()
            mag_z_mem = mag_z
            mag_z = mps.magnetization_z_calculation()
            mag_x = mps.magnetization_x_calculation()
            print(mps.iter_counter, np.real(energy), np.real(mag_z), np.real(mag_x), entropy_a, entropy_b, mps.tau)

            # t = (iter_id+1) * mps.tau
            f = open('data.txt', 'a')
            # f.write('%.6f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n'
            #        % (t, np.real(energy), np.real(mag_z), np.real(mag_x), entropy_a, entropy_b, mps.tau))
            f.write('%d\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\t%.15f\n'
                    % (iter_id, np.real(energy), np.real(mag_z), np.real(mag_x), entropy_a, entropy_b, mps.tau))
            f.close()


class IMPS(object):

    def __init__(self, dim, model=None, field=None, tau=None, evol_mode=None, xi=1):

        # choose between 1st, 2nd, 3rd, or 4th order Trotter expansion
        # set the hamiltonian / model - and also u_gates somewhere
        # choose between real- and imaginary-time evolution
        # keep track of the number of iterations

        if model == 'Ising':
            initialize_gamma = initialize_gamma_ising
            hamiltonian_function = ham_ising_test  # create_hamiltonian_ising
        elif model == 'Heisenberg':
            initialize_gamma = initialize_gamma_heisenberg
            hamiltonian_function = create_hamiltonian_heisenberg
            # u_gate = create_u_gate_heisenberg(h, tau)
            ground_state_energy = 1 / 4 - math.log(2)  # Heisenberg model ground state energy
            print('Heisenberg model ground state energy for h=0:', ground_state_energy)
        elif model == 'Wire':
            initialize_gamma = initialize_gamma_heisenberg
            hamiltonian_function = create_hamiltonian_bulk
        else:
            initialize_gamma = None
            hamiltonian_function = None
            # raise ValueError('model should be specified - "Ising" or "Heisenberg"')

        self.model = model
        self.field = field
        self.hamiltonian_function = hamiltonian_function
        if hamiltonian_function is not None:
            self.hamiltonian = hamiltonian_function(field)
        self.tau = tau
        self.evol_mode = evol_mode

        self.dim = dim
        if initialize_gamma is not None:
            self.gamma_a, self.gamma_b = initialize_gamma()

        self.lambda_a, self.lambda_b = np.array((xi, )), np.array((xi, ))
        self.lambda_a[0], self.lambda_b[0] = 1., 1.
        self.psi = None
        self.iter_counter = 0

        # self.u_gate = self.construct_u_gate(self.tau)
        # self.u_gate_even = self.construct_u_gate(self.tau / 2)
        # self.u_gate_odd = self.construct_u_gate(self.tau)

    def save_mps(self, file_path=''):

        # file_path += f'/{self.model}/D{self.dim}'

        timestamp = f"{int(time.time())}"

        save_array(self.gamma_a, file_path + 'gamma_a_' + timestamp)
        save_array(self.gamma_b, file_path + 'gamma_b_' + timestamp)
        save_array(self.lambda_a, file_path + 'lambda_a_' + timestamp)
        save_array(self.lambda_b, file_path + 'lambda_b_' + timestamp)

        with open("itebd.log", "a") as f:
            f.write(f"{timestamp}, {self.model}, evol mode = {self.evol_mode}, h = {self.field}, tau = {self.tau}, "
                    f"D = {self.dim}, iter = {self.iter_counter}\n")

    def update_hamiltonian(self, field):
        self.field = field
        self.hamiltonian = self.hamiltonian_function(field)

    def construct_u_gate(self, time_step):
        if self.evol_mode == 'real':
            return linalg.expm(- 1j * time_step * self.hamiltonian)
        elif self.evol_mode == 'imaginary':
            return linalg.expm(- time_step * self.hamiltonian)
        else:
            raise ValueError('evol_mode should be specified - "real" or "imaginary"')

    def create_gates(self, trotter_order=4):
        if trotter_order == 4:
            coefficients = trotter_coefficients_4th_order()
        elif trotter_order == 3:
            coefficients = trotter_coefficients_3rd_order()
        elif trotter_order == 2:
            coefficients = trotter_coefficients_2nd_order()
        elif trotter_order == 1:
            coefficients = trotter_coefficients_1st_order()
        else:
            raise ValueError('Incorrect value for trotter order; supported values: 1st, 2nd, 3rd, 4th')
        return tuple(self.construct_u_gate(c * self.tau) for c in coefficients)

    def evolve_1step(self, gates):  # TODO: use this function as a general case
        for gate in gates:
            self.psi_expansion()
            self.mps_update(gate)
            self.normalize()
            self.parity_swap()
        if len(gates) % 2 == 1:
            self.parity_swap()
        self.iter_counter += 1

    """
    def evolve_2nd_order_1step(self):
        self.psi_expansion()
        self.mps_update(self.u_gate_even)
        self.normalize()
        self.parity_swap()

        self.psi_expansion()
        self.mps_update(self.u_gate_odd)
        self.normalize()
        self.parity_swap()

        self.psi_expansion()
        self.mps_update(self.u_gate_even)
        self.normalize()
        # self.psi_expansion()
        self.iter_counter += 1
    """

    """
    def evolve_1st_order_1step(self):
        for _ in range(2):
            self.psi_expansion()
            self.mps_update(self.u_gate)
            self.normalize()
            self.parity_swap()
        self.iter_counter += 1
    """

    def mps_evolve(self, steps, trotter_order=4):
        gates = self.create_gates(trotter_order)

        energy = 0
        energy_mem = 1

        counter = 0

        for _ in range(steps):

            """
            if counter > 256 and abs(energy_mem - energy) < 1.E-13:  # adaptive tau
                if self.tau > 1.E-32:
                    self.tau /= 2
                    if self.dim < 64:
                        self.dim *= 2
                    gates = self.create_gates(trotter_order)
                    # print('new tau', self.tau)
                counter = 0
            counter += 1
            """

            self.evolve_1step(gates)

            self.psi_expansion()

            energy_mem = energy
            energy = self.energy_calculation(self.hamiltonian)
            energy = np.real(energy)
            entropy_a, entropy_b = self.entropy_calculation()
            mag_z = self.magnetization_z_calculation()
            mag_z = np.real(mag_z)
            mag_x = self.magnetization_x_calculation()
            mag_x = np.real(mag_x)

            print(self.iter_counter, energy, mag_z, mag_x, entropy_a, entropy_b, self.tau)

    """
    def mps_evolve_4th_order(self, n):
        coefficients = trotter_coefficients_4th_order()
        gates = []
        for c in coefficients:
            gates.append(self.construct_u_gate(c * self.tau))
        for _ in range(n):
            self.evolve_1step(gates)
            self.psi_expansion()
            energy = self.energy_calculation(self.hamiltonian)
            energy = np.real(energy)
            entropy = self.entropy_calculation()
            mag_z = self.magnetization_z_calculation()
            mag_z = np.real(mag_z)
            mag_x = self.magnetization_x_calculation()
            mag_x = np.real(mag_x)
            print(self.iter_counter, energy, mag_z, mag_x, entropy, self.tau)
    """

    """
    def mps_evolve_3rd_order(self, n):
        g = 0.5  # arbitrary parameter
        u0 = self.construct_u_gate(self.tau * (1 - g))  # even
        u1 = self.construct_u_gate(self.tau / (2 * g))  # odd
        u2 = self.construct_u_gate(self.tau * g)  # even
        u3 = self.construct_u_gate(self.tau * (1 - (2 * g)))  # odd
        gates = [u0, u1, u2, u3]
        for _ in range(n):
            self.evolve_1step(gates)
            self.psi_expansion()
            energy = self.energy_calculation(self.hamiltonian)
            entropy = self.entropy_calculation()
            mag_z = self.magnetization_z_calculation()
            mag_x = self.magnetization_x_calculation()
            print(self.iter_counter, energy, mag_z, mag_x, entropy, self.tau)
    """

    """
    def mps_evolve_2nd_order(self, n):
        self.u_gate_even = self.construct_u_gate(self.tau / 2)
        self.u_gate_odd = self.construct_u_gate(self.tau)
        energy_mem = 0
        energy = -1
        i = 0
        for _ in range(n):
            if i > 100 and abs(energy_mem - energy) < 1.E-9:  # adaptive tau
                if self.tau < 1.E-14:
                    self.tau = 1.E-06
                    if self.dim < 64:
                        self.dim *= 2
                    else:
                        self.tau = 1.E-10
                else:
                    self.tau /= 2
                print('new tau', self.tau)
                i = 0
            i += 1
            self.evolve_1step()
            self.psi_expansion()
            energy_mem = energy
            energy = self.energy_calculation(self.hamiltonian)
            entropy = self.entropy_calculation()
            mag_z = self.magnetization_z_calculation()
            mag_x = self.magnetization_x_calculation()
            #if energy < ground_state_energy:
            #    print('energy calculated is too low', energy)
            #    break
            print(self.iter_counter, energy, mag_z, mag_x, entropy, self.tau)
    """

    """
    def mps_evolve_1st_order(self, n):
        self.u_gate = self.construct_u_gate(self.tau)
        for _ in range(n):
            self.evolve_1step()
            self.psi_expansion()
            energy = self.energy_calculation(self.hamiltonian)
            entropy = self.entropy_calculation()
            mag_z = self.magnetization_z_calculation()
            mag_x = self.magnetization_x_calculation()
            # if energy < ground_state_energy:
            #    print('energy calculated is too low', energy)
            #    break
            print(self.iter_counter, energy, mag_z, mag_x, entropy)
    """

    def parity_swap(self):
        # even/odd --> odd/even swap
        self.gamma_a, self.gamma_b = self.gamma_b, self.gamma_a
        self.lambda_a, self.lambda_b = self.lambda_b, self.lambda_a

    def psi_expansion(self):
        ga, gb = self.gamma_a, self.gamma_b
        la, lb = self.lambda_a, self.lambda_b

        # psi1 = psi_expansion_check(ga, gb, la, lb)

        x1 = ga * lb[None, :, None]
        x1 = x1 * la[None, None, :]
        x2 = gb * lb[None, None, :]
        self.psi = np.tensordot(x1, x2, axes=(2, 1))  # x1_{iab} * x2_{jbc} --> iajc
        self.psi = np.swapaxes(self.psi, 0, 1)  # self.psi = self.psi.transpose((1, 0, 2, 3))

        # self.psi = psi_expansion_slow(ga, gb, la, lb)

        # path = np.einsum_path('a,iab,b,jbc,c->aijc', lb, ga, la, gb, lb, optimize='optimal')[0]
        # self.psi = np.einsum('a,iab,b,jbc,c->aijc', lb, ga, la, gb, lb, optimize=path)
        # print('difference in psi')
        # print('\nmax difference in psi:', np.max(np.abs(self.psi - psi1)))

    def get_xi(self):
        return self.lambda_b.shape[0]

    def mps_update(self, u_gate):
        u = u_gate.reshape((D, D, D, D))  # u_{ij,kl}

        theta = np.tensordot(u, self.psi, axes=([2, 3], [1, 2]))  # theta_{ijac} = u_gate_{ijkl} * psi_{aklc}
        # print('->theta shape: ', theta.shape)
        theta = theta.transpose((2, 0, 1, 3))  # theta_{aijc}

        xi = self.get_xi()
        # print('theta.shape', theta.shape)
        theta = theta.reshape((xi * D, D * xi))  # theta_{ai, jc}
        # x, ss, y = linalg.svd(theta, lapack_driver='gesdd')  # x.shape should be (xi * D, D * xi)
        x, ss, y = linalg.svd(theta, lapack_driver='gesvd')  # x.shape should be (xi * D, D * xi)
        # x, ss, y = linalg.svd(theta)  # x.shape should be (xi * D, D * xi)

        xi_p = min(self.dim, D * xi)  # D if d * xi > D else d * xi

        lambda_a_new = []
        for s in ss[:xi_p]:
            if s < EPS:
                # print('s too small', s)
                break
            lambda_a_new.append(s)

        self.lambda_a = np.array(lambda_a_new)

        xi_p = self.lambda_a.shape[0]
        # print('xi_p', xi_p)
        # print(self.lambda_a)

        x = x[:, :xi_p]
        # print('identity test: ')
        # print(np.max(np.abs(np.identity(x.shape[0]) - np.tensordot(x, x, axes=(1, 1)))))

        # xx = np.tensordot(x, x, axes=(1, 1))
        # xx[np.abs(xx) < 1.E-10] = 0
        # for r in xx:
        #    print(r)

        y = y[:xi_p, :]
        # print(np.max(np.abs(np.identity(y.shape[1]) - np.tensordot(y, y, axes=(0, 0)))))

        # print('testing svd identity...')
        # t1 = x * ss[None, :xi_p]
        # t2 = np.tensordot(x, y * ss[:xi_p, None], axes=(1, 0))
        # print(np.max(np.abs(theta - t2)))

        x = x.reshape((xi, D, xi_p))
        self.gamma_a = x.transpose((1, 0, 2))
        y = y.reshape((xi_p, D, xi))
        self.gamma_b = y.transpose((1, 0, 2))

        # print(self.lambda_b)
        # print(self.lambda_a)

        self.gamma_a = self.gamma_a / self.lambda_b[None, :, None]
        # for i in range(D):
        #    for a in range(self.lambda_b.shape[0]):
        #        for b in range(self.lambda_a.shape[0]):
        #            self.gamma_a[i][a][b] = self.gamma_a[i][a][b] / self.lambda_b[a]

        self.gamma_b = self.gamma_b / self.lambda_b[None, None, :]
        # for i in range(D):
        #    for a in range(self.lambda_a.shape[0]):
        #        for b in range(self.lambda_b.shape[0]):
        #            self.gamma_b[i][a][b] = self.gamma_b[i][a][b] / self.lambda_b[b]

    def normalize(self):
        self.lambda_a = self.lambda_a / self.get_norm(self.lambda_a)
        # self.lambda_b = self.lambda_b / self.get_norm(self.lambda_b)

    @staticmethod
    def get_norm(lam):
        norm = np.tensordot(lam, lam, axes=(0, 0))
        return math.sqrt(norm)

    def psi_norm(self):

        """
        xi = self.psi.shape[0]
        result = 0
        for a in range(xi):
            for i in range(D):
                for j in range(D):
                    for c in range(xi):
                        result += self.psi[a][i][j][c] * self.psi[a][i][j][c]
        return result
        """
        return np.tensordot(self.psi, np.conj(self.psi), axes=([0, 1, 2, 3], [0, 1, 2, 3]))
        # return np.tensordot(self.psi, self.psi, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

    def energy_calculation(self, hamiltonian):
        ham_tensor = hamiltonian.reshape((D, D, D, D))
        # print('ham.shape', ham_tensor.shape)
        # print('psi shape', self.psi.shape)
        temp = np.tensordot(self.psi, ham_tensor, axes=([1, 2], [0, 1]))  # temp_{acij} = psi_{aklc} * ham_{klij}
        # print('temp shape', temp.shape)

        # result = np.tensordot(self.psi, temp, axes=([0, 1, 2, 3], [0, 2, 3, 1]))
        # energy = psi_{aijc} * temp_{acij}
        result = np.tensordot(np.conj(self.psi), temp, axes=([0, 1, 2, 3], [0, 2, 3, 1]))

        # print('psi norm', self.psi_norm())
        if self.model == 'Heisenberg':
            return result / (4 * self.psi_norm())
        return result / self.psi_norm()

    def magnetization_z_calculation(self):
        lam_a2 = np.einsum('i,i->i', self.lambda_a, self.lambda_a)
        lam_b2 = np.einsum('i,i->i', self.lambda_b, self.lambda_b)
        x = self.gamma_a * lam_b2[None, :, None]
        y = self.gamma_a * lam_a2[None, None, :]
        norm = np.tensordot(x, np.conj(y), axes=([0, 1, 2], [0, 1, 2]))
        # op = np.array([1, -1])
        # opx = x * op[:, None, None]
        # op = np.array([[1, 0], [0, -1]])
        op = constants.SZ
        opx = np.tensordot(op, x, axes=(1, 0))
        return np.tensordot(opx, np.conj(y), axes=([0, 1, 2], [0, 1, 2])) / norm

    def magnetization_x_calculation(self):
        lam_a2 = np.einsum('i,i->i', self.lambda_a, self.lambda_a)
        lam_b2 = np.einsum('i,i->i', self.lambda_b, self.lambda_b)
        x = self.gamma_a * lam_b2[None, :, None]
        y = self.gamma_a * lam_a2[None, None, :]
        norm = np.tensordot(x, np.conj(y), axes=([0, 1, 2], [0, 1, 2]))
        # op = np.array([[0, 1], [1, 0]])  # Pauli sigma_x
        op = constants.SX
        opy = np.tensordot(op, y, axes=(1, 0))
        return np.tensordot(np.conj(x), opy, axes=([0, 1, 2], [0, 1, 2])) / norm

    def entropy_calculation(self):
        lam_2 = np.einsum('i,i->i', self.lambda_a, self.lambda_a)
        lam_2log = np.log(lam_2)  # natural logarithm
        entropy_a = - np.dot(lam_2, lam_2log)

        lam_2 = np.einsum('i,i->i', self.lambda_b, self.lambda_b)
        lam_2log = np.log(lam_2)  # natural logarithm
        entropy_b = - np.dot(lam_2, lam_2log)

        return entropy_a, entropy_b


# print('\nh:', h, 'tau:', tau, '\n')
np.set_printoptions(precision=10)  # https://docs.scipy.org/doc/numpy/reference/generated/numpy.set_printoptions.html


h = 1.0
tau = 0.00001
bond_dim = 128

# my_mps = IMPS(bond_dim, model)
# u_gate2 = linalg.expm(-tau * hamiltonian)
# print('\nmax difference in exp:', np.max(np.abs(u_gate - u_gate2)))

"""
for i in range(10):
    for _ in range(2):
        my_mps.psi_expansion()
        my_mps.mps_update(u_gate)
        my_mps.normalize()
        my_mps.parity_swap()

    my_mps.psi_expansion()
    # print('max gamma_a', np.max(my_mps.gamma_a))
    # print('max gamma_b', np.max(my_mps.gamma_b))
    # print('psi norm', my_mps.psi_norm())
    energy = my_mps.energy_calculation(hamiltonian)

    if energy < ground_state_energy:
        print('energy calculated is too low', energy)
        break
    print(i, energy)
"""


def main():
    # TODO: prepare GS for different models and parameters as required - store them in appropriate way
    pass


if __name__ == '__main__':
    main()

"""
u_gate = create_u_gate_ising(h, tau)
print('u_gate (method 1):\n\n', u_gate)
u_gate2 = linalg.expm(-tau * create_hamiltonian_ising(h))
print('u_gate (method 2):\n\n', u_gate2)
"""

"""
# u_gate = create_u_gate_heisenberg(h, tau)
u_gate = create_u_gate_heisenberg(h, tau)
print('u_gate (method 1):\n\n', u_gate)
u_gate2 = linalg.expm(-tau * create_hamiltonian_heisenberg(h))
print('u_gate (method 2):\n\n', u_gate2)
print('\nmax difference in exp:', np.max(np.abs(u_gate - u_gate2)))

xi = 1  # Schmidt rank

# Initializations

gamma_a, gamma_b = gamma_heisenberg_init()

print('gamma_a\n')
print(gamma_a)
print('gamma_b\n')
print(gamma_b)

lambda_a = np.array((xi,))
lambda_b = np.array((xi,))
lambda_a[0] = 1.
lambda_b[0] = 1.
print('lambda_a\n')
print(lambda_a)

print('psi check:\n')
print(psi_expansion_check(gamma_a, gamma_b, lambda_a, lambda_b))
"""

"""
print('gamma a\n')
print(my_mps.gamma_a)
print('lambda a\n')
print(my_mps.lambda_a)
print('gamma b\n')
print(my_mps.gamma_b)
print('lambda b\n')
print(my_mps.lambda_b)
"""


"""
a = np.array([[1, 2], [3, 4]])
print(a)
b = np.array([[1, 1], [2, 2]])
print(b)
print(np.einsum('ij, ij -> i', a, b))
"""
