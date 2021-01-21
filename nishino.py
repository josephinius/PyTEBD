import math
import numpy as np
from scipy import linalg
import constants
import copy
import itebd
import time
import pickle
from tqdm import tqdm

EPS = 1.E-16  # 1.E-32
D = 2  # physical dimension

"""

Example

N=10

gammas: L_a, L_b, A, A, A, B, B, B, R_a, R_b = gamma0, gamma1, ..., gamma9
lambdas: l0, l1, l2, ..., l9, l10

        |        |        |        |        |        |        |        |        |        |
---<>---O---<>---O---<>---O---<>---O---<>---O---<>---O---<>---O---<>---O---<>---O---<>---O---<>---  

       L_a      L_b       A        A        A        B        B        B       R_a       R_b
       
   l0       l1       l2       l3       l4        l5      l6       l7       l8       l9       l10
       
"""


def create_hamiltonian_link(hopping_amplitude, bias_left=0, bias_right=-0):
    hopping = - hopping_amplitude * (np.kron(constants.SX, constants.SX) + np.kron(constants.SY, constants.SY)) / 2
    # potential = (bias / 2) * (np.kron(constants.SZ, constants.ID) - np.kron(constants.ID, constants.SZ)) / 4
    left_potential = (bias_left / 2) * np.kron(constants.SZ, constants.ID) / 4
    right_potential = (bias_right / 2) * np.kron(constants.ID, constants.SZ) / 4
    return hopping + left_potential + right_potential


def set_trotter_coefficients(trotter_order):
    coefficients = None
    if trotter_order == 4:
        coefficients = constants.TROTTER_4TH_ORDER
    elif trotter_order == 3:
        coefficients = constants.TROTTER_4TH_ORDER
    elif trotter_order == 2:
        coefficients = constants.TROTTER_4TH_ORDER
    elif trotter_order == 1:
        coefficients = constants.TROTTER_4TH_ORDER
    return coefficients


def u_gate_factory(evol_mode):
    def wrapper(step_size, hamiltonian):
        u_gate = None
        if evol_mode == 'real':
            u_gate = linalg.expm(- 1j * step_size * hamiltonian)
        elif evol_mode == 'imaginary':
            u_gate = linalg.expm(- step_size * hamiltonian)
        return u_gate.reshape((D, D, D, D))  # u_{ij,kl}
    return wrapper


def psi_expansion(gamma_a, gamma_b):
    psi = np.tensordot(gamma_a, gamma_b, axes=(2, 1))  # gamma_a_{iab} * gamma_b_{jbc} --> iajc
    return np.swapaxes(psi, 0, 1)  # self.psi = self.psi.transpose((1, 0, 2, 3))


def create_theta(gamma_a, gamma_b, u_gate):
    psi = psi_expansion(gamma_a, gamma_b)
    # print('u_gate.shape', u_gate.shape)
    # print('psi.shape', psi.shape)
    theta = np.tensordot(u_gate, psi, axes=([2, 3], [1, 2]))  # theta_{ijac} = u_gate_{ijkl} * psi_{aklc}
    theta = theta.transpose((2, 0, 1, 3))  # theta_{aijc}
    return theta


def get_norm(lam):
    return np.sum(lam)  # np.tensordot(lam, lam, axes=(0, 0))


def entropy_calculation(sing_values):
    # sing_values_squared = np.einsum('i,i->i', sing_values, sing_values)
    # log_sing_values_squared = np.log(sing_values_squared)  # natural logarithm
    log_sing_values = np.log(sing_values)  # natural logarithm
    return - np.dot(sing_values, log_sing_values)


def write_profile_for_GS(file_name, step, tau, data_list):
    f = open(file_name, 'a')
    f.write('%d\t' % step)  # time is a global variable here
    f.write('%.15f\t' % tau)  # time is a global variable here
    for data in data_list:
        f.write('%.15f\t' % data)
    f.write('\n')
    f.close()


def write_profile(file_name, t, data_list):
    f = open(file_name, 'a')
    f.write('%.8f\t' % t)  # TODO: time is a global variable here - change it
    for data in data_list:
        f.write('%.15f\t' % data)
    f.write('\n')
    f.close()


def equal_list_eps(list1, list2, eps=1.E-16):
    rv = True
    for a, b in zip(list1, list2):
        if abs(a - b) > eps:
            return False
    return True


def load_mps(model_id):
    file_name = 'nishino_' + model_id + '.p'
    with open('nishino.log') as f:
        for line in f.readlines():
            if model_id in line:
                print('Initializing Nishino with parameters:')
                print(line)
                break
        else:
            raise ValueError(f'mode_id {model_id} not found')
    with open(file_name, 'rb') as f:
        return pickle.load(f)  # returns NISHINO object


class NISHINO(object):

    def __init__(self, model, size, dim, dim_itebd, field, tau, evol_mode, trotter_order=4, model_id=None, xi=1):

        # choose between 1st, 2nd, 3rd, or 4th order Trotter expansion
        # set the hamiltonian / model - and also u_gates somewhere
        # choose between real- and imaginary-time evolution
        # keep track of the number of iterations

        if model == 'Ising':
            # initialize_gamma = initialize_gamma_ising
            hamiltonian_function = itebd.ham_ising_test  # create_hamiltonian_ising
        elif model == 'Heisenberg':
            # initialize_gamma = initialize_gamma_heisenberg
            hamiltonian_function = itebd.create_hamiltonian_heisenberg
            # u_gate = create_u_gate_heisenberg(h, tau)
            # ground_state_energy = 1 / 4 - math.log(2)  # Heisenberg model ground state energy
            # print('Heisenberg model ground state energy for h=0:', ground_state_energy)
        elif model == 'Wire':
            # initialize_gamma = initialize_gamma_heisenberg
            # hamiltonian_function = itebd.create_hamiltonian_bulk
            hamiltonian_function = None
            """
            Notice that the ground state should be found first without any bias;
            the real-time evolution is studied after applying bias.
            """
            self.bias1 = 0.  # note that the same bias1 should be applied inside of the left lead
            self.bias2 = -0.  # note that the same bias2 should be applied inside of the right lead
            self.g1 = field  # 0.5  # also for the left lead - Luttinger parameter (must match the init from itebd)
            self.g2 = field  # 0.5  # also for the right lead - Luttinger parameter (must match the init from itebd)
            self.t_hopping = 0.4  # hopping amplitude at the center
        else:
            raise ValueError('model should be specified - "Ising", "Heisenberg", or "Wire"')

        self.model = model
        self.size = size  # window size
        self.field = field
        self.iter_counter = 0
        self.hamiltonian_function = hamiltonian_function

        self.tau = tau
        self.evol_mode = evol_mode
        self.trotter_order = trotter_order

        self.dim = dim
        self.dim_itebd = dim_itebd  # bond dimension outside of the window

        # "leads"
        self.imps_left = None
        self.imps_right = None
        # window
        self.gammas = [None] * self.size
        self.lambdas = [None] * (self.size - 1)

        self.hamiltonians = []

        self.trotter_coefficients = set_trotter_coefficients(trotter_order)
        self.gates = [None] * (self.size + 3)

        self.model_id = model_id
        self.initialize_mps_by_itebd()

        self.initialize_gates()
        # self.initialize_damping_gates(3)
        self.errors = [None] * (self.size - 1)  # truncation errors inside of window

    def initialize_damping_gates(self, margin=1):  # this seems to be a terrible idea

        assert self.evol_mode == 'real'
        assert margin >= 1

        # gates = [None] * (self.size + 3)
        create_u_gate = u_gate_factory(self.evol_mode)

        for position, ham in enumerate(self.hamiltonians):

            angle = 0

            if position <= margin:
                angle = 1. if position == 0 else (margin - position + 1) / margin
            elif position >= self.size + 2 - margin:
                angle = 1. if position == (self.size + 2) else (position - (self.size + 2 - margin - 1)) / margin
            angle *= math.pi / 2
            dt = self.tau * (math.cos(angle) - 1j * math.sin(angle))

            print('pos', position, 'angle', angle, 'dt', dt)

            if position == 0 or position == self.size + 2:  # leftmost or rightmost gates
                self.gates[position] = tuple(create_u_gate(dt * c, ham) for c in self.trotter_coefficients)
            else:
                if position % 2 == 0:
                    self.gates[position] = tuple(create_u_gate(dt * c, ham) for c in self.trotter_coefficients[::2])
                else:
                    self.gates[position] = tuple(create_u_gate(dt * c, ham) for c in self.trotter_coefficients[1::2])
        # return gates

    def initialize_gates_wire(self):

        ham_left = itebd.create_hamiltonian_bulk(self.g1, self.bias1)
        ham_mid = create_hamiltonian_link(self.t_hopping, self.bias1, self.bias2)
        ham_right = itebd.create_hamiltonian_bulk(self.g2, self.bias2)

        create_u_gate = u_gate_factory(self.evol_mode)
        mid = self.size // 2 + 1

        for pos in range((self.size + 3)):

            if pos < mid:
                ham = ham_left
            elif pos == mid:
                ham = ham_mid
            elif pos > mid:
                ham = ham_right

            if pos == 0 or pos == self.size + 2:  # leftmost or rightmost gates
                self.gates[pos] = tuple(create_u_gate(self.tau * c, ham) for c in self.trotter_coefficients)
            else:
                if pos % 2 == 0:
                    self.gates[pos] = tuple(create_u_gate(self.tau * c, ham) for c in self.trotter_coefficients[::2])
                else:
                    self.gates[pos] = tuple(create_u_gate(self.tau * c, ham) for c in self.trotter_coefficients[1::2])

    def initialize_gates_for_hamiltonians(self):
        create_u_gate = u_gate_factory(self.evol_mode)
        for pos, ham in enumerate(self.hamiltonians):
            if pos == 0 or pos == self.size + 2:  # leftmost or rightmost gates
                self.gates[pos] = tuple(create_u_gate(self.tau * c, ham) for c in self.trotter_coefficients)
            else:
                if pos % 2 == 0:
                    self.gates[pos] = tuple(create_u_gate(self.tau * c, ham) for c in self.trotter_coefficients[::2])
                else:
                    self.gates[pos] = tuple(create_u_gate(self.tau * c, ham) for c in self.trotter_coefficients[1::2])

    def initialize_uniform_hamiltonians(self, hamiltonian_function):
        hamiltonian = hamiltonian_function(self.field)
        for _ in range(self.size + 3):
            self.hamiltonians.append(copy.deepcopy(hamiltonian))

    def initialize_gates(self):
        if self.model == 'Wire':
            """List of hamiltonians is not initialized in the case of model == 'Wire';
            and the hamiltonians are calculated inside of the method initialize_gates_wire()"""
            self.initialize_gates_wire()  # initialization for the wire junction problem
        else:
            # initialization for uniform hamiltonians
            self.initialize_uniform_hamiltonians(self.hamiltonian_function)
            self.initialize_gates_for_hamiltonians()

    def initialize_mps_by_itebd(self):
        # load iMPS
        print(f'Loading iMPS of model_id {self.model_id}\n')

        # TODO: check the log file and print out the parameters! 

        time_stamp_init = self.model_id
        gamma_a = itebd.load_array('gamma_a' + '_' + time_stamp_init + '.npy')
        print('gamma_a.shape:', gamma_a.shape)
        gamma_b = itebd.load_array('gamma_b' + '_' + time_stamp_init + '.npy')
        print('gamma_b.shape:', gamma_b.shape)
        lambda_a = itebd.load_array('lambda_a' + '_' + time_stamp_init + '.npy')
        print('lambda_a.shape:', lambda_a.shape)
        lambda_b = itebd.load_array('lambda_b' + '_' + time_stamp_init + '.npy')
        print('lambda_b.shape:', lambda_b.shape)

        # left iMPS
        self.imps_left = itebd.IMPS(self.dim_itebd)
        self.imps_left.gamma_a = copy.deepcopy(gamma_a)
        self.imps_left.gamma_b = copy.deepcopy(gamma_b)
        self.imps_left.lambda_a = copy.deepcopy(lambda_a)
        self.imps_left.lambda_b = copy.deepcopy(lambda_b)

        # right iMPS
        self.imps_right = itebd.IMPS(self.dim_itebd)
        self.imps_right.gamma_a = copy.deepcopy(gamma_a)
        self.imps_right.gamma_b = copy.deepcopy(gamma_b)
        self.imps_right.lambda_a = copy.deepcopy(lambda_a)
        self.imps_right.lambda_b = copy.deepcopy(lambda_b)

        # window
        self.gammas = [None] * self.size
        self.lambdas = [None] * (self.size - 1)
        gamma_pair = (gamma_a, gamma_b)
        lambda_pair = (lambda_a, lambda_b)
        for pos in range(self.size):
            self.gammas[pos] = copy.deepcopy(gamma_pair[pos % 2])
        for pos in range(self.size - 1):
            self.lambdas[pos] = copy.deepcopy(lambda_pair[pos % 2])
        # A
        self.gammas[0] = self.gammas[0] * lambda_b[None, :, None]
        # B
        self.gammas[-1] = self.gammas[-1] * lambda_b[None, None, :]

    def mps_update(self, gamma_a, gamma_b, u_gate):

        # print('gamma_a.shape', gamma_a.shape)
        # print('gamma_b.shape', gamma_b.shape)

        # xi = psi.shape[0]
        xi_left = gamma_a.shape[1]
        xi_right = gamma_b.shape[2]
        # assert xi == psi.shape[-1]
        assert D == gamma_a.shape[0]
        assert D == gamma_b.shape[0]
        # assert xi == gamma_b.shape[2]
        assert gamma_a.shape[2] == gamma_b.shape[1]

        theta = create_theta(gamma_a, gamma_b, u_gate)
        theta = theta.reshape((xi_left * D, D * xi_right))  # theta_{ai, jc}
        x, ss, y = linalg.svd(theta, lapack_driver='gesvd', full_matrices=False)  # x.shape should be (xi * D, D * xi)
        xi_p = min(self.dim, D * min(xi_left, xi_right))  # D if d * xi > D else d * xi
        lambda_a_new = []

        for s in ss[:xi_p]:
            if s < EPS:
                # print('singular value too small')
                break
            lambda_a_new.append(s)

        ss_sum = sum(ss)
        error = (ss_sum - sum(lambda_a_new)) / ss_sum

        lambda_a = np.array(lambda_a_new)
        xi_p = lambda_a.shape[0]
        # print('xi_p', xi_p)
        x = x[:, :xi_p]
        y = y[:xi_p, :]
        x = x.reshape((xi_left, D, xi_p))
        gamma_a = x.transpose((1, 0, 2))
        y = y.reshape((xi_p, D, xi_right))
        gamma_b = y.transpose((1, 0, 2))

        return gamma_a, gamma_b, lambda_a, error

    def window_layer_update(self, position, layer):

        # print('layer update position', position)
        # print('layer update layer', layer)

        gamma_a = self.gammas[position]
        if position > 0:
            gamma_a = gamma_a * self.lambdas[position - 1][None, :, None]

        gamma_a = gamma_a * self.lambdas[position][None, None, :]

        gamma_b = self.gammas[position + 1]
        if position < self.size - 2:
            gamma_b = gamma_b * self.lambdas[position + 1][None, None, :]

        gate = self.gates[position + 2][layer // 2]

        gamma_a, gamma_b, lambda_a, error = self.mps_update(gamma_a, gamma_b, gate)

        if layer // 2 == 0:
            self.errors[position] = error
        else:
            self.errors[position] += error

        # lambda_a = lambda_a / get_norm(lambda_a)  # normalization
        lambda_a = lambda_a / get_norm(lambda_a)  # normalization

        self.lambdas[position] = lambda_a

        self.gammas[position] = gamma_a
        if position > 0:
            self.gammas[position] = self.gammas[position] / self.lambdas[position - 1][None, :, None]

        self.gammas[position + 1] = gamma_b
        if position < self.size - 2:
            self.gammas[position + 1] = self.gammas[position + 1] / self.lambdas[position + 1][None, None, :]

    @staticmethod
    def imps_normalize(imps):
        la = imps.lambda_a
        imps.lambda_a = la / get_norm(la)

    def evolve_one_step(self):

        number_of_layers = len(self.gates[0])
        assert number_of_layers == len(self.gates[-1])

        for l in range(number_of_layers):
            if l % 2 == 0:  # "even" update
                self.imps_left.psi_expansion()
                self.imps_left.mps_update(self.gates[0][l])
                self.imps_normalize(self.imps_left)
                # self.imps_left.normalize()

                self.imps_right.psi_expansion()
                self.imps_right.mps_update(self.gates[-1][l])
                self.imps_normalize(self.imps_right)
                # self.imps_right.normalize()

                for position in range(0, self.size - 1, 2):
                    self.window_layer_update(position, l)
            else:  # "odd" update
                # (1) "left" update
                a_leftmost = self.gammas[0]  # current left-most A tensor
                left_b = self.imps_left.gamma_b * self.imps_left.lambda_a[None, :, None]
                theta_left = create_theta(left_b, a_leftmost, self.gates[1][l // 2])

                self.imps_left.parity_swap()
                self.imps_left.psi_expansion()
                self.imps_left.mps_update(self.gates[0][l])
                # self.imps_left.normalize()
                self.imps_normalize(self.imps_left)
                self.imps_left.parity_swap()

                left_b = self.imps_left.gamma_b * self.imps_left.lambda_a[None, :, None]
                self.gammas[0] = np.tensordot(np.conj(left_b), theta_left, axes=([0, 1], [1, 0]))
                self.gammas[0] = self.gammas[0].transpose((1, 0, 2))  # new a_leftmost

                # (2) "right" update
                b_rightmost = self.gammas[-1]  # current right-most B tensor
                right_a = self.imps_right.gamma_a * self.imps_right.lambda_a[None, None, :]
                theta_right = create_theta(b_rightmost, right_a, self.gates[-2][l // 2])

                self.imps_right.parity_swap()
                self.imps_right.psi_expansion()
                self.imps_right.mps_update(self.gates[-1][l])
                # self.imps_right.normalize()
                self.imps_normalize(self.imps_right)
                self.imps_right.parity_swap()

                right_a = self.imps_right.gamma_a * self.imps_right.lambda_a[None, None, :]
                self.gammas[-1] = np.tensordot(theta_right, np.conj(right_a), axes=([2, 3], [0, 2]))
                self.gammas[-1] = self.gammas[-1].transpose((1, 0, 2))  # new b_rightmost

                for position in range(1, self.size - 2, 2):
                    self.window_layer_update(position, l)

        self.iter_counter += 1

    def one_site_observation(self, position, observable):
        gamma = self.gammas[position]
        if position > 0:
            gamma = gamma * self.lambdas[position - 1][None, :, None]
        if position < self.size - 1:
            gamma = gamma * self.lambdas[position][None, None, :]
        temp = np.tensordot(observable, gamma, axes=(1, 0))
        norm = np.tensordot(np.conj(gamma), gamma, axes=([0, 1, 2], [0, 1, 2]))
        return np.tensordot(np.conj(gamma), temp, axes=([0, 1, 2], [0, 1, 2])) / norm

    def two_site_observation(self, position, observable=constants.J):  # be default, this is just local current

        assert position >= 0
        assert position < self.size - 1

        gamma1 = self.gammas[position]
        gamma2 = self.gammas[position + 1]

        if position > 0:
            gamma1 = gamma1 * self.lambdas[position - 1][None, :, None]
        gamma1 = gamma1 * self.lambdas[position][None, None, :]
        if position + 1 < self.size - 1:
            gamma2 = gamma2 * self.lambdas[position + 1][None, None, :]

        u_gate = observable.reshape((D, D, D, D))

        psi = psi_expansion(gamma1, gamma2)

        theta = np.tensordot(u_gate, psi, axes=([2, 3], [1, 2]))  # theta_{ijac} = u_gate_{ijkl} * psi_{aklc}
        theta = theta.transpose((2, 0, 1, 3))  # theta_{aijc}

        norm = np.tensordot(np.conj(psi), psi, axes=([0, 1, 2, 3], [0, 1, 2, 3]))

        return np.tensordot(np.conj(psi), theta, axes=([0, 1, 2, 3], [0, 1, 2, 3])) / norm

    def flip_x_excitation(self):
        n0 = self.size // 2
        self.gammas[n0] = np.tensordot(constants.SX, self.gammas[n0], axes=(1, 0))

    def domain_wall_excitation(self):
        n0 = self.size // 2
        for pos in range(n0):
            self.gammas[pos] = - np.tensordot(constants.SZ, self.gammas[pos], axes=(1, 0))
        self.imps_left.gamma_a = - np.tensordot(constants.SZ, self.imps_left.gamma_a, axes=(1, 0))
        self.imps_left.gamma_b = - np.tensordot(constants.SZ, self.imps_left.gamma_b, axes=(1, 0))

    def jordan_wigner_excitation(self):
        self.flip_x_excitation()
        self.domain_wall_excitation()

    @staticmethod
    def boundary_entropy(left, right):
        xi_left = left.shape[1]
        xi_right = right.shape[2]
        psi = psi_expansion(left, right)
        psi = psi.reshape((xi_left * D, D * xi_right))  # theta_{ai, jc}
        ss = linalg.svd(psi, lapack_driver='gesvd', full_matrices=False, compute_uv=False)
        # xi_p = min(self.dim, D * min(xi_left, xi_right))  # D if d * xi > D else d * xi
        lambda_new = []
        # for s in ss[:xi_p]:
        for s in ss:
            if s < EPS:
                # print('singular value too small')
                break
            lambda_new.append(s)
        return entropy_calculation(lambda_new / get_norm(lambda_new))

    def get_entropy_profile(self):
        result = []

        # left-side entropy
        xi_left = self.imps_left.gamma_b.shape[1]
        xi_right = self.gammas[0].shape[2]

        leftmost = self.gammas[0] * self.lambdas[0][None, None, :]
        left_b = self.imps_left.gamma_b * self.imps_left.lambda_a[None, :, None]

        left_boundary_entropy = NISHINO.boundary_entropy(left_b, leftmost)
        result.append(left_boundary_entropy)
        # print('left-boundary entropy', left_boundary_entropy)

        entropy_in_window_list = [entropy_calculation(lamb / get_norm(lamb)) for lamb in self.lambdas]
        result.extend(entropy_in_window_list)

        # right-side entropy
        xi_left = self.gammas[-1].shape[1]
        xi_right = self.imps_right.gamma_a.shape[2]

        rightmost = self.gammas[-1] * self.lambdas[-1][None, :, None]
        right_a = self.imps_right.gamma_a * self.imps_right.lambda_a[None, None, :]

        right_boundary_entropy = NISHINO.boundary_entropy(rightmost, right_a)
        result.append(right_boundary_entropy)
        # print('right-boundary entropy', right_boundary_entropy)

        return result

    def save_nishino_mps(self):

        timestamp = f"{int(time.time())}"
        file_name = 'nishino_' + timestamp + '.p'

        self.model_id = timestamp

        with open(file_name, "wb") as f:
            pickle.dump(self, f)

        with open('nishino.log', 'a') as f:
            f.write(f"{timestamp}, {self.model}, evol mode = {self.evol_mode}, N =  {self.size}, tau = {self.tau}, "
                    f"dim = {self.dim}, dim_itebd = {self.dim_itebd}, trotter order = {self.trotter_order}, ")
            if self.model == 'Wire':
                f.write(f"bias1 = {self.bias1}, bias2 = {self.bias2}, "
                        f"g1 = {self.g1}, g2 = {self.g2}, tau = {self.tau}, "
                        f"t_hopping = {self.t_hopping}, iter = {self.iter_counter}\n")

    # def evolve_nishino_mps(self, num_of_steps, reinit_gates=True, trotter_order=None):
    def evolve_nishino_mps(self, num_of_steps, reinit_gates=True):

        if reinit_gates:
            # self.trotter_order = trotter_order or self.trotter_order
            self.initialize_gates()

        for _ in tqdm(range(num_of_steps)):  # I use tqdm() just for testing
            # for _ in range(num_of_steps):
            self.evolve_one_step()

    def measurements(self):
        sz = np.real(list(self.one_site_observation(j, constants.SZ) for j in range(self.size)))
        sx = np.real(list(self.one_site_observation(j, constants.SX) for j in range(self.size)))
        current = np.real(list(self.two_site_observation(j) for j in range(self.size - 1)))
        entropy = self.get_entropy_profile()
        return sz, sx, current, entropy

    def write_output_header(self, file_name):
        with open(file_name, 'w') as f:
            f.write('# Model: %s, (initialization id: %s)\n' % (self.evol_mode, self.model_id))
            f.write('# Evolution mode: %s\n' % self.evol_mode)
            f.write('# Order of the Trotter expansion: %d\n' % self.trotter_order)
            f.write('# N=%d, D=%d, D_itebd=%d, field=%.3E\n' % (self.size, self.dim, self.dim_itebd, self.field))
            # f.write('# iter\t\ttau\t\tprofile\n')  # for GS
            # f.write('# time\t\tprofile\n')  # for real-time evol with fixed tau

    def find_ground_state(self, output=False, tau_start=0.01):

        """

        # A complete example use:

        import nishino
        mps = nishino.NISHINO('Wire', 28, 40, 40, 1., 10., 'imaginary', 2, '1575631675')
        # or
        mps = nishino.NISHINO('Wire', 24, 16, 16, 1., 10., 'imaginary', 2, '1575631675')
        # '1575631675' - is the iTEBD GS initialization for the Wire model I am currently testing
        mps.find_ground_state(True)

        """

        self.evol_mode = 'imaginary'
        self.tau = tau_start

        if output:

            entropy_file_name = 'GS_nishino_entropy.txt'
            mag_x_file_name = 'GS_nishino_sx.txt'
            mag_z_file_name = 'GS_nishino_sz.txt'

            self.write_output_header(entropy_file_name)
            self.write_output_header(mag_x_file_name)
            self.write_output_header(mag_z_file_name)

        refresh_rate = 100  # for output
        convergence_rate = 100

        # epsilon = 1.E-10
        # tau_min = 1.E-13
        epsilon = 1.E-6  # this value is just for testing
        tau_min = 1.E-10  # this value is just for testing

        step_counter = 0
        reinit_gates_flag = True

        sz = [1. for _ in range(self.size)]
        sx = [1. for _ in range(self.size)]
        entropy = [1. for _ in range(self.size + 1)]

        while True:

            self.evolve_nishino_mps(convergence_rate, reinit_gates=reinit_gates_flag)
            reinit_gates_flag = False

            step_counter += convergence_rate
            # t = step_counter * self.tau

            # isz_left = mps.imps_left.magnetization_z_calculation()
            # print('mag z left', isz_left)
            # isz_right = mps.imps_right.magnetization_z_calculation()
            # print('mag z right', isz_right)

            # isx_left = mps.imps_left.magnetization_x_calculation()
            # print('mag z left', isx_left)
            # isx_right = mps.imps_right.magnetization_x_calculation()
            # print('mag z right', isx_right)

            # entropy_left_even = entropy_calculation(mps.imps_left.lambda_a / get_norm(mps.imps_left.lambda_a))
            # entropy_left_odd = entropy_calculation(mps.imps_left.lambda_b / get_norm(mps.imps_left.lambda_b))
            # print('left entropy even', entropy_left_even, 'odd', entropy_left_odd)
            # entropy_right_even = entropy_calculation(mps.imps_right.lambda_a / get_norm(mps.imps_right.lambda_a))
            # entropy_right_odd = entropy_calculation(mps.imps_right.lambda_b / get_norm(mps.imps_right.lambda_b))
            # print('right entropy even', entropy_right_even, 'odd', entropy_right_odd)

            """
            for i, ten in enumerate(mps.gammas):
                print('i', i, 'tensor shape', ten.shape)
            """

            # print('left gamma_a shape', mps.imps_left.gamma_a.shape)
            # print('left gamma_b shape', mps.imps_left.gamma_b.shape)
            # print('right gamma_a shape', mps.imps_right.gamma_a.shape)
            # print('right gamma_a shape', mps.imps_right.gamma_b.shape)

            sz_new, sx_new, _, entropy_new = self.measurements()  # we don't really care about the current here

            if output and step_counter % refresh_rate == 0:
                write_profile_for_GS(mag_z_file_name, step_counter, self.tau, sz_new)
                write_profile_for_GS(mag_x_file_name, step_counter, self.tau, sx_new)
                write_profile_for_GS(entropy_file_name, step_counter, self.tau, entropy_new)

            # TODO: print some quantities measured in the center, and at the left and right boundaries

            sz_test = equal_list_eps(sz, sz_new, epsilon)
            sx_test = equal_list_eps(sx, sx_new, epsilon)
            entropy_test = equal_list_eps(entropy, entropy_new, epsilon)

            sz, sx, entropy = sz_new, sx_new, entropy_new

            if sz_test and sx_test and entropy_test:
                self.tau /= 2.
                if self.tau < tau_min:
                    break
                reinit_gates_flag = True

    def real_evolution(self, num_of_steps=100_000, tau=0.002, refresh=100):

        self.evol_mode = 'real'
        self.tau = tau

        self.initialize_gates()

        entropy_file_name = 'nishino_entropy.txt'
        mag_x_file_name = 'nishino_sx.txt'
        mag_z_file_name = 'nishino_sz.txt'
        current_file_name = 'nishino_current.txt'

        self.write_output_header(entropy_file_name)
        self.write_output_header(mag_x_file_name)
        self.write_output_header(mag_z_file_name)
        self.write_output_header(current_file_name)

        sz_new, sx_new, current, entropy_new = self.measurements()

        write_profile(mag_z_file_name, 0, sz_new)
        write_profile(mag_x_file_name, 0, sx_new)
        write_profile(entropy_file_name, 0, entropy_new)
        write_profile(current_file_name, 0, current)

        for step_counter in range(0, num_of_steps, refresh):

            self.evolve_nishino_mps(refresh, reinit_gates=False)
            t = (step_counter + refresh) * tau

            sz_new, sx_new, current, entropy_new = self.measurements()
            write_profile(mag_z_file_name, t, sz_new)
            write_profile(mag_x_file_name, t, sx_new)
            write_profile(entropy_file_name, t, entropy_new)
            write_profile(current_file_name, t, current)


def quench_problem():

    # mps = NISHINO(...)
    # also load itebd mps init

    # mps = NISHINO('Wire', 24, 16, 16, 1., 10., 'imaginary', 2, '1575631675')
    mps = NISHINO('Wire', 100, 8, 8, 1., 100., 'imaginary', 2, '1575631675')

    # '1575631675' - is the iTEBD GS initialization for the Wire model I am currently testing

    # mps.find_ground_state and log measurements
    mps.find_ground_state(True)

    # save the MPS GS for future reference and calculations
    mps.save_nishino_mps()  # model id is updated here
    mps_id = mps.model_id

    mps = load_mps(mps_id)

    # Quench: apply the bias voltage and switch to real-time evolution
    # TODO: test - perform real-time evolution without quench - the mps should not change much!
    mps.bias1 = 1.
    mps.bias2 = -1.
    mps.evol_mode = 'real'

    # real-time evolution and log measurements
    mps.real_evolution(num_of_steps=40_000, tau=0.002, refresh=100)

    # save the evolved MPS so that it can continue again later
    mps.save_nishino_mps()


if __name__ == '__main__':
    quench_problem()

"""
if __name__ == '__main__':

    entropy_file_name = 'entropy_profile_1.txt'  # TODO: prevent over-writing my data
    mag_x_file_name = 'mag_x_1.txt'
    mag_z_file_name = 'mag_z_1.txt'
    truncation_err_file_name = 'truncation_errors_1.txt'
    current_file_name = 'current_profile_1.txt'

    model = "Wire"  # "Ising"
    quench = False  # True
    DIM = 40  # 120  # max bond dimension
    DIM_iTEBD = 40
    tau = 0.02  # 0.002
    field = 1.  # 0.45  # this is Luttinger parameter in the case of the wire junction problem
    N = 20  # 10  # window size; has to be even
    trotter_order = 2
    # evolution_mode = 'imaginary'
    evolution_mode = 'real'  # 'real'
    refresh = 10  # 10

    assert N % 2 == 0
    assert N >= 4
    assert trotter_order in (1, 2, 3, 4)

    model_id = '1575631675'  # TODO: make the itebd initialization somehow nicer and more automatic

    mps = NISHINO(model, N, DIM, DIM_iTEBD, field, tau, evolution_mode, trotter_order, model_id)
    # mps.initialize_mps_by_itebd(model_id)

    t = 0

    f = open(entropy_file_name, 'w')
    f.write('# Evolution mode: %s\n' % evolution_mode)
    f.write('# Order of the Trotter expansion: %d\n' % trotter_order)
    f.write('# N=%d, D=%d, D_itebd=%d, tau=%.10E, field=%.3E\n' % (N, DIM, DIM_iTEBD, tau, field))
    f.write('# time\t\tentropy profile\n')
    f.close()
    # entropies = [entropy_calculation(s / get_norm(s)) for s in mps.lambdas]
    entropies = mps.get_entropy_profile()

    print('entropy', *entropies)
    write_profile(entropy_file_name, t, entropies)

    entropy_left_even = entropy_calculation(mps.imps_left.lambda_a / get_norm(mps.imps_left.lambda_a))
    entropy_left_odd = entropy_calculation(mps.imps_left.lambda_b / get_norm(mps.imps_left.lambda_b))
    print('left entropy even', entropy_left_even, 'odd', entropy_left_odd)
    entropy_right_even = entropy_calculation(mps.imps_right.lambda_a / get_norm(mps.imps_right.lambda_a))
    entropy_right_odd = entropy_calculation(mps.imps_right.lambda_b / get_norm(mps.imps_right.lambda_b))
    print('right entropy even', entropy_right_even, 'odd', entropy_right_odd)

    f = open(truncation_err_file_name, 'w')
    f.write('# Evolution mode: %s\n' % evolution_mode)
    f.write('# Order of the Trotter expansion: %d\n' % trotter_order)
    f.write('# N=%d, D=%d, D_itebd=%d, tau=%.10E, field=%.3E\n' % (N, DIM, DIM_iTEBD, tau, field))
    f.write('# time\t\ttruncation errors profile\n')
    f.close()

    op = constants.SZ
    mag_z_list = [mps.one_site_observation(position, op) for position in range(N)]
    mag_z_list = np.real(mag_z_list)
    print('<s_z>', *mag_z_list)

    op = constants.SX
    mag_x_list = [mps.one_site_observation(position, op) for position in range(N)]
    mag_x_list = np.real(mag_x_list)
    print('<s_x>', *mag_x_list)

    current_list = [mps.two_site_observation(pos) for pos in range(N-1)]
    current_list = np.real(current_list)
    print('J', *current_list)

    f = open(mag_x_file_name, 'w')
    f.write('# Evolution mode: %s\n' % evolution_mode)
    f.write('# Order of the Trotter expansion: %d\n' % trotter_order)
    # f.write('# D=%d, tau=%.10E, field=%.3E\n' % (DIM, tau, field))
    f.write('# N=%d, D=%d, D_itebd=%d, tau=%.10E, field=%.3E\n' % (N, DIM, DIM_iTEBD, tau, field))
    f.write('# time\t\tmag x profile\n')
    f.close()

    write_profile(mag_x_file_name, t, mag_x_list)

    f = open(mag_z_file_name, 'w')
    f.write('# Evolution mode: %s\n' % evolution_mode)
    f.write('# Order of the Trotter expansion: %d\n' % trotter_order)
    # f.write('# D=%d, tau=%.10E, field=%.3E\n' % (DIM, tau, field))
    f.write('# N=%d, D=%d, D_itebd=%d, tau=%.10E, field=%.3E\n' % (N, DIM, DIM_iTEBD, tau, field))
    f.write('# time\t\tmag z profile\n')
    f.close()

    write_profile(mag_z_file_name, t, mag_z_list)

    f = open(current_file_name, 'w')
    f.write('# Evolution mode: %s\n' % evolution_mode)
    f.write('# Order of the Trotter expansion: %d\n' % trotter_order)
    f.write('# N=%d, D=%d, D_itebd=%d, tau=%.10E, field=%.3E\n' % (N, DIM, DIM_iTEBD, tau, field))
    f.write('# time\t\tcurrent profile\n')
    f.close()

    # Quench
    if quench:
        mps.jordan_wigner_excitation()
        # mps.flip_x_excitation()
        # mps.domain_wall_excitation()

    op = constants.SZ
    mag_z_list = [mps.one_site_observation(position, op) for position in range(N)]
    mag_z_list = np.real(mag_z_list)
    print('<s_z>', *mag_z_list)

    op = constants.SX
    mag_x_list = [mps.one_site_observation(position, op) for position in range(N)]
    mag_x_list = np.real(mag_x_list)
    print('<s_x>', *mag_x_list)

    write_profile(mag_x_file_name, t, mag_x_list)
    write_profile(mag_z_file_name, t, mag_z_list)

    # entropies = [entropy_calculation(s / get_norm(s)) for s in mps.lambdas]
    entropies = mps.get_entropy_profile()
    write_profile(entropy_file_name, t, entropies)
    entropies_old = [1. for _ in entropies]

    flag = True

    step = 1
    # for step in range(1, 50_001):
    while True:

        mps.evolve_one_step()

        # isz_left = mps.imps_left.magnetization_z_calculation()
        # print('mag z left', isz_left)
        # isz_right = mps.imps_right.magnetization_z_calculation()
        # print('mag z right', isz_right)

        # isx_left = mps.imps_left.magnetization_x_calculation()
        # print('mag z left', isx_left)
        # isx_right = mps.imps_right.magnetization_x_calculation()
        # print('mag z right', isx_right)

        if step % refresh == 0:

            t = tau * step
            print('t', t)

            isz_left = mps.imps_left.magnetization_z_calculation()
            print('mag z left', isz_left)
            isz_right = mps.imps_right.magnetization_z_calculation()
            print('mag z right', isz_right)

            isx_left = mps.imps_left.magnetization_x_calculation()
            print('mag z left', isx_left)
            isx_right = mps.imps_right.magnetization_x_calculation()
            print('mag z right', isx_right)

            entropy_left_even = entropy_calculation(mps.imps_left.lambda_a / get_norm(mps.imps_left.lambda_a))
            entropy_left_odd = entropy_calculation(mps.imps_left.lambda_b / get_norm(mps.imps_left.lambda_b))
            print('left entropy even', entropy_left_even, 'odd', entropy_left_odd)
            entropy_right_even = entropy_calculation(mps.imps_right.lambda_a / get_norm(mps.imps_right.lambda_a))
            entropy_right_odd = entropy_calculation(mps.imps_right.lambda_b / get_norm(mps.imps_right.lambda_b))
            print('right entropy even', entropy_right_even, 'odd', entropy_right_odd)

            for i, ten in enumerate(mps.gammas):
                print('i', i, 'tensor shape', ten.shape)

            print('left gamma_a shape', mps.imps_left.gamma_a.shape)
            print('left gamma_b shape', mps.imps_left.gamma_b.shape)
            print('right gamma_a shape', mps.imps_right.gamma_a.shape)
            print('right gamma_a shape', mps.imps_right.gamma_b.shape)

            op = constants.SZ
            mag_z_list = [mps.one_site_observation(position, op) for position in range(N)]
            mag_z_list = np.real(mag_z_list)
            # print('<s_z>', *mag_z_list)

            op = constants.SX
            mag_x_list = [mps.one_site_observation(position, op) for position in range(N)]
            mag_x_list = np.real(mag_x_list)
            # print('<s_x>', *mag_x_list)

            write_profile(mag_x_file_name, t, mag_x_list)
            write_profile(mag_z_file_name, t, mag_z_list)

            # print('step', step, 'entropy', entropy, 'mag_z', *mag_z_list)
            # entropies = entropy_profile_calculation(list_of_gammas, lambda_central)

            # entropies = [entropy_calculation(s / get_norm(s)) for s in mps.lambdas]
            entropies_old = entropies
            entropies = mps.get_entropy_profile()
            print('entropy', *entropies)

            write_profile(entropy_file_name, t, entropies)

            # print('errors:')
            # print(mps.errors)

            write_profile(truncation_err_file_name, t, mps.errors)

            current_list = [mps.two_site_observation(pos) for pos in range(N-1)]
            current_list = np.real(current_list)
            print('J', *current_list)

            if flag and equal_list_eps(entropies, entropies_old, 1.E-8):  # test the convergence here
                # If converged, apply the bias voltage and switch to real-time evolution
                mps.bias1 = 1.
                mps.bias2 = -1.
                mps.evol_mode = 'real'
                mps.initialize_gates_wire()
                # and start writing down the current J by setting flag = False
                flag = False
                step = 0  # now starting from 0 (not from 1) because it will be incremented at the end of this block
            if not flag:  # write down the current
                write_profile(current_file_name, t, current_list)

        step += 1
"""


"""
import numpy as np
import matplotlib.pyplot as plt

df = np.loadtxt('entropy_profile_trotter4_dim120.txt')
x1 = [f[1:] for f in df]

df = np.loadtxt('entropy_profile_trotter4_dim32.txt')
x2 = [f[1:] for f in df]

df = np.loadtxt('entropy_profile_trotter2_dim32.txt')
x3 = [f[1:] for f in df]

df = np.loadtxt('entropy_profile_trotter4_dim16.txt')
x4 = [f[1:] for f in df]

x = (x1, x2, x3, x4)

rows = 1
columns = 4

fig=plt.figure(figsize=(rows, columns))

# plt.imshow(x[1:], cmap='gnuplot2_r', aspect=1/10)
# plt.colorbar()
# plt.show()

for i in range(1, columns*rows +1):
    img = x[i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap='gnuplot2_r', aspect=1/10)
plt.show()

"""