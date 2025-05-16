import numpy as np
import scipy.io as sio
from scipy.fftpack import fft
import matplotlib.pyplot as plt

def calculateEI(t, vth, timespike, tmax, ignore_transient=False):
    """
    Calculates the Error Index (EI) for thalamic cells.

    Parameters:
    t : np.ndarray
        Time vector (ms).
    vth : np.ndarray
        Array with membrane potentials of each thalamic cell (shape: [num_cells, time_steps]).
    timespike : np.ndarray
        Times of each SMC input pulse (ms).
    tmax : float
        Maximum time considered for calculation (ms).

    Returns:
    float
        Error index (EI).
    """
    m = vth.shape[0]  # Number of thalamic cells
    e = np.zeros(m)   # Error accumulator for each cell

    if ignore_transient:
        # Ignore the first 20 ms and last 25 ms
        b1 = np.where(timespike >= 20)[0][0] if np.any(timespike >= 20) else 0
        b2 = np.where(timespike <= tmax - 25)[0][-1] if np.any(timespike <= tmax - 25) else len(timespike) - 1
    else:
        # Use the full range of timespike
        b1 = 0
        b2 = len(timespike)

    for i in range(m):  # Loop over each thalamic cell
        compare = []  # Stores spike times for this cell
        # Detect spikes in the membrane potential
        for j in range(1, vth.shape[1]):
            if vth[i, j - 1] < -40 and vth[i, j] > -40:
                compare.append(t[j])

        for p in range(b1, b2):  # Loop over each timespike
            if p != b2 - 1:
                # Current spike window
                a = [c for c in compare if timespike[p] <= c < timespike[p] + 25]
                # Next spike window
                b = [c for c in compare if timespike[p] + 25 <= c < timespike[p + 1]]
            else:
                # For the last spike window
                a = [c for c in compare if timespike[p] <= c < tmax]
                b = []

            # Update error index
            if len(a) == 0:
                e[i] += 1  # No spike in the window
            elif len(a) > 1:
                e[i] += 1  # Multiple spikes in the window

            if len(b) > 0:
                e[i] += len(b)  # Extra spikes in the next window

    # Normalize and compute the mean error index
    er = np.mean(e / (b2 - b1 + 1))
    return er

def create_pulse_train_from_ints(pattern, tmax, dt, pattern_length=200):
    """
    Create a pulse train based on a pattern of intervals.
    Args:
        pattern (list or np.ndarray): Pattern of intervals.
        tmax (float): Total duration of the train in ms.
        dt (float): Time step in ms.
        pattern_length (int, optional): The length of the pattern. Default is 200.
    Returns:
        np.ndarray: Pulse train with the desired pattern.
    """

    # Extend pattern for the total duration (tmax)
    pattern_tmax = np.zeros(int(len(pattern) * tmax / pattern_length))
    for i in range(int(tmax / pattern_length)):
        pattern_tmax[i * len(pattern):(i + 1) * len(pattern)] = pattern_length * (i) + np.array(pattern)

    # Initialize the pulse train
    ID = np.zeros(int(tmax / dt))
    pulse_duration = int(0.3 / dt)  # Duration of each pulse in terms of time steps

    # Populate the pulse train based on the pattern
    for p in pattern_tmax:
        start_idx = int(p / dt)
        end_idx = start_idx + pulse_duration
        if end_idx < len(ID):  # Ensure we don't go out of bounds
            ID[start_idx:end_idx] = 300

    return ID


def normalize_pattern(pattern, pattern_length):
    """
    Normalize the pattern to ensure its sum matches the desired pattern length.
    Args:
        pattern (list or np.ndarray): Pattern of intervals.
        pattern_length (int): Desired pattern length.
    Returns:
        np.ndarray: Normalized pattern.
    """
    pattern = np.array(pattern)
    if pattern.sum() > pattern_length:
        pattern = pattern * (pattern_length / pattern.sum())  # Normalize to pattern_length
        pattern = np.ceil(pattern).astype(int)

        # Adjust the pattern to ensure the sum doesn't exceed the pattern length
        while pattern.sum() > pattern_length:
            pattern[pattern > 1] -= 1

    return pattern


def create_bit_string(pattern, pattern_length, num_of_vars):
    """
    Create a binary string representation of a pattern.
    Args:
        pattern (list or np.ndarray): Pattern of intervals.
        pattern_length (int): Desired pattern length.
        num_of_vars (int): Number of variables to consider.
    Returns:
        str: Binary string representation of the pattern.
    """
    pattern = normalize_pattern(pattern, pattern_length)
    tx_message = ['0'] * pattern_length  # Initialize with '0'

    for i in range(num_of_vars):
        idx = int(sum(pattern[:i + 1]))
        if idx < len(tx_message):
            tx_message[idx] = '1'

    return ''.join(tx_message)

class BGNetwork:
    def __init__(self, pd, init_file, wstim=0, freq=None, dt=0.1, tmax=1000, arrayOfIntsForPulses=None):
        self.pd = pd
        self.wstim = wstim
        self.freq = freq
        self.dt = dt
        self.tmax = tmax
        self.t = np.arange(0, tmax, dt)
        self.n = 10  # Assume 'n' is the number of neurons
        self.Idbs = self.initialize_dbs(wstim, freq, arrayOfIntsForPulses)
        self.initialize_matrices()
        self.set_membrane_parameters()
        self.set_synapse_parameters()
        self.load_initial_conditions(init_file)

    def initialize_dbs(self, wstim, freq=None, arrayOfIntsForPulses=None):
        if not wstim:
            return np.zeros(len(self.t))

        if arrayOfIntsForPulses:
            print("Initializing DBS with custom pulse pattern...")
            Id = create_pulse_train_from_ints(arrayOfIntsForPulses, self.tmax, self.dt)
            return np.concatenate(([0], Id))  # Add initial zero

        elif freq:
            print(f"Initializing DBS with frequency {freq} Hz...")
            return creatdbs(freq, self.tmax, self.dt)

        else:
            raise ValueError("DBS is enabled, but neither frequency nor custom pulse pattern is provided.")

    def initialize_matrices(self):
        # Initialize voltage and synaptic matrices
        self.vth = np.zeros((self.n, len(self.t)))
        self.vsn = np.zeros((self.n, len(self.t)))
        self.vge = np.zeros((self.n, len(self.t)))
        self.vgi = np.zeros((self.n, len(self.t)))

        self.S2 = np.zeros(self.n)
        self.S3 = np.zeros(self.n)
        self.Z2 = np.zeros(self.n)
        self.Z4 = np.zeros(self.n)
        # Synaptic state matrices
        self.S2 = np.zeros(self.n)
        self.S21 = np.zeros(self.n)
        self.S3 = np.zeros(self.n)
        self.S31 = np.zeros(self.n)
        self.S32 = np.zeros(self.n)
        self.S4 = np.zeros(self.n)
        self.Z2 = np.zeros(self.n)
        self.Z4 = np.zeros(self.n)
        # Additional trackers
        self.n3t = np.zeros((self.n, len(self.t)))
        self.h3t = np.zeros((self.n, len(self.t)))
        self.r3t = np.zeros((self.n, len(self.t)))
        self.ca3t = np.zeros((self.n, len(self.t)))
        self.s2t = np.zeros((self.n, len(self.t)))
        self.s3t = np.zeros((self.n, len(self.t)))
        self.ISaveSTN = np.zeros((self.n, len(self.t)))
        self.ISaveGPE = np.zeros((self.n, len(self.t)))

        # Synaptic trackers
        self.aGPe = np.zeros((self.n, len(self.t)))  # Track GPe activity
        self.aSTN1 = np.zeros((self.n, len(self.t)))  # Track STN activity

        # Additional trackers
        self.h2t = np.zeros((self.n, len(self.t)))
        self.n2t = np.zeros((self.n, len(self.t)))
        self.r2t = np.zeros((self.n, len(self.t)))
        self.c2t = np.zeros((self.n, len(self.t)))
        self.ca2t = np.zeros((self.n, len(self.t)))


    def set_membrane_parameters(self):
        # Membrane Parameters
        # In order of Th (Thalamus), STN, GP or Th, STN, GPe, GPi
        # Membrane capacitance
        self.Cm = 1
        # Leak conductance and reversal potentials
        self.gl = [0.05, 2.25, 0.1]  # Leak conductance
        self.El = [-70, -60, -65]    # Leak reversal potential
        # Sodium conductance and reversal potentials
        self.gna = [3, 37, 120]      # Sodium conductance
        self.Ena = [50, 55, 55]      # Sodium reversal potential
        # Potassium conductance and reversal potentials
        self.gk = [5, 45, 30]        # Potassium conductance
        self.Ek = [-75, -80, -80]    # Potassium reversal potential
        # T-type calcium conductance and reversal potential
        self.gt = [5, 0.5, 0.5]      # T-type calcium conductance
        self.Et = 0                  # T-type calcium reversal potential
        # Calcium conductance and reversal potentials
        self.gca = [0, 2, 0.15]      # Calcium conductance
        self.Eca = [0, 140, 120]     # Calcium reversal potential
        # AHP (after-hyperpolarization) conductance
        self.gahp = [0, 20, 10]      # AHP conductance
        self.k1 = [0, 15, 10]        # Scaling constant for AHP current
        self.kca = [0, 22.5, 15]     # Scaling constant for calcium-dependent AHP
        # Synaptic parameters
        self.A = [0, 3, 2, 2]        # Scaling factor A
        self.B = [0, 0.1, 0.04, 0.04]  # Scaling factor B
        self.the = [0, 30, 20, 20]   # Threshold values

    def set_synapse_parameters(self):
        # Setting synapse parameters
        self.gsyn = [1, 0.3, 1, 0.3, 1, 0.08]   # Synaptic conductance
        self.Esyn = [-85, 0, -85, 0, -85, -85]  # Synaptic reversal potentials
        self.tau = 5                            # Time constant for synaptic dynamics
        # Peak conductances
        self.gpeak = 0.43
        self.gpeak1 = 0.3

    def load_initial_conditions(self, init_file):
        mat_data = sio.loadmat(init_file)

        self.timespike = mat_data['timespike'].flatten()
        self.r = mat_data['r'].flatten()
        self.Istim = mat_data['Istim'].flatten()

        # Load v1, v2, v3, v4 from file
        v1 = mat_data['v1'].flatten()
        v2 = mat_data['v2'].flatten()
        v3 = mat_data['v3'].flatten()
        v4 = mat_data['v4'].flatten()

        # Set initial membrane voltages
        self.vth[:, 0] = v1[:self.n]
        self.vsn[:, 0] = v2[:self.n]
        self.vge[:, 0] = v3[:self.n]
        self.vgi[:, 0] = v4[:self.n]

        # Initialize gating variables
        self.N2 = stn_ninf(self.vsn[:, 0])
        self.N3 = gpe_ninf(self.vge[:, 0])
        self.N4 = gpe_ninf(self.vgi[:, 0])

        self.H1 = th_hinf(self.vth[:, 0])
        self.H2 = stn_hinf(self.vsn[:, 0])
        self.H3 = gpe_hinf(self.vge[:, 0])
        self.H4 = gpe_hinf(self.vgi[:, 0])

        self.R1 = th_rinf(self.vth[:, 0])
        self.R2 = stn_rinf(self.vsn[:, 0])
        self.R3 = gpe_rinf(self.vge[:, 0])
        self.R4 = gpe_rinf(self.vgi[:, 0])

        self.CA2 = 0.1
        self.CA3 = self.CA2
        self.CA4 = self.CA2
        self.C2 = stn_cinf(self.vsn[:, 0])

        # Populate additional trackers with initial values
        self.h2t[:, 0] = self.H2
        self.n2t[:, 0] = self.N2
        self.r2t[:, 0] = self.R2
        self.c2t[:, 0] = self.C2
        self.ca2t[:, 0] = self.CA2
        self.n3t[:, 0] = self.N3
        self.h3t[:, 0] = self.H3
        self.r3t[:, 0] = self.R3
        self.ca3t[:, 0] = self.CA3

    # # Placeholder gating
    # def membrane_potentials(self, i): # function used nowhere
    #     # Compute membrane potentials for the current time step
    #     Il_th = self.gl[0] * (self.vth[:, i - 1] - self.El[0])
    #     Ina_th = self.gna[0] * (self.vth[:, i - 1] ** 3) * (self.vth[:, i - 1] - self.Ena[0])
    #     # Other currents follow the same structure
    #     # Update voltages using the computed currents
    #     self.vth[:, i] = self.vth[:, i - 1] + self.dt * (-Il_th - Ina_th)

    def run_simulation(self):
        EI_sample_duration = 100
        ei_sequence = np.zeros(int(self.tmax / EI_sample_duration))
        pb_sequence = np.zeros(int(self.tmax / EI_sample_duration))

        num_chunks = int(len(self.Idbs) / (EI_sample_duration / self.dt))
        for idx in range(num_chunks):
            start_idx = int(idx * EI_sample_duration / self.dt)
            end_idx = int((idx + 1) * EI_sample_duration / self.dt)

            current_pulse = np.zeros(end_idx - start_idx + 1)
            current_pulse[1:] = self.Idbs[start_idx:end_idx]
            # print(f"Chunk {idx + 1}: {current_pulse}")

        for idx in range(num_chunks):
            start_idx = int(idx * EI_sample_duration / self.dt)
            end_idx = int((idx + 1) * EI_sample_duration / self.dt)

            for i in range(start_idx, end_idx):
                # Extracting the previous state variables
                V1 = self.vth[:, i - 1]  # Thalamus membrane potential
                V2 = self.vsn[:, i - 1]  # STN membrane potential
                V3 = self.vge[:, i - 1]  # GPe membrane potential
                V4 = self.vgi[:, i - 1]  # GPi membrane potential
                # Update S21 (circular shift right by 1)
                self.S21[1:] = self.S2[:-1]  # S21(2:n) = S2(1:n-1)
                self.S21[0] = self.S2[-1]    # S21(1) = S2(n)

                # Update S31 (circular shift left by 1)
                self.S31[:-1] = self.S3[1:]  # S31(1:n-1) = S3(2:n)
                self.S31[-1] = self.S3[0]    # S31(n) = S3(1)

                # Update S32 (circular shift left by 2)
                self.S32[2:] = self.S3[:-2]  # S32(3:n) = S3(1:n-2)
                self.S32[:2] = self.S3[-2:]  # S32(1:2) = S3(n-1:n)

                # Membrane parameters
                m1 = th_minf(V1)
                m2 = stn_minf(V2)
                m3 = gpe_minf(V3)
                m4 = gpe_minf(V4)

                n2 = stn_ninf(V2)
                n3 = gpe_ninf(V3)
                n4 = gpe_ninf(V4)

                h1 = th_hinf(V1)
                h2 = stn_hinf(V2)
                h3 = gpe_hinf(V3)
                h4 = gpe_hinf(V4)

                p1 = th_pinf(V1)

                a2 = stn_ainf(V2)
                a3 = gpe_ainf(V3)
                a4 = gpe_ainf(V4)

                b2 = stn_binf(self.R2)

                s3 = gpe_sinf(V3)
                s4 = gpe_sinf(V4)

                r1 = th_rinf(V1)
                r2 = stn_rinf(V2)
                r3 = gpe_rinf(V3)
                r4 = gpe_rinf(V4)

                c2 = stn_cinf(V2)

                # Time constants
                tn2 = stn_taun(V2)  # Time constant for STN neuron (n gate)
                tn3 = gpe_taun(V3)  # Time constant for GPe neuron (n gate)
                tn4 = gpe_taun(V4)  # Time constant for GPi neuron (n gate)

                th1 = th_tauh(V1)  # Time constant for Thalamic neuron (h gate)
                th2 = stn_tauh(V2)  # Time constant for STN neuron (h gate)
                th3 = gpe_tauh(V3)  # Time constant for GPe neuron (h gate)
                th4 = gpe_tauh(V4)  # Time constant for GPi neuron (h gate)

                tr1 = th_taur(V1)  # Time constant for Thalamic neuron (r gate)
                tr2 = stn_taur(V2)  # Time constant for STN neuron (r gate)
                tr3 = 30  # Constant time constant for GPe neurons (r gate)
                tr4 = 30  # Constant time constant for GPi neurons (r gate)

                tc2 = stn_tauc(V2)  # Time constant for calcium dynamics in STN neuron

                # Thalamic cell currents
                Il1 = self.gl[0] * (V1 - self.El[0])  # Leak current
                Ina1 = self.gna[0] * (m1 ** 3) * self.H1 * (V1 - self.Ena[0])  # Sodium current
                Ik1 = self.gk[0] * ((0.75 * (1 - self.H1)) ** 4) * (V1 - self.Ek[0])  # Potassium current
                It1 = self.gt[0] * (p1 ** 2) * self.R1 * (V1 - self.Et)  # T-type calcium current
                Igith = 1.4 * self.gsyn[5] * (V1 - self.Esyn[5]) * self.S4  # GABAergic synaptic input

                # STN cell currents
                Il2 = self.gl[1] * (V2 - self.El[1])  # Leak current
                Ik2 = self.gk[1] * (self.N2 ** 4) * (V2 - self.Ek[1])  # Potassium current
                Ina2 = self.gna[1] * (m2 ** 3) * self.H2 * (V2 - self.Ena[1])  # Sodium current
                It2 = self.gt[1] * (a2 ** 3) * (b2 ** 2) * (V2 - self.Eca[1])  # T-type calcium current
                Ica2 = self.gca[1] * (self.C2 ** 2) * (V2 - self.Eca[1])  # Calcium current
                Iahp2 = self.gahp[1] * (V2 - self.Ek[1]) * (self.CA2 / (self.CA2 + self.k1[1]))  # After-hyperpolarization current
                Igesn = 0.5 * (self.gsyn[0] * (V2 - self.Esyn[0]) * (self.S3 + self.S31))  # GPe-STN synaptic input
                Iappstn = 33 - self.pd * 10  # Applied current to STN

                # GPe cell currents
                Il3 = self.gl[2] * (V3 - self.El[2])  # Leak current
                Ik3 = self.gk[2] * (self.N3 ** 4) * (V3 - self.Ek[2])  # Potassium current
                Ina3 = self.gna[2] * (m3 ** 3) * self.H3 * (V3 - self.Ena[2])  # Sodium current
                It3 = self.gt[2] * (a3 ** 3) * self.R3 * (V3 - self.Eca[2])  # T-type calcium current
                Ica3 = self.gca[2] * (s3 ** 2) * (V3 - self.Eca[2])  # Calcium current
                Iahp3 = self.gahp[2] * (V3 - self.Ek[2]) * (self.CA3 / (self.CA3 + self.k1[2]))  # After-hyperpolarization current
                Isnge = 0.5 * (self.gsyn[1] * (V3 - self.Esyn[1]) * (self.S2 + self.S21))  # STN-GPe synaptic input
                Igege = 0.5 * (self.gsyn[2] * (V3 - self.Esyn[2]) * (self.S31 + self.S32))  # GPe-GPe synaptic input
                Iappgpe = 21 - 13 * self.pd + self.r  # Applied current to GPe

                # GPi cell currents
                Il4 = self.gl[2] * (V4 - self.El[2])  # Leak current
                Ik4 = self.gk[2] * (self.N4 ** 4) * (V4 - self.Ek[2])  # Potassium current
                Ina4 = self.gna[2] * (m4 ** 3) * self.H4 * (V4 - self.Ena[2])  # Sodium current
                It4 = self.gt[2] * (a4 ** 3) * self.R4 * (V4 - self.Eca[2])  # T-type calcium current
                Ica4 = self.gca[2] * (s4 ** 2) * (V4 - self.Eca[2])  # Calcium current
                Iahp4 = self.gahp[2] * (V4 - self.Ek[2]) * (self.CA4 / (self.CA4 + self.k1[2]))  # After-hyperpolarization current
                Isngi = 0.5 * (self.gsyn[3] * (V4 - self.Esyn[3]) * (self.S2 + self.S21))  # STN-GPi synaptic input
                Igigi = 0.5 * (self.gsyn[4] * (V4 - self.Esyn[4]) * (self.S31 + self.S32))  # GPe-GPi synaptic input
                Iappgpi = 22 - self.pd * 6  # Applied current to GPi

                # Differential equations for the cells
                # Thalamic
                self.vth[:, i] = V1 + self.dt * (1 / self.Cm * (-Il1 - Ik1 - Ina1 - It1 - Igith + self.Istim[i]))
                self.H1 = self.H1 + self.dt * ((h1 - self.H1) / th1)
                self.R1 = self.R1 + self.dt * ((r1 - self.R1) / tr1)

                # STN
                self.vsn[:, i] = V2 + self.dt * (1 / self.Cm * (-Il2 - Ik2 - Ina2 - It2 - Ica2 - Iahp2 - Igesn + Iappstn + self.Idbs[i]))
                self.N2 = self.N2 + self.dt * (0.75 * (n2 - self.N2) / tn2)
                self.H2 = self.H2 + self.dt * (0.75 * (h2 - self.H2) / th2)
                self.R2 = self.R2 + self.dt * (0.2 * (r2 - self.R2) / tr2)
                self.CA2 = self.CA2 + self.dt * (3.75 * 10**-5 * (-Ica2 - It2 - self.kca[1] * self.CA2))
                self.C2 = self.C2 + self.dt * (0.08 * (c2 - self.C2) / tc2)
                # Detect spikes
                a = np.where((self.vsn[:, i - 1] < -10) & (self.vsn[:, i] > -10))[0]
                u = np.zeros(self.n)
                u[a] = self.gpeak / (self.tau * np.exp(-1)) / self.dt
                self.S2 = self.S2 + self.dt * self.Z2
                zdot = u - 2 / self.tau * self.Z2 - 1 / (self.tau**2) * self.S2
                self.Z2 = self.Z2 + self.dt * zdot

                # GPe
                self.vge[:, i] = V3 + self.dt * (1 / self.Cm * (-Il3 - Ik3 - Ina3 - It3 - Ica3 - Iahp3 - Isnge - Igege + Iappgpe))
                self.N3 = self.N3 + self.dt * (0.1 * (n3 - self.N3) / tn3)
                self.H3 = self.H3 + self.dt * (0.05 * (h3 - self.H3) / th3)
                self.R3 = self.R3 + self.dt * (1 * (r3 - self.R3) / tr3)
                self.CA3 = self.CA3 + self.dt * (1 * 10**-4 * (-Ica3 - It3 - self.kca[2] * self.CA3))
                self.S3 = self.S3 + self.dt * (self.A[2] * (1 - self.S3) * Hinf(V3 - self.the[2]) - self.B[2] * self.S3)

                # GPi
                self.vgi[:, i] = V4 + self.dt * (1 / self.Cm * (-Il4 - Ik4 - Ina4 - It4 - Ica4 - Iahp4 - Isngi - Igigi + Iappgpi))
                self.N4 = self.N4 + self.dt * (0.1 * (n4 - self.N4) / tn4)
                self.H4 = self.H4 + self.dt * (0.05 * (h4 - self.H4) / th4)
                self.R4 = self.R4 + self.dt * (1 * (r4 - self.R4) / tr4)
                self.CA4 = self.CA4 + self.dt * (1 * 10**-4 * (-Ica4 - It4 - self.kca[2] * self.CA4))
                # Detect spikes
                a = np.where((self.vgi[:, i - 1] < -10) & (self.vgi[:, i] > -10))[0]
                u = np.zeros(self.n)
                u[a] = self.gpeak1 / (self.tau * np.exp(-1)) / self.dt
                self.S4 = self.S4 + self.dt * self.Z4
                zdot = u - 2 / self.tau * self.Z4 - 1 / (self.tau**2) * self.S4
                self.Z4 = self.Z4 + self.dt * zdot

                # Additional trackers
                # Uncomment the following lines if needed:
                # self.h2t[:, i] = self.H2
                # self.n2t[:, i] = self.N2
                # self.r2t[:, i] = self.R2
                # self.c2t[:, i] = self.C2
                # self.ca2t[:, i] = self.CA2

                self.n3t[:, i] = self.N3
                self.h3t[:, i] = self.H3
                self.r3t[:, i] = self.R3
                self.ca3t[:, i] = self.CA3
                self.s2t[:, i] = self.S2
                self.s3t[:, i] = self.S31  # Assuming S31 corresponds to S21 in Python
            #END of FOR
            current_spike_indices = np.where((self.timespike < end_idx) & (self.timespike > start_idx))[0]
            current_spikes = self.timespike[current_spike_indices]

            EI = calculateEI(self.t, self.vth, self.timespike, self.tmax)
            # Alternative calculation with predefined spike times
            predefined_spike_times = np.arange(1, 1001, 5)  # Equivalent to MATLAB's 1:5:1000
            EI_predefined = calculateEI(self.t, self.vth, predefined_spike_times, self.tmax)

            # Initialize fft_result for all cells
            fft_length = 8192
            downsampling_factor = 12
            sampling_rate = 98304
            # Downsampled sampling rate
            downsampled_rate = sampling_rate / downsampling_factor
            fft_result = np.zeros((self.n, 8192))
            # Compute FFT for each cell in vgi with downsampling
            for k in range(self.n):
                fft_result[k, :] = np.abs(np.fft.fft(self.vgi[k, ::12], n=8192))  # Downsample every 12th point

            # Alternatively, process only the first cell as in the MATLAB code
            fft_result_single = np.abs(np.fft.fft(self.vgi[0, ::12], n=8192))

            # Generate the frequency bins for the FFT
            freqs = np.fft.fftfreq(fft_length, d=1 / downsampled_rate)
            # Filter for the power band frequencies (13-35 Hz)
            freq_band_indices = np.where((freqs >= 13) & (freqs <= 35))[0]
            # Compute the power band
            Pb = np.mean(fft_result_single[freq_band_indices]) / 1e3 - 1.5

            ei_sequence[idx] = EI
            pb_sequence[idx] = Pb
        #END OF FOR
        return ei_sequence, pb_sequence

    def calculate_error_index(self):
        # Placeholder for error index calculation
        return np.random.rand()  # Replace with actual calculation

    def calculate_ei(t, vth, timespike, tmax):
    
    # Calculates the Error Index (EI).

    # Parameters:
    #     t (numpy.ndarray): Time vector (ms).
    #     vth (numpy.ndarray): Array with membrane potentials of each thalamic cell.
    #     timespike (numpy.ndarray): Time of each SMC input pulse.
    #     tmax (float): Maximum time taken into consideration for calculation.

    # Returns:
    #     float: Error index.
    
    # Load initialization parameters
    # Simulated `init.mat` logic; replace with actual initialization if needed
    # For example: data = scipy.io.loadmat('init.mat')
    
        m = vth.shape[0]  # Number of thalamic cells
        e = np.zeros(m)   # Initialize error array
    
        b1 = 0
        b2 = len(timespike)
    
        for i in range(m):
            compare = []
    
            # Find time points where membrane potential crosses -40 mV upwards
            for j in range(1, len(vth[i])):
                if vth[i][j - 1] < -40 and vth[i][j] > -40:
                    compare.append(t[j])
    
            for p in range(b1, b2):
                if p != b2 - 1:
                    a = [c for c in compare if timespike[p] <= c < timespike[p] + 25]
                    b = [c for c in compare if timespike[p] + 25 <= c < timespike[p + 1]]
                elif b2 == len(timespike):
                    a = [c for c in compare if timespike[p] <= c < tmax]
                    b = []
                else:
                    a = [c for c in compare if timespike[p] <= c < timespike[p + 1]]
                    b = [c for c in compare if timespike[p] + 25 <= c < timespike[p + 1]]
    
                # Update error count
                if len(a) == 0 or len(a) > 1:
                    e[i] += 1
                if len(b) > 0:
                    e[i] += len(b)
    
        # Calculate the mean error index
        er = np.mean(e / (b2 - b1))
        return er


    def calculate_power_band(self):
        # Placeholder for power band calculation
        fft_result = np.abs(np.fft.fft(self.vgi[0, :]))
        power_band = np.mean(fft_result[13:35])
        return power_band





def creatdbs(f, tmax, dt):
    """
    Creates a DBS train of frequency `f`, duration `tmax` (ms), and time step `dt` (ms).

    Parameters:
    f : float
        Frequency of the DBS pulses in Hz.
    tmax : float
        Total duration of the DBS train in milliseconds.
    dt : float
        Time step for discretizing the signal in milliseconds.

    Returns:
    np.ndarray
        DBS train as a 1D array.
    """
    # Create time vector
    t = np.arange(0, tmax + dt, dt)

    # Initialize DBS train array
    ID = np.zeros(len(t))

    # Pulse amplitude and pulse definition
    iD = 300
    pulse = iD * np.ones(int(0.3 / dt))

    # Generate DBS train
    i = 0
    while i < len(t):
        ID[i:i + len(pulse)] = pulse  # Insert pulse
        instfreq = f
        isi = 1000 / instfreq  # Inter-stimulus interval in ms
        i += round(isi / dt)  # Increment index by ISI steps

    return ID



def gpe_ainf(V):
    """
    Steady-state activation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    print(1 / (1 + np.exp(-(V + 57) / 2)))
    return 1 / (1 + np.exp(-(V + 57) / 2))

def gpe_minf(V):
    """
    Steady-state activation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 37) / 10))

def gpe_hinf(V):
    """
    Steady-state inactivation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 58) / 12))

def gpe_ninf(V):
    """
    Steady-state activation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 50) / 14))


def gpe_rinf(V):
    """
    Steady-state inactivation function for GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 70) / 2))


def gpe_sinf(V):
    """
    Steady-state activation function for synaptic input to GPe neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 35) / 2))


def gpe_tauh(V):
    """
    Time constant function for GPe neuron inactivation.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 0.05 + 0.27 / (1 + np.exp(-(V + 40) / -12))


def gpe_taun(V):
    """
    Time constant function for GPe neuron activation.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 0.05 + 0.27 / (1 + np.exp(-(V + 40) / -12))


def Hinf(V):
    """
    Steady-state activation function.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 57) / 2))


def stn_ainf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 63) / 7.8))


def stn_binf(R):
    """
    Steady-state activation function for STN neurons based on variable R.

    Parameters:
    R : float or np.ndarray
        Input variable.

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(R - 0.4) / 0.1)) - 1 / (1 + np.exp(0.4 / 0.1))


def stn_cinf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 20) / 8))


def stn_hinf(V):
    """
    Steady-state inactivation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 39) / 3.1))


def stn_minf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 30) / 15))


def stn_ninf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 32) / 8.0))


def stn_rinf(V):
    """
    Steady-state inactivation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 67) / 2))


def stn_sinf(V):
    """
    Steady-state activation function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 39) / 8))


def stn_tauc(V):
    """
    Time constant function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 1 + 10 / (1 + np.exp((V + 80) / 26))


def stn_tauh(V):
    """
    Time constant function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 1 + 500 / (1 + np.exp(-(V + 57) / -3))


def stn_taun(V):
    """
    Time constant function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 1 + 100 / (1 + np.exp(-(V + 80) / -26))


def stn_taur(V):
    """
    Time constant function for STN neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 7.1 + 17.5 / (1 + np.exp(-(V - 68) / -2.2))


def th_hinf(V):
    """
    Steady-state inactivation function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 41) / 4))


def th_minf(V):
    """
    Computes the steady-state activation (minf) for a thalamic neuron.

    Parameters:
    V (numpy array or float): Membrane potential (in mV)

    Returns:
    numpy array or float: Steady-state activation
    """
    return 1 / (1 + np.exp(-(V + 37) / 7))


import numpy as np

def th_pinf(V):
    """
    Steady-state activation function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state activation value.
    """
    return 1 / (1 + np.exp(-(V + 60) / 6.2))




def th_rinf(V):
    """
    Steady-state inactivation function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Steady-state inactivation value.
    """
    return 1 / (1 + np.exp((V + 84) / 4))


def th_tauh(V):
    """
    Time constant function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 1 / (ah(V) + bh(V))

def ah(V):
    """
    Helper function to compute `a_h` for th_tauh.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
    """
    return 0.128 * np.exp(-(V + 46) / 18)

def bh(V):
    """
    Helper function to compute `b_h` for th_tauh.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
    """
    return 4 / (1 + np.exp(-(V + 23) / 5))



def th_taur(V):
    """
    Time constant function for thalamic neurons.

    Parameters:
    V : float or np.ndarray
        Membrane potential (mV).

    Returns:
    float or np.ndarray
        Time constant (ms).
    """
    return 0.15 * (28 + np.exp(-(V + 25) / 10.5))








bg_network = BGNetwork(pd=0, init_file="init.mat", wstim=0)
ei_sequence, pb_sequence = bg_network.run_simulation()
print("Error Index:", ei_sequence, '\n\n', "Spectral Density:", pb_sequence)
print(ei_sequence.sum())
print(pb_sequence.sum())