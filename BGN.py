"""
Based on MATLAB implementation from:
https://github.com/ModelDBRepository/141699/tree/master
Reworked into a class to more easily track variable
changes over time, and so that the environment can
easily be stepped through at a discrete time interval.
"""

from typing import Literal
import numpy as np
from activations import *
from scipy.signal import find_peaks


class BGN:
    def __init__(
        self,
        pd: Literal[0, 1],
        wstim: Literal[0, 1],
        freq: float,
        freq_time: int = 200, # in ms
        tmax: int = 1000,
        n: int = 10,
        seed: int | None = None,
    ) -> None:
        np.random.seed(seed)
        self.pd = pd
        self.wstim = wstim
        self.freq = freq
        self.freq_time = freq_time
        self.tmax = tmax
        self.dt = 0.01
        self.n = n
        self.t = np.arange(0, self.tmax, self.dt)
        self.Cm = 1

        # Create DBS pulses
        if wstim:
            t = np.arange(0, self.tmax + self.dt, self.dt)
            self.Idbs = np.zeros(len(t))
            iD = 300
            pulse = iD * np.ones(int(0.3 / self.dt))
            i = 0
            while i < len(t):
                if(i + len(pulse) > len(self.Idbs)):
                    self.Idbs[i : len(self.Idbs)] = pulse[:len(self.Idbs)-i]
                else:
                    self.Idbs[i : i + len(pulse)] = pulse
                instfreq = self.freq
                isi = 1000 / instfreq
                i += round(isi / self.dt)
        else:
            self.Idbs = np.zeros(len(self.t))

        # create SMC input to thalmic cells
        self.Istim = np.zeros(len(self.t))
        dsm = 5
        ism = 3.5
        pulse = ism * np.ones(int(dsm / self.dt))
        p = 1
        cv = 0.2
        smc_freq = 14
        A = 1 / (cv**2)
        B = smc_freq / A
        instfreq = np.random.gamma(A, B)
        ipi = 1000 / instfreq
        p = int(p + np.round(ipi / self.dt))
        self.smc_pulse = []
        while p < len(self.t):
            self.smc_pulse.append(self.t[p])

            if(p + len(pulse) > len(self.Istim)):
                self.Istim[p: len(self.Istim)] = pulse[:len(self.Istim)-p]
                
            else:
                self.Istim[p : int(p + dsm / self.dt)] = pulse

            # self.Istim[p : int(p + dsm / self.dt)] = pulse
            instfreq = np.random.gamma(A, B)
            ipi = 1000 / instfreq
            p = int(p + np.round(ipi / self.dt))

        self.smc_pulse = np.round(np.array(self.smc_pulse))

        self.gl = np.array([0.05, 2.25, 0.1])
        self.El = np.array([-70, -60, -65])
        self.gna = np.array([3, 37, 120])
        self.Ena = np.array([50, 55, 55])
        self.gk = np.array([5, 45, 30])
        self.Ek = np.array([-75, -80, -80])
        self.gt = np.array([5, 0.5, 0.5])
        self.Et = 0
        self.gca = np.array([0, 2, 0.15])
        self.Eca = np.array([0, 140, 120])
        self.gahp = np.array([0, 20, 10])
        self.k1 = np.array([0, 15, 10])
        self.kca = np.array([0, 22.5, 15])
        self.A = np.array([0, 3, 2, 2])
        self.B = np.array([0, 0.1, 0.04, 0.04])
        self.the = np.array([0, 30, 20, 20])
        self.gsyn = np.array([1, 0.3, 1, 0.3, 1, 0.08])
        self.Esyn = np.array([-85, 0, -85, 0, -85, -85])
        self.tau = 5
        self.gpeak = 0.43
        self.gpeak1 = 0.3
        self.vth = np.zeros((self.n, len(self.t)))
        self.vsn = np.zeros((self.n, len(self.t)))
        self.vge = np.zeros((self.n, len(self.t)))
        self.vgi = np.zeros((self.n, len(self.t)))
        self.S2 = np.zeros(self.n)
        self.S21 = np.zeros(self.n)
        self.S3 = np.zeros(self.n)
        self.S31 = np.zeros(self.n)
        self.S32 = np.zeros(self.n)
        self.S4 = np.zeros(self.n)
        self.Z2 = np.zeros(self.n)
        self.Z4 = np.zeros(self.n)
        self.n3t = np.zeros((self.n, len(self.t)))
        self.h3t = np.zeros((self.n, len(self.t)))
        self.r3t = np.zeros((self.n, len(self.t)))
        self.ca3t = np.zeros((self.n, len(self.t)))
        self.s2t = np.zeros((self.n, len(self.t)))
        self.s3t = np.zeros((self.n, len(self.t)))
        self.ISaveSTN = np.zeros((self.n, len(self.t)))
        self.ISaveGPE = np.zeros((self.n, len(self.t)))

        self.r = np.random.randn(1, self.n) * 2
        self.v1 = np.random.uniform(-74.4314, -55.7487, self.n)
        self.v2 = np.random.uniform(-74.4314, -55.7487, self.n)
        self.v3 = np.random.uniform(-74.4314, -55.7487, self.n)
        self.v4 = np.random.uniform(-74.4314, -55.7487, self.n)

        self.vth[:, 0] = self.v1
        self.vsn[:, 0] = self.v2
        self.vge[:, 0] = self.v3
        self.vgi[:, 0] = self.v4

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

        self.n3t[:, 0] = self.N3
        self.h3t[:, 0] = self.H3
        self.r3t[:, 0] = self.R3
        self.ca3t[:, 0] = self.CA3

        self.aGPe = np.zeros((self.n, len(self.t)))
        self.aSTN1 = np.zeros((self.n, len(self.t)))

        self.V1 = 0
        self.V2 = 0
        self.V3 = 0
        self.V4 = 0

        self.m1 = 0
        self.m2 = 0
        self.m3 = 0
        self.m4 = 0
        self.n2 = 0
        self.n3 = 0
        self.n4 = 0
        self.h1 = 0
        self.h2 = 0
        self.h3 = 0
        self.h4 = 0
        self.p1 = 0
        self.a2 = 0
        self.a3 = 0
        self.a4 = 0
        self.b2 = 0
        self.s3 = 0
        self.s4 = 0
        self.r1 = 0
        self.r2 = 0
        self.r3 = 0
        self.r4 = 0
        self.c2 = 0
        self.tn2 = 0
        self.tn3 = 0
        self.tn4 = 0
        self.th1 = 0
        self.th2 = 0
        self.th3 = 0
        self.th4 = 0
        self.tr1 = 0
        self.tr2 = 0
        self.tr3 = 0
        self.tr4 = 0
        self.tc2 = 0

        self.Ina1 = 0
        self.Il1 = 0
        self.Ik1 = 0
        self.It1 = 0
        self.Igith = 0

        self.Il2 = 0
        self.Ik2 = 0
        self.Ina2 = 0
        self.It2 = 0
        self.Ica2 = 0
        self.Iahp2 = 0
        self.Igesn = 0
        self.Iappstn = 0

        self.Il4 = 0
        self.Ik4 = 0
        self.Ina4 = 0
        self.It4 = 0
        self.Ica4 = 0
        self.Iahp4 = 0
        self.Isngi = 0
        self.Igigi = 0
        self.Iappgpi = 0

        self.a = 0
        self.u = 0
        self.zdot = 0

        self.PB = 0
        self.PBs = np.zeros(len(self.t))
        self.EI = 0

        self.i = 2
        self.new_state = 0
        self.reward = 0
        self.new_state = None
        self.sgi_sum = np.zeros(freq_time)
        self.bursts = np.zeros(freq_time)
        self.sgi = np.zeros((self.n, len(self.t)))
  

    def step(self, action = (0, 0), mini_steps = 20000):
        for mini_step in range(mini_steps):
            i = self.i
            n = self.n
            self.V1 = self.vth[:, i - 2]
            self.V2 = self.vsn[:, i - 2]
            self.V3 = self.vge[:, i - 2]
            self.V4 = self.vgi[:, i - 2]
            self.S21[1:n] = self.S2[0 : n - 1]
            self.S21[0] = self.S2[n - 1]
            self.S31[0 : n - 1] = self.S3[1:n]
            self.S31[n - 1] = self.S3[0]
            self.S32[2:n] = self.S3[0 : n - 2]
            self.S32[0:2] = self.S3[n - 2 : n]

            self.m1 = th_minf(self.V1)
            self.m2 = stn_minf(self.V2)
            self.m3 = gpe_minf(self.V3)
            self.m4 = gpe_minf(self.V4)
            self.n2 = stn_ninf(self.V2)
            self.n3 = gpe_ninf(self.V3)
            self.n4 = gpe_ninf(self.V4)
            self.h1 = th_hinf(self.V1)
            self.h2 = stn_hinf(self.V2)
            self.h3 = gpe_hinf(self.V3)
            self.h4 = gpe_hinf(self.V4)
            self.p1 = th_pinf(self.V1)
            self.a2 = stn_ainf(self.V2)
            self.a3 = gpe_ainf(self.V3)
            self.a4 = gpe_ainf(self.V4)
            self.b2 = stn_binf(self.R2)
            self.s3 = gpe_sinf(self.V3)
            self.s4 = gpe_sinf(self.V4)
            self.r1 = th_rinf(self.V1)
            self.r2 = stn_rinf(self.V2)
            self.r3 = gpe_rinf(self.V3)
            self.r4 = gpe_rinf(self.V4)
            self.c2 = stn_cinf(self.V2)
            self.tn2 = stn_taun(self.V2)
            self.tn3 = gpe_taun(self.V3)
            self.tn4 = gpe_taun(self.V4)
            self.th1 = th_tauh(self.V1)
            self.th2 = stn_tauh(self.V2)
            self.th3 = gpe_tauh(self.V3)
            self.th4 = gpe_tauh(self.V4)
            self.tr1 = th_taur(self.V1)
            self.tr2 = stn_taur(self.V2)
            self.tr3 = 30
            self.tr4 = 30
            self.tc2 = stn_tauc(self.V2)

            self.Il1 = self.gl[0] * (self.V1 - self.El[0])
            self.Ina1 = self.gna[0] * (self.m1**3) * self.H1 * (self.V1 - self.Ena[0])
            self.Ik1 = self.gk[0] * ((0.75 * (1 - self.H1)) ** 4) * (self.V1 - self.Ek[0])
            self.It1 = self.gt[0] * (self.p1**2) * self.R1 * (self.V1 - self.Et)
            self.Igith = 1.4 * self.gsyn[5] * (self.V1 - self.Esyn[5]) * self.S4

            self.Il2 = self.gl[1] * (self.V2 - self.El[1])
            self.Ik2 = self.gk[1] * (self.N2**4) * (self.V2 - self.Ek[1])
            self.Ina2 = self.gna[1] * (self.m2**3) * self.H2 * (self.V2 - self.Ena[1])
            self.It2 = self.gt[1] * (self.a2**3) * (self.b2**2) * (self.V2 - self.Eca[1])
            self.Ica2 = self.gca[1] * (self.C2**2) * (self.V2 - self.Eca[1])
            self.Iahp2 = (
                self.gahp[1] * (self.V2 - self.Ek[1]) * (self.CA2 / (self.CA2 + self.k1[1]))
            )
            self.Igesn = 0.5 * (
                self.gsyn[0] * (self.V2 - self.Esyn[0]) * (self.S3 + self.S31)
            )
            self.Iappstn = 33 - self.pd * 10

            self.Il3 = self.gl[2] * (self.V3 - self.El[2])
            self.Ik3 = self.gk[2] * (self.N3**4) * (self.V3 - self.Ek[2])
            self.Ina3 = self.gna[2] * (self.m3**3) * self.H3 * (self.V3 - self.Ena[2])
            self.It3 = self.gt[2] * (self.a3**3) * self.R3 * (self.V3 - self.Eca[2])
            self.Ica3 = self.gca[2] * (self.s3**2) * (self.V3 - self.Eca[2])
            self.Iahp3 = (
                self.gahp[2] * (self.V3 - self.Ek[2]) * (self.CA3 / (self.CA3 + self.k1[2]))
            )
            self.Isnge = 0.5 * (
                self.gsyn[1] * (self.V3 - self.Esyn[1]) * (self.S2 + self.S21)
            )
            self.Igege = 0.5 * (
                self.gsyn[2] * (self.V3 - self.Esyn[2]) * (self.S31 + self.S32)
            )
            self.ISaveSTN[:, i - 1] = self.Isnge
            self.ISaveGPE[:, i - 1] = self.Igege
            self.Iappgpe = 21 - 13 * self.pd + self.r

            self.Il4 = self.gl[2] * (self.V4 - self.El[2])
            self.Ik4 = self.gk[2] * (self.N4**4) * (self.V4 - self.Ek[2])
            self.Ina4 = self.gna[2] * (self.m4**3) * self.H4 * (self.V4 - self.Ena[2])
            self.It4 = self.gt[2] * (self.a4**3) * self.R4 * (self.V4 - self.Eca[2])
            self.Ica4 = self.gca[2] * (self.s4**2) * (self.V4 - self.Eca[2])
            self.Iahp4 = (
                self.gahp[2] * (self.V4 - self.Ek[2]) * (self.CA4 / (self.CA4 + self.k1[2]))
            )
            self.Isngi = 0.5 * (
                self.gsyn[3] * (self.V4 - self.Esyn[3]) * (self.S2 + self.S21)
            )
            self.Igigi = 0.5 * (
                self.gsyn[4] * (self.V4 - self.Esyn[4]) * (self.S31 + self.S32)
            )
            self.Iappgpi = 22 - self.pd * 6

            self.vth[:, i - 1] = self.V1 + self.dt * (
                1
                / self.Cm
                * (
                    -self.Il1
                    - self.Ik1
                    - self.Ina1
                    - self.It1
                    - self.Igith
                    + self.Istim[i - 1]
                )
            )
            self.H1 = self.H1 + self.dt * ((self.h1 - self.H1) / self.th1)
            self.R1 = self.R1 + self.dt * ((self.r1 - self.R1) / self.tr1)

            self.vsn[:, i - 1] = self.V2 + self.dt * (
                1
                / self.Cm
                * (
                    -self.Il2
                    - self.Ik2
                    - self.Ina2
                    - self.It2
                    - self.Ica2
                    - self.Iahp2
                    - self.Igesn
                    + self.Iappstn
                    + self.Idbs[mini_step]
                )
            )  # pretty sure action replaces Idbs[i]
            self.N2 = self.N2 + self.dt * (0.75 * (self.n2 - self.N2) / self.tn2)
            self.H2 = self.H2 + self.dt * (0.75 * (self.h2 - self.H2) / self.th2)
            self.R2 = self.R2 + self.dt * (0.2 * (self.r2 - self.R2) / self.tr2)
            self.CA2 = self.CA2 + self.dt * (
                3.75 * 10**-5 * (-self.Ica2 - self.It2 - self.kca[1] * self.CA2)
            )
            self.C2 = self.C2 + self.dt * (0.08 * (self.c2 - self.C2) / self.tc2)
            a = np.where((self.vsn[:, i - 2] < -10) & (self.vsn[:, i - 1] > -10))[0]
            u = np.zeros(self.n)
            u[a] = self.gpeak / (self.tau * np.exp(-1)) / self.dt
            self.S2 = self.S2 + self.dt * self.Z2
            zdot = u - 2 / self.tau * self.Z2 - 1 / (self.tau**2) * self.S2
            self.Z2 = self.Z2 + self.dt * zdot
            # self.aSTN1[:, i-1] = (self.vsn[:,i-2]<-10 & self.vsn[:,i-1]>-10)

            self.vge[:, i - 1] = self.V3 + self.dt * (
                1
                / self.Cm
                * (
                    -self.Il3
                    - self.Ik3
                    - self.Ina3
                    - self.It3
                    - self.Ica3
                    - self.Iahp3
                    - self.Isnge
                    - self.Igege
                    + self.Iappgpe
                )
            )
            self.N3 = self.N3 + self.dt * (0.1 * (self.n3 - self.N3) / self.tn3)
            self.H3 = self.H3 + self.dt * (0.05 * (self.h3 - self.H3) / self.th3)
            self.R3 = self.R3 + self.dt * (1 * (self.r3 - self.R3) / self.tr3)
            self.CA3 = self.CA3 + self.dt * (
                1 * 10**-4 * (-self.Ica3 - self.It3 - self.kca[2] * self.CA3)
            )
            self.S3 = self.S3 + self.dt * (
                self.A[2] * (1 - self.S3) * Hinf(self.V3 - self.the[2])
                - self.B[2] * self.S3
            )
            # self.aGPe[:,i-1]=(self.vge[:,i-2]<-10 & self.vge[:,i-1]>-10)

            self.vgi[:, i - 1] = self.V4 + self.dt * (
                1
                / self.Cm
                * (
                    -self.Il4
                    - self.Ik4
                    - self.Ina4
                    - self.It4
                    - self.Ica4
                    - self.Iahp4
                    - self.Isngi
                    - self.Igigi
                    + self.Iappgpi
                )
            )

            self.N4 = self.N4 + self.dt * (0.1 * (self.n4 - self.N4) / self.tn4)
            self.H4 = self.H4 + self.dt * (0.05 * (self.h4 - self.H4) / self.th4)
            self.R4 = self.R4 + self.dt * (1 * (self.r4 - self.R4) / self.tr4)
            self.CA4 = self.CA4 + self.dt * (
                1 * 10**-4 * (-self.Ica4 - self.It4 - self.kca[2] * self.CA4)
            )
            a = np.where((self.vgi[:, i - 2] < -10) & (self.vgi[:, i - 1] > -10))[0]
            u = np.zeros(self.n)
            u[a] = self.gpeak1 / (self.tau * np.exp(-1)) / self.dt
            self.S4 = self.S4 + self.dt * self.Z4
            zdot = u - 2 / self.tau * self.Z4 - 1 / (self.tau**2) * self.S4
            self.Z4 = self.Z4 + self.dt * zdot

            self.n3t[:, i - 1] = self.N3
            self.h3t[:, i - 1] = self.H3
            self.r3t[:, i - 1] = self.R3
            self.ca3t[:, i - 1] = self.CA3
            self.s2t[:, i - 1] = self.S2
            self.s3t[:, i - 1] = self.S31
            self.sgi[:, i-2] = self.S4

            self.i = self.i + 1

            if (mini_step%100 == 0 and i > self.freq_time*100): 
                self.bursts[int(mini_step/100)] = self.calculate_bursts(self.vgi[:, i-self.freq_time*100:i])
                self.sgi_sum[int(mini_step/100)] = np.sum(np.abs(np.fft.fft(self.sgi[:, i-self.freq_time*100:i]))[:, 1:50])

    
            if i == self.tmax*100:
                self.new_state = self.sgi_sum
                self.reward = np.log(self.sgi_sum[-1]+1)
                return (self.new_state, self.reward, True)

        self.new_state = self.sgi_sum
        self.reward = np.log(self.sgi_sum[-1]+0.001)
        return (self.new_state, self.reward, False)
    
    def calculate_bursts(self, vgi_reading):
        """
        Calculates the number of bursts, defined as an instance 
        where one neuron fires 10ms or less after already firing
        """
        dists = []
        for i in range(self.n):
            peaks = find_peaks(vgi_reading[i], height=40)[0]
            for j in range(len(peaks)-1):
                dists.append(np.abs(peaks[j] - peaks[j+1]))
        bursts = len(np.where(np.array(dists) < 1000)[0])/self.n
        return bursts

    def calculate_pb(self, start, end) -> int:
        #last_1000_elements = self.vgi[0][-1000:]
        #return sum(abs(np.fft.fft(last_1000_elements[1 : self.tmax : 25]))[12:30])
        pb_sum = 0
        for i in range(len(self.vgi)):
            #pb = sum(abs(np.fft.fft(self.vgi[i][1 : self.tmax]))[13:35])
            pb_sum += sum(abs(np.fft.fft(self.vgi[i][start:end]))[13:35])
        return pb_sum/self.n

    def calculate_ei(self, b2: int | None = None) -> np.floating:
        m = self.vth.shape[0]  # Number of thalamic cells
        e = np.zeros(m)  # Error accumulator for each cell

        # Use the full range of self.smc_pulse
        b1 = 0
        if not b2:
            b2 = len(self.smc_pulse)

        for i in range(m):  # Loop over each thalamic cell
            compare = []  # Stores spike times for this cell
            # Detect spikes in the membrane potential
            for j in range(1, self.vth.shape[1]):
                if self.vth[i, j - 1] < -40 and self.vth[i, j] > -40:
                    compare.append(self.t[j])

            for p in range(b1, b2):  # Loop over each self.smc_pulse
                if p != b2 - 1:
                    # Current spike window
                    a = [
                        c
                        for c in compare
                        if self.smc_pulse[p] <= c < self.smc_pulse[p] + 25
                    ]
                    # Next spike window
                    b = [
                        c
                        for c in compare
                        if self.smc_pulse[p] + 25 <= c < self.smc_pulse[p + 1]
                    ]
                else:
                    # For the last spike window
                    a = [c for c in compare if self.smc_pulse[p] <= c < self.tmax]
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