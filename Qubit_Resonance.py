#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import random

#Define constants
h_bar = 1.055 * 10**(-34)
R_period=6*10**(-6)  #seconds
R_freq=2*np.pi/(R_period) #seconds^-1
D_rate=0.2*R_freq
State_initial = np.array([1, 0]) # a = 1 | b = 0

#Analytical calculation of down state population
def mod_b_sq(R_freq, D_rate, t):
    B = R_freq**2 / (2 *R_freq**2 + D_rate**2)
    B*= 1 - np.exp(-3 * D_rate * t / 4) * (np.cos(np.sqrt(R_freq**2 - (D_rate**2/16)) * t)+( 3 * D_rate / np.sqrt(16 * R_freq**2 - D_rate**2)) * np.sin(np.sqrt(R_freq**2 - D_rate**2 / 16) * t))
    return B

#Simulation
times = np.arange(0,10*R_period, 0.01*R_period)
B_values=mod_b_sq(R_freq, D_rate, times)

class QuantumSystem:
    def __init__(self, R_period, R_freq, D_rate, State_initial):
        self.H_eff = (h_bar / 2) * np.array([[0, R_freq], [R_freq, -(D_rate) * 1j]]) #Non-Hermitian hamiltonian
        self.Jump = np.sqrt(D_rate / 2) * np.array([[0, 1], [0, 0]]) #Spontaneous jump operator
        self.State = State_initial
        self.dt = 10**-2 * R_period

    def evolve_state(self):
        self.State = self.State - (1j/h_bar) * self.dt * self.H_eff.dot(self.State) #Evolves sytem under non-Hermitian Schrödinger dynamics

    def q_jump(self):
        self.State = self.Jump.dot(self.State)

    def norm_sq(self):
        return np.conj(self.State[0]) * self.State[0] + np.conj(self.State[1]) * self.State[1]  #Normalise evolved system

    # Montecarlo simulation
    def run_simulation(self, times):
        Q_traj = [self.State[1]]
        for _ in times[1:]:
            self.evolve_state()
            r = random.random()
            if r < self.norm_sq():
                self.State = self.State / np.real(np.sqrt(self.norm_sq()))
            else:
                self.q_jump()
                self.State = self.State / np.real(np.sqrt(self.norm_sq()))
            Q_traj.append(np.conj(self.State[1]) * self.State[1])
        return np.real(Q_traj)


#Simulate for 10000 qubits
system = QuantumSystem(R_period, R_freq, D_rate, State_initial)
times = np.arange(0, 10 * R_period, system.dt)
N_runs = 10000
B = np.zeros(len(times))

for _ in range(N_runs):
    system.State = State_initial
    B += system.run_simulation(times)

values_MC = B / N_runs

#Simulate for 1000 qubits
system = QuantumSystem(R_period, R_freq, D_rate, State_initial)
times = np.arange(0, 10 * R_period, system.dt)

N_runs_2 = 1000
B_2 = np.zeros(len(times))

for _ in range(N_runs_2):
    system.State = State_initial
    B_2 += system.run_simulation(times)

values_MC_2 = B_2 / N_runs_2

# Plot the time evolution and analytical solution
fig1 = plt.figure(1)

frame1 = fig1.add_axes((0,0, 1, 1))

plt.plot(times/(R_period*0.01), values_MC, linestyle = 'dashed', color = 'dodgerblue', label = "10000 qubits") 
plt.plot(times/(R_period*0.01), values_MC_2, linestyle = 'dotted', color = "crimson", label = "1000 qubits") 

plt.plot(times/(R_period*0.01),B_values, color = 'black', linewidth = 0.7, label = "Analytic Solution") 
plt.ylabel('|b|$^2$')
frame1.set_xticklabels([]) 
plt.legend(loc="upper right")


