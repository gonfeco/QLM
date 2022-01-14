'''
Author: Juan Santos Suárez
Version: Initial version

MyQLM version: myqlm/1.2.2-python-3.6.12
'''

import numpy as np
import matplotlib.pyplot as plt
from qat.lang.AQASM import Program, RY, CNOT, H, AbstractGate, QRoutine
from qat.core.console import display
from qat.qpus import PyLinalg
from expectation_module import load_probabilities, load_function, load_U0, load_U1, load_Q



'''
In this program we want to use Grover search to compute an expectation value.
To do that, we will repeat the calculations for different number of iterations and then perform a least squares fit.

a      (float)    : lower bond of the interval
b      (float)    : upper bond of the interval
nqbits (int)      : number of qubits that will be used for discretizing and encoding our probability distribution
p      (function) : probability distribution that will be loaded. Must be positive and will be normalized automatically 
f      (function) : function for which the expectation value will be computed.
k_max  (int)      : maximum number of times that the Grover search will be done.
'''

a = 0.
b = 1.
nqbits = 6
k_max = 10

def p(x):
	return x*x
	
def f(x):
	return np.sin(x)

# Define the gates
centers, probs, P_gate = load_probabilities(nqbits, p, a, b)
R_gate, y = load_function(centers, f, nqbits)

Q_gate, U0_gate, U1_gate  = load_Q(nqbits, P_gate, R_gate)

ks = np.arange(0, k_max+1)

Ps_exact = np.array([])
Ps_shots = np.array([])
shots = [0, 1000] # We try an example with infinite shots and an example with 1000 shots
#Perform the search for different ks
for k in ks:
	qprog = Program()
	qbits = qprog.qalloc(nqbits+1)
	qprog.apply(P_gate, qbits[:-1])
	qprog.apply(R_gate, qbits)
	for _ in range(k):
		qprog.apply(Q_gate, qbits)
	circuit = qprog.to_circ()
	result = PyLinalg().submit(circuit.to_job(qubits=[nqbits]))
	for res in result:
		if int(res.state[0]) == 1:
			P = res.probability
			Ps_exact = np.append(Ps_exact, P)
	result = PyLinalg().submit(circuit.to_job(nbshots = shots[1], qubits=[nqbits]))
	trig  = True
	for res in result:
		if int(res.state[0]) == 1:
			trig = False
			P = res.probability
			Ps_shots = np.append(Ps_shots, P)
	if trig: #Sometimes the probability is so small that they might be no measurements
		Ps_shots = np.append(Ps_shots, 0.)
	
import scipy.optimize as so

# Define the function to fit and plot the results
def fun(k, theta):
	return np.sin((2*k+1)*theta)**2

Integral = np.sum(probs*y)
print('Expected value of the integral', Integral)
theta_t = np.arcsin(np.sqrt(Integral))
print('True value of theta', theta_t)
par = [theta_t] # We use the theorical value of theta to estimate the parameter of the fit. This is like cheating!

sol = so.curve_fit(fun, ks, Ps_exact, p0=par, bounds=(0., np.pi/2), maxfev=10000)
theta_0 = sol[0]
stheta = np.sqrt(np.diag(sol[1]))
klin = np.linspace(ks[0], ks[-1], 1000)
Plin_0 = fun(klin, theta_0)
a_0 = np.sin(theta_0)**2
print('\n')
print('Results without shots')
print('Theta =', theta_0[0], 'utheta =', stheta[0])
print('Integral =', a_0[0])

sol = so.curve_fit(fun, ks, Ps_shots, p0=par, bounds=(0., np.pi/2), maxfev=10000)
theta_1 = sol[0]
stheta = np.sqrt(np.diag(sol[1]))
klin = np.linspace(ks[0], ks[-1], 1000)
Plin_1 = fun(klin, theta_1)
a_1 = np.sin(theta_1)**2

print('\n')
print('Results with shots')
print('Theta =', theta_1[0], 'utheta =', stheta[0])
print('Integral =', a_1[0])

# Plot the results

plt.rcParams['font.size']=17
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
axs[0].plot(ks, Ps_exact, '*', color='teal', label='Obtained values')
axs[0].plot(klin, Plin_0, '--', color='teal', label=r'$\sin((2k+1)\theta)^2, \theta = %f$' % (theta_0))
axs[0].set_xlabel('k')
axs[0].set_ylabel('P')
axs[0].set_title('Exact result \n' + r'$\theta = %f, a= %f$' % (theta_0, a_0))

axs[1].plot(ks, Ps_shots, '*', color='teal', label='Obtained values')
axs[1].plot(klin, Plin_1, '--', color='teal', label=r'$\sin((2k+1)\theta)^2, \theta = %f$' % (theta_1))
axs[1].set_xlabel('k', fontsize=17)
axs[1].set_ylabel('P', fontsize=17)
axs[1].set_title('$n_{shots}=%i$' % (shots[1]) + '\n' + r'$\theta = %f, a= %f$' % (theta_1, a_1))

fig.savefig('kvsp.png')

