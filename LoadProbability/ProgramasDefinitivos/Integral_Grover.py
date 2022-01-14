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
This process will have 4 steps. First, load the probability distribution using nqbits qubits.
Then, load the function for which we want to compute the expectation value.
After that has been done, create the Grover operation and apply it k times
Finally, simulate the circuit and try to get the result

a      (float)    : lower bond of the interval
b      (float)    : upper bond of the interval
nqbits (int)      : number of qubits that will be used for discretizing and encoding our probability distribution
p      (function) : probability distribution that will be loaded. Must be positive and will be normalized automatically 
f      (function) : function for which the expectation value will be computed.
k      (int)      : number of times that the Grover operation will be applied
'''

a = 0.
b = 1.
nqbits = 4
k = 1

def p(x):
	return x*x
	
def f(x):
	return np.sin(x)

# Define all gates

centers, probs, P_gate = load_probabilities(nqbits, p, a, b)
R_gate, y = load_function(centers, f, nqbits)

Q_gate, U0_gate, U1_gate  = load_Q(nqbits, P_gate, R_gate)

qprog = Program()
qbits = qprog.qalloc(nqbits+1)
qprog.apply(P_gate, qbits[:-1])
qprog.apply(R_gate, qbits)

# Checked that probability distribution and function are loaded on the tests

# The only thing left is to apply Grover operator k times

for _ in range(k):
	qprog.apply(Q_gate, qbits)
circuit = qprog.to_circ()
display(circuit, batchmode=True, max_depth=None)

Integral = np.sum(probs*y)
print(u'Expected value of the integral:', Integral)
k_t = 0.5*(np.pi/(2*np.arcsin(np.sqrt(Integral)))-1.)
print(u'Number of Grover iterations needed:', k_t)
result = PyLinalg().submit(circuit.to_job(qubits=[nqbits]))
for res in result:
	print("Qubit %s has value %s (the probability of getting this result is %s)"%(nqbits, int(res.state[0]), res.probability))
	if int(res.state[0]) == 1:
		P = res.probability
		theta = (1./(2*k+1))*np.array([np.arcsin(np.sqrt(res.probability)), np.pi-np.arcsin(np.sqrt(res.probability)), np.pi+np.arcsin(np.sqrt(res.probability)), ((2.*np.pi)-np.arcsin(np.sqrt(res.probability)))])
		integral = np.sin(theta)**2
print(u'First 4 possible values of the integral:')
print(integral)
print(u'There is no reliable way to distinguish which is the correct value apart from comparing with the results obtained with other number of iterations.')

