'''
Author: Juan Santos Suárez
Version: Initial version

MyQLM version: myqlm/1.2.2-python-3.6.12
'''

import numpy as np
import matplotlib.pyplot as plt
from qat.lang.AQASM import Program, RY, CNOT, H, AbstractGate, QRoutine, SWAP, PH, I
from qat.core.console import display
from qat.qpus import PyLinalg
from qat.lang.AQASM.qftarith import QFT
from scipy.integrate import quad
from expectation_module import load_probabilities, load_function, load_U0, load_U1, load_Q
plt.rcParams['font.size']=17


'''
In this program we want to use Grover search combined with phase estimation to perform an amplitude estimation.
This algorithm is exactly the same as quantum counting but in this case theta encodes the expected value that we want to compute.

a      (float)    : lower bond of the interval
b      (float)    : upper bond of the interval
nqbits (int)      : number of qubits that will be used for discretizing and encoding our probability distribution
p      (function) : probability distribution that will be loaded. Must be positive and will be normalized automatically 
f      (function) : function for which the expectation value will be computed. Its image must be restricted to the interval [0, 1]
n_aux  (int)      : Number of qubits in the register used to calculate the Fourier transform
'''

a = 0.
b = 1.
nqbits = 6
n_aux  = 8

def p(x):
	return x*x
	
def f(x):
	return np.sin(x)
	
centers, probs, P_gate = load_probabilities(nqbits, p, a, b)
R_gate, y = load_function(centers, f, nqbits)

qprog = Program()
qbits = qprog.qalloc(nqbits+1)
qprog.apply(P_gate, qbits[:-1])
qprog.apply(R_gate, qbits)
	
Q_gate, U0_gate, U1_gate  = load_Q(nqbits, P_gate, R_gate)

# Define the auxiliary register and apply H to every qubit of it
q_aux = qprog.qalloc(n_aux)
for i, aux in enumerate(q_aux):
	qprog.apply(H, aux)
	# Apply Q**(2**i) controlled by the auxiliary register
	for _ in range(2**(i)):
		qprog.apply(Q_gate.ctrl(), aux, qbits)

# Apply the inverse Fourier transform
qprog.apply(QFT(n_aux).dag(), q_aux)
   
# Simulate the circuit
circuit = qprog.to_circ()
display(circuit, batchmode=True, max_depth=None)
result = PyLinalg().submit(circuit.to_job(qubits = [i+nqbits+1 for i in range(n_aux)])) #Measure only auxiliary qubits

states, probabilities = [], []
for res in result:
	states.append(res.state.int) # Store the decimal number that represents each state
	probabilities.append(res.probability) # Store its probability
	print("Aux qbits have value %s (the probability of getting this result is %s)"%(res.state, res.probability))

# Histogram those values
plt.figure(figsize=(12, 6))
plt.bar(states, probabilities)
plt.xlabel('Measured value')
plt.ylabel('Probability')
plt.tight_layout()
plt.savefig('ProbDistribution.png')

# Take the maximum probability
i_max = np.argsort(probabilities)[-1]


'''
The eigenvalues of the Grover operator are e^{\pm i*2*\theta}
Here, we are implementing the operator -Q, not Q. Tipically this is regarded
as a global phase, but when controlling that operator it is not. 
That - sign is just the same as considering that the operator is not marking the states that we are looking for, but rather the states that we are nor looking for.
Thus, we can consider that the eigenvalues of our implemented operator are e^{\pm i*2*\psi} with
\ket{\psi} = \cos{(phi)} \ket{\psi_1} + \sin{(\phi)} \ket{\psi_0}.
Obviously, \theta and \phi are related bu \phi = \theta + \pi, rememeber that the integral is codified in the amplitudes of \ket{\psi_1}
All in all we will be computing the angle \phi as 2*\phi = 2*pi*measured_value/2^{n_aux}
'''
# Compute phi for the measured value with its upper and lower limits
phis = (np.pi/(2.**n_aux))*np.array([states[i_max], states[i_max]+0.5, states[i_max]-0.5]) #+0.5 and -0.5 are the most extreme values that we could have for M if it was not an int

I = np.cos(phis)**2. #Remember that the integral is codified in the amplitudes of \ket{\psi_1}

print(u'Obtained value of the integral %f' % (I[0]))
print(u'Upper and lower bonds', I[1], I[2])


Integral = np.sum(probs*y)
print(u'Discretized value of the integral:', Integral)

def g(x):
	return f(x)*p(x)/(quad(p, a, b)[0])
print(u'Exact value of the integral:', quad(g, a, b))


