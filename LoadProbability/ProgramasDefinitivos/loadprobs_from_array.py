'''
Author: Juan Santos Suárez
Version: Initial version

MyQLM version: myqlm/1.2.2-python-3.6.12
'''

import numpy as np
import matplotlib.pyplot as plt
from qat.lang.AQASM import Program, RY, CNOT
from qat.core.console import display
from expectation_module import load_probabilities_from_array
from qat.qpus import PyLinalg

'''
The objective of this program is to initialize a quantum state codifying a probability distribution.
The best way to do this is to codify directly the square root of the probabilities in the quantum amplitudes,
as if it was an histogram.

INPUTS

a      (float)    : lower bond of the interval
b      (float)    : upper bond of the interval
nqbits (int)      : number of qubits that will be used for discretizing and encoding our probability distribution
p      (np.ndarray) : probability distribution that will be loaded. Must be positive and will be normalized automatically 
'''

# Obviously there should be enough qubits to load all the states
nqbits = 3
p = np.array([1., 2., 3., 4., 1., 2., 8. ,1.])

centers, probs, P_gate = load_probabilities_from_array(nqbits, p)
print(probs)
# Now create the quantum program and add the gate
qprog = Program()
qbits = qprog.qalloc(nqbits)
qprog.apply(P_gate, qbits)

# Export this program into a quantum circuit
circuit = qprog.to_circ()
display(circuit, batchmode=True, max_depth=None)

# Create a Quantum Processor Unit
linalgqpu = PyLinalg()

# Create a job
job = circuit.to_job()

# Submit the job to the QPU
result = linalgqpu.submit(job)

# Check the results
for i, sample in enumerate(result):
	print("State %s , probability %s, wanted probability %s" % (sample.state, sample.probability, probs[i]))

