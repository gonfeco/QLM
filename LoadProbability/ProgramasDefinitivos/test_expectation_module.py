'''
Author: Juan Santos Suárez
Version: Initial version

MyQLM version: myqlm/1.2.2-python-3.6.12
'''

import unittest
from test import support
import numpy as np
from expectation_module import load_probabilities, load_function, load_U0, load_U1, load_Q
from qat.lang.AQASM import Program, RY, CNOT, AbstractGate, QRoutine
from qat.qpus import PyLinalg

class Test(unittest.TestCase):

	def test_loadprobs_works(self):
		"""
		Test that the loading of probabilities works properly
		"""
		a = 0.
		b = 1.
		
		test_functions = [lambda x : x*x, lambda x : x, lambda x: np.exp(x), lambda x: np.exp(-x**2)]
		for p in test_functions:
			nqbits = 8
			nbins = 2**nqbits
			qprog = Program()
			qbits = qprog.qalloc(nqbits)
			centers, probs, P_gate = load_probabilities(nqbits, p, a, b)
			qprog.apply(P_gate, qbits)
			circuit = qprog.to_circ()
			linalgqpu = PyLinalg()
			job = circuit.to_job()
			result = linalgqpu.submit(job)

			for i, sample in enumerate(result):
				assert np.isclose(sample.probability, probs[i]), 'The load of the function is not working properly'
		

	def test_loadprobs_negative(self):
		"""
		Test that the loading of probabilities gets some assert if probabilities are negative
		"""
		a = 0.
		b = 1.
		
		test_functions = [lambda x : -x*x, lambda x : -x, lambda x: -np.exp(x), lambda x: -np.exp(-x**2)]
		for p in test_functions:
			nqbits = 8
			try:
				centers, probs = load_probabilities(nqbits, p, a, b)
				raise ValueError
			except AssertionError as msg:
				pass
			except ValueError:
				raise ValueError('There should be something checking if probabilities are negative')
	
	def test_loadfunction_works(self):
		a = 0.
		b = 1.
		nqbits = 8
		def p(x):
			return x*x
			
		test_functions =  [lambda x : x*x, lambda x : x, lambda x: np.exp(-x**2)]
		for f in test_functions:
			centers, probs, P_gate = load_probabilities(nqbits, p, a, b)
			R_gate, y = load_function(centers, f, nqbits)

			qprog = Program()
			qbits = qprog.qalloc(nqbits+1)
			qprog.apply(P_gate, qbits[:-1])
			qprog.apply(R_gate, qbits)
			
			circuit = qprog.to_circ()
			result = PyLinalg().submit(circuit.to_job())
			j = 0
			for sample in result:
				if int(sample.state[nqbits]) == 1:
					assert(np.isclose(sample.probability, probs[j]*y[j]))
					j += 1

	def test_loadfunction_negative(self):
		a = 0.
		b = 2.*np.pi
		nqbits = 8
		def p(x):
			return x*x
			
		test_functions =  [lambda x : np.exp(x), lambda x : np.sin(x)]
		for f in test_functions:
			centers, probs, P_gate = load_probabilities(nqbits, p, a, b)
			try:
				R_gate, y = load_function(centers, f, nqbits)
				raise ValueError
			except AssertionError as msg:
				pass
			except ValueError:
				raise ValueError('There should be something checking if the functions image is allowed')
				
	
	def test_Q(self):
		a = 0.
		b = 1.
		nqbits = 8
		def p(x):
			return x*x
			
		def f(x):
			return np.sin(x)
			
		centers, probs, P_gate = load_probabilities(nqbits, p, a, b)
		R_gate, y = load_function(centers, f, nqbits)
		a = np.sum(y*probs)
		theta = np.arcsin(np.sqrt(a))
		P_t = np.sin(3.*theta)**2
		qprog = Program()
		qbits = qprog.qalloc(nqbits+1)
		qprog.apply(P_gate, qbits[:-1])
		qprog.apply(R_gate, qbits)
			
		Q_gate, U0_gate, U1_gate  = load_Q(nqbits, P_gate, R_gate)
		qprog.apply(Q_gate, qbits)
		circuit = qprog.to_circ()
		result = PyLinalg().submit(circuit.to_job(qubits=[nqbits]))
		for res in result:
			if int(res.state[0]) == 1:
				P = res.probability
				assert np.isclose(P_t, P), 'Grover operator is not working properly'

if __name__ == '__main__':
    unittest.main()
