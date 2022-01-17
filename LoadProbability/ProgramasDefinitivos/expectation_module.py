#!/usr/bin/env python

"""
Author: Juan Santos Suárez
Version: Initial version

MyQLM version: myqlm/1.2.2-python-3.6.12


The objective of this module is to initialize a quantum state codifying a probability distribution.
The best way to do this is to codify directly the square root of the probabilities in the quantum amplitudes,
as if it was an histogram.
Then, the function load_function will allow us to load a function on those probabilities using an auxiliary qubit;
the function will be codified on those qubits in which that qubit's value is 1
After that, we can create the Grover operator Q
"""

import numpy as np
from qat.lang.AQASM import Program, RY, CNOT, AbstractGate, QRoutine
from qat.lang.AQASM import Z, X, H

def get_histogram(p, a, b, nbin):
	"""
	Given a function p, convert it into a histogram. The function must be positive, the normalization is automatic.
	Note that instead of having an analytical expression, p could just create an arbitrary vector of the right dimensions and positive amplitudes
	so that this procedure could be used to initialize any quantum state with real amplitudes
	
	a    (float)    = lower limit of the interval
	b    (float)    = upper limit of the interval
	p    (function) = function that we want to convert to a probability mass function. It does not have to be normalized but must be positive in the interval
	nbin (int)      = number of bins in the interval
	"""
	step = (b-a)/nbin
	centers = np.array([a+step*(i+1/2) for i in range(nbin)]) #Calcula directamente los centros de los bines

	prob_n = p(centers)
	assert np.all(prob_n>=0.), 'Probabilities must be positive, so p must be a positive function'
	probs = prob_n/np.sum(prob_n)
	assert np.isclose(np.sum(probs), 1.), 'Probability is not getting normalized properly'
	return centers, probs
	
	
def multiplexor_RY_m_recurs(qprog, qbits, thetas, m, j, sig = 1.):
	"""	
	Auxiliary function to create the recursive part of a multiplexor that applies an RY gate
	
	qprog = Quantum Program in which we want to apply the gates
	qbits = Nmber of qubits of the quantum program
	thetas (np.ndarray) = numpy array containing the set of angles that we want to apply
	m   (int) = number of remaining controls
	j   (int) = index of the target qubits
	sig (float) = accounts for wether our multiplexor is being decomposed with its lateral CNOT at the right or at the left, even if that CNOT is not present because it cancelled out (its values can only be +1. and -1.)
	"""
	assert isinstance(m, int), 'm must be an integer'
	assert isinstance(j, int), 'j must be an integer'
	assert sig == 1. or sig == -1., 'sig can only be -1. or 1.'
	if m > 1:
		# If there is more that one control, the multiplexor shall be decomposed.
		# It can be checked that the right way to decompose it taking into account the simplifications is as
		x_l = 0.5*np.array([thetas[i]+sig*thetas[i+len(thetas)//2] for i in range (len(thetas)//2)]) #left angles
		x_r = 0.5*np.array([thetas[i]-sig*thetas[i+len(thetas)//2] for i in range (len(thetas)//2)]) #right angles
		
		multiplexor_RY_m_recurs(qprog, qbits, x_l, m-1, j, 1.)
		qprog.apply(CNOT, qbits[j-m], qbits[j])
		multiplexor_RY_m_recurs(qprog, qbits, x_r, m-1, j, -1.)
		
		# Just for clarification, if we hadn't already simplify the CNOTs, the code should have been
		# if sign == -1.:
		# 	multiplexor_RY_m_recurs(qprog, qbits, x_l, m-1, j, -1.)
		# qprog.apply(CNOT, qbits[j-m], qbits[j])
		# multiplexor_RY_m_recurs(qprog, qbits, x_r, m-1, j, -1.)
		# qprog.apply(CNOT, qbits[j-m], qbits[j])
		# if sign == 1.:
		# 	multiplexor_RY_m_recurs(qprog, qbits, x_l, m-1, j, 1.)
		
	else: 
		# If there is only one control just apply the Ry gates
		qprog.apply(RY(thetas[0]+sig*thetas[1]), qbits[j])
		qprog.apply(CNOT, qbits[j-1], qbits[j])
		qprog.apply(RY(thetas[0]-sig*thetas[1]), qbits[j])
		
			
def multiplexor_RY_m(qprog, qbits, thetas, m, j):
	"""
	Create a multiplexor that applies an RY gate on a qubit controlled by the former m qubits
	It will have its lateral cnot on the right.

	qprog = Quantum Program in which we want to apply the gates
	qbits = Nmber of qubits of the quantum program
	thetas (np.ndarray) = numpy array containing the set of angles that we want to apply
	m      (int) = number of remaining controls
	j      (int) = index of the target qubits
	"""
	multiplexor_RY_m_recurs(qprog, qbits, thetas, m, j)
	qprog.apply(CNOT, qbits[j-m], qbits[j])
	
def load_probabilities(nqbits, p, a, b):
	"""
	Function that creates a gate that loads a probability distribution given by a function into a quantum state
	Note that instead of having an analytical expression, p could just create an arbitrary vector of the right dimensions and positive amplitudes
	so that this procedure could be used to initialize any quantum state with real amplitudes
	
	
	PARAMETERS:
	nqbits (int)      = number of qubits used. It will set the number of bins to 2**nqubits
	p      (function) = function that we want to convert to a probability mass function. It does not have to be normalized but must be positive in the interval
	a      (float)    = lower limit of the interval
	b      (float)    = upper limit of the interval
	
	
	
	RETURNS:
	centers (np.ndarray) = array containing the center of each bin
	probs   (np.ndarray) = array conntainig the probability of each bin
	P_gate  (ParamGate)  = quantum gate that loads the probability distribution p in nqbits qubits
	"""
	
	assert isinstance(a, (int, float)), 'a must be a real number'
	assert isinstance(b, (int, float)), 'b must be a real number'
	assert isinstance(nqbits, int), 'nqbits must be an integer'
	nbins = 2**nqbits
	assert b>a, 'b must be the upper limit of the interval and a the lower limit'   
	assert callable(p), 'p must be a function'
	assert isinstance(p(np.array([a, b])), np.ndarray), 'the output of the function p must be a numpy array'

	# First of all compute how is the probability distribution that we want to load
	centers, probs = get_histogram(p, a, b, nbins)
	P = AbstractGate("P", [int])
	def P_generator(nqbits):
		rout = QRoutine()
		reg = rout.new_wires(nqbits)
		# Now go iteratively trough each qubit computing the probabilities and adding the corresponding multiplexor
		for m in range(nqbits):
			n_parts = 2**(m+1) #Compute the number of subzones which the current state is codifying
			edges = np.array([a+(b-a)*(i)/n_parts for i in range(n_parts+1)]) #Compute the edges of that subzones
		
			# Compute the probabilities of each subzone by suming the probabilities of the original histogram.
			# There is no need to compute integrals since the limiting accuracy is given by the original discretization.
			# Moreover, this approach allows to handle non analytical probability distributions, measured directly from experiments
			p_zones = np.array([np.sum(probs[np.logical_and(centers>edges[i],centers<edges[i+1])]) for i in range(n_parts)])
			# Compute the probability of standing on the left part of each zone 
			p_left = p_zones[[2*j for j in range(n_parts//2)]]
			# Compute the probability of standing on each zone (left zone + right zone)
			p_tot = p_left + p_zones[[2*j+1 for j in range(n_parts//2)]]
			
			# Compute the rotation angles
			thetas = np.arccos(np.sqrt(p_left/p_tot))

			if m == 0:
				# In the first iteration it is only needed a RY gate
				rout.apply(RY(2*thetas[0]), reg[0])
			else:
				# In the following iterations we have to apply multiplexors controlled by m qubits
				# We call a function to construct the multiplexor, whose action is a block diagonal matrix of Ry gates with angles theta
				multiplexor_RY_m(rout, reg, thetas, m, m)
		return rout
	P.set_circuit_generator(P_generator)
	P_gate = P(nqbits)
	return centers, probs, P_gate


def load_function(centers, f, nqbits):
	"""
	Load the values of the function f on the states in which the value of the auxiliary qubit is 1 once the probabilities are already loaded.
	
	PARAMETERS:
	centers (np.ndarray) : center of each considered bin
	f       (function)   : function that we want to load. Its image must be contained in [0, 1]
	nqbits  (int)        : number of qubits used WITHOUT counting the auxiliary qubit
	
	RETURNS:
	R_gate (ParamGate) : gate that loads the function into the amplitudes
	y      (np.ndarray)  : array containing the value of the function in each bin
	"""
	y = f(centers)
	assert np.all(y<=1.), 'The image of the function must be less than 1. Rescaling is required'
	assert np.all(y>=0.), 'The image of the function must be greater than 0. Rescaling is required'
	assert isinstance(y, np.ndarray), 'the output of the function p must be a numpy array'
	thetas = np.arcsin(np.sqrt(y))

	R = AbstractGate("R", [int] + [float for theta in thetas])
	def R_generator(nqbits, *thetas):
		rout = QRoutine()
		reg = rout.new_wires(nqbits+1)
		multiplexor_RY_m(rout, reg, thetas, nqbits, nqbits)
		return rout
	R.set_circuit_generator(R_generator)
	R_gate = R(nqbits, *thetas)
	return R_gate, y
	

def load_U0(nqbits):
	"""
	Creates the gate U0, whose action is to flip the sign of the marked states.
	Marked states are those in which the auxiliary qubit has value 1, so this is achieved with a Z gate
	acting on that qubit. This function is more or less unnecessary.
	
	ARGUMENTS:
	nqbits (int) : number of qubits without including the auxiliary one
	
	RETURNS:
	U0_gate (ParamGate)
	"""
	U0 = AbstractGate("U0", [int])
	def U0_generator(nqbits):
		rout = QRoutine()
		reg = rout.new_wires(nqbits+1)
		rout.apply(Z, reg[-1])
		return rout
	U0.set_circuit_generator(U0_generator)
	U0_gate = U0(nqbits)
	return U0_gate

def load_U1(nqbits, P_gate, R_gate):
	"""
	Function that creates the gate which performs the operation of flipping the sign of the component along psi
	
	ARGUMENTS:
	nqbits (int) : number of qubits
	R_gate (ParamGate) : Gate that implements the operator R
	P_gate (ParamGate) : Gate that implements the operator P
	
	RETURNS:
	U1_gate (ParamGate)
	"""
	U1 = AbstractGate("U1", [int])
	def U1_generator(nqbits):
		rout = QRoutine()
		reg = rout.new_wires(nqbits+1)
		rout.apply(R_gate.dag(), reg)
		#rout.apply(P_gate.dag(), reg)
		rout.apply(P_gate.dag(), reg[:-1])
		for wire in reg:
			rout.apply(X, wire)
		rout.apply(H, reg[-1])
		s = 'X' + '.ctrl()'*nqbits
		rout.apply(eval(s), reg)
		rout.apply(H, reg[-1])
		for wire in reg:
			rout.apply(X, wire)
		#rout.apply(P_gate, reg)
		rout.apply(P_gate, reg[:-1])
		rout.apply(R_gate, reg)
		return rout
	U1.set_circuit_generator(U1_generator)
	U1_gate = U1(nqbits)
	return U1_gate
	
def load_Q(nqbits, P_gate, R_gate):
	"""
	Creates the Grover operator Q except a global phase, so it really creates the gate -Q
	
	ARGUMENTS:
	nqbits  (int)       : number of qubits
	U0_gate (ParamGate) : Gate U_psi_0
	U1_gate (ParamGate) : Gate U_psi
	
	RETURNS:
	Q_gate (ParamGate)  : Gate -Q=U1U0
	"""
	U0_gate = load_U0(nqbits)
	U1_gate = load_U1(nqbits, P_gate, R_gate)
	Q = AbstractGate("Q", [int])
	def Q_generator(nqbits):
		rout = QRoutine()
		reg  = rout.new_wires(nqbits+1)
		rout.apply(U0_gate, reg)
		rout.apply(U1_gate, reg)
		return rout
	Q.set_circuit_generator(Q_generator)
	Q_gate = Q(nqbits)
	return Q_gate, U0_gate, U1_gate
	
def load_probabilities_from_array(nqbits, p):
	"""
	Function that loads an arbitrary real valued positive array p as probabilities on a quantum state
	
	
	PARAMETERS:
	nqbits (int)        = number of qubits used. It will set the number of bins to 2**nqubits
	p      (np.ndarray) = array containing the probability of each bin
	
	
	RETURNS:
	centers (np.ndarray) = array containing the center of each bin
	probs   (np.ndarray) = array conntainig the probability of each bin
	P_gate  (ParamGate)  = quantum gate that loads the probability distribution p in nqbits qubits
	"""

	assert isinstance(nqbits, int), 'nqbits must be an integer'
	nbins = 2**nqbits
	a = 0
	b = nbins-1
	# First of all compute how is the probability distribution that we want to load
	step = (b-a)/nbins
	centers = np.array([a+step*(i+1/2) for i in range(nbins)]) #Calcula directamente los centros de los bines

	probs = p/np.sum(p)
	P = AbstractGate("P", [int])
	def P_generator(nqbits):
		rout = QRoutine()
		reg = rout.new_wires(nqbits)
		# Now go iteratively trough each qubit computing the probabilities and adding the corresponding multiplexor
		for m in range(nqbits):
			n_parts = 2**(m+1) #Compute the number of subzones which the current state is codifying
			edges = np.array([a+(b-a)*(i)/n_parts for i in range(n_parts+1)]) #Compute the edges of that subzones
		
			# Compute the probabilities of each subzone by suming the probabilities of the original histogram.
			# There is no need to compute integrals since the limiting accuracy is given by the original discretization.
			# Moreover, this approach allows to handle non analytical probability distributions, measured directly from experiments
			p_zones = np.array([np.sum(probs[np.logical_and(centers>edges[i],centers<edges[i+1])]) for i in range(n_parts)])
			# Compute the probability of standing on the left part of each zone 
			p_left = p_zones[[2*j for j in range(n_parts//2)]]
			# Compute the probability of standing on each zone (left zone + right zone)
			p_tot = p_left + p_zones[[2*j+1 for j in range(n_parts//2)]]
			
			# Compute the rotation angles
			thetas = np.arccos(np.sqrt(p_left/p_tot))

			if m == 0:
				# In the first iteration it is only needed a RY gate
				rout.apply(RY(2*thetas[0]), reg[0])
			else:
				# In the following iterations we have to apply multiplexors controlled by m qubits
				# We call a function to construct the multiplexor, whose action is a block diagonal matrix of Ry gates with angles theta
				multiplexor_RY_m(rout, reg, thetas, m, m)
		return rout
	P.set_circuit_generator(P_generator)
	P_gate = P(nqbits)
	return centers, probs, P_gate

