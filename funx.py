#qtf
import numpy as np
from numpy import pi
from qiskit import QuantumCircuit

# audio_tools
from pydub import AudioSegment
from pydub.playback import play
from pydub.generators import Sine
import os
#from tools.tools import prepare_circuit, truncate_samples, normalize
#from tools.qft import qft

# tools
from qiskit import QuantumCircuit, QuantumRegister, Aer, execute
from qiskit.visualization import plot_histogram, plot_state_city
from math import sqrt, pi, log
import numpy as np
import matplotlib.pyplot as plt

# qtf
def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cu1(pi/2**(n-qubit), qubit, n)
    #note the recursion
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def prepare_circuit(samples, normalize=True):

    """
    Args:
    amplitudes: List - A list of amplitudes with length equal to power of 2
    normalize: Bool - Optional flag to control normalization of samples, True by default
    Returns:
    circuit: QuantumCircuit - a quantum circuit initialized to the state given by amplitudes
    """
    num_amplitudes = len(samples)
    assert isPow2(num_amplitudes), 'len(amplitudes) should be power of 2'

    num_qubits = int(getlog2(num_amplitudes))
    q = QuantumRegister(num_qubits)
    qc = QuantumCircuit(q)

    # normalize samples for clean math
    if(normalize):
        ampls = samples / np.linalg.norm(samples)
    else:
        ampls = samples

    qc.initialize(ampls, [q[i] for i in range(num_qubits)])

    return qc

# audio_tools
def get_frequencies_from_fft(fft, frame_rate, n=5):
    """
    Returns top n loudest frequencies in fft
    """

    N = len(fft)
    T = 1.0/frame_rate

    fstep = (1/(2.0*T))/((N//2)-1)



    yf = fft[:N//2]
    indexes = np.argsort(yf)
    freqs = indexes*fstep

    return freqs[:n]


def get_audio_samples(segment, n_qubits):
    """
    get the first 2^num_qubits samples from the given audio segment
    """

    samples = segment.get_array_of_samples()
    assert len(samples) >= 2**n_qubits, 'Audio segment too short'

    return samples[:2**n_qubits]


def prepare_circuit_from_audiosegment(segment, n_qubits):
    """
    Prepares a quantum circuit with n_qubits initialized with a quantum state corresponding
    to the samples from the audio segment
    """

    samples = get_audio_samples(segment, n_qubits)
    qc = prepare_circuit(samples)

    return qc

def prepare_circuit_from_samples(samples, n_qubits):
    """
    Prepares a quantum circuit with n_qubits initialized with a quantum state corresponding
    to the samples provided
    """

    samples = truncate_samples(samples, n_qubits)
    qc = prepare_circuit(samples)

    return qc

def qft_rotations(circuit, n):
    """Performs qft on the first n qubits in circuit (without swaps)"""
    if n == 0:
        return circuit
    n -= 1
    circuit.h(n)
    for qubit in range(n):
        circuit.cu1(pi/2**(n-qubit), qubit, n)
    #note the recursion
    qft_rotations(circuit, n)

def swap_registers(circuit, n):
    for qubit in range(n//2):
        circuit.swap(qubit, n-qubit-1)
    return circuit

def qft(circuit, n):
    """QFT on the first n qubits in circuit"""
    qft_rotations(circuit, n)
    swap_registers(circuit, n)
    return circuit

def inverse_qft(circuit, n):
    """Does the inverse QFT on the first n qubits in circuit"""
    qft_circ = qft(QuantumCircuit(n), n)
    invqft_circ = qft_circ.inverse()
    circuit.append(invqft_circ, circuit.qubits[:n])
    return circuit.decompose() # .decompose() allows to see the individual gates

# tools
def plot_samples(samples):

    plt.plot(list(range(len(samples))), samples)


def get_fft_from_counts(counts, n_qubits):

    out = []
    keys = counts.keys()
    for i in range(2**n_qubits):
        id = get_bit_string(i, n_qubits)
        if(id in keys):
            out.append(counts[id])
        else:
            out.append(0)

    return out

def get_bit_string(n, n_qubits):
    """
    Returns the binary string of an integer with n_qubits characters
    """

    assert n < 2**n_qubits, 'n too big to binarise, increase n_qubits or decrease n'

    bs = "{0:b}".format(n)
    bs = "0"*(n_qubits - len(bs)) + bs

    return bs


def isPow2(x):
    return (x!=0) and (x & (x-1)) == 0

def getlog2(x):

    return (log(x)/log(2))


def normalize(samples):

    norm = np.linalg.norm(samples)

    return samples/norm, norm

def truncate_samples(samples, n_qubits):

    if(len(samples) <= 2**n_qubits):
        pass
    else:
        samples = samples[:2**n_qubits]
    return samples


def prepare_circuit(samples, normalize=True):

    """
    Args:
    amplitudes: List - A list of amplitudes with length equal to power of 2
    normalize: Bool - Optional flag to control normalization of samples, True by default
    Returns:
    circuit: QuantumCircuit - a quantum circuit initialized to the state given by amplitudes
    """


    num_amplitudes = len(samples)
    assert isPow2(num_amplitudes), 'len(amplitudes) should be power of 2'

    num_qubits = int(getlog2(num_amplitudes))
    q = QuantumRegister(num_qubits)
    qc = QuantumCircuit(q)

    if(normalize):
        ampls = samples / np.linalg.norm(samples)
    else:
        ampls = samples

    qc.initialize(ampls, [q[i] for i in range(num_qubits)])

    return qc
