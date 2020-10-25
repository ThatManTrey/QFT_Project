# followed tutorial from here:
# https://sarangzambare.github.io/jekyll/update/2020/06/13/quantum-frequencies.html
from funx import *
from pydub.generators import Sine
from pydub import AudioSegment
from pydub.playback import play
#%config InlineBackend.figure_format = 'svg'
from qiskit import Aer, execute, QuantumCircuit, IBMQ
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor
from math import sqrt, pi

# Step 1 - Input Audio file
n_qubits = 3 # should be 16, but too many bits for IBMQ
            # at 4 & 5, introduces too much noise
n_samples = 2**n_qubits # sample of sound to limit computation
audio = AudioSegment.from_file('900hz.wav')
audio = audio.set_frame_rate(2000)
frame_rate = audio.frame_rate # 2000
samples = audio.get_array_of_samples()[:n_samples]

# show sample plot
#plt.xlabel('sample_num')
#plt.ylabel('value')
#plt.plot(list(range(len(samples))), samples)

# and prepare a quantum state (prepare circuit)
qcs = prepare_circuit_from_samples(samples, n_qubits)

# Step 2 - Apply QFT and get amplitudes in the fourier space
qft(qcs, n_qubits)
qcs.measure_all()

# setup provider for quantum computating
IBMQ.load_account()
provider = IBMQ.get_provider(hub='ibm-q')
backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= n_qubits
                                       and not x.configuration().simulator
                                       and x.status().operational==True))
#print("least busy backend: ", backend)
#=> least busy backend:  ibmqx2
out = execute(qcs, backend, shots=8192).result()

# Step 3 - Measurement
counts = out.get_counts()
fft = get_fft_from_counts(counts, n_qubits)[:n_samples//2]
#plot_samples(fft[:2000])

plt.xlabel('sample number')
plt.ylabel('value')
plt.plot(list(range(len(fft[:]))), fft[:])
plt.savefig('figure.png', dpi=150, bbox_inches='tight')

top_indices = np.argsort(-np.array(fft))
freqs = top_indices*frame_rate/n_samples
# get top 5 detected frequencies
freqs[:5]
#=> Prints: array([875. , 937.5, 812.5, 750. , 687.5])
