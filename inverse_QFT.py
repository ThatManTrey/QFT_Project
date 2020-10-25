import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import librosa
# Importing standard Qiskit libraries and configuring account
from qiskit import IBMQ, execute, QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister
import wave # .wav file manip
import inv_funx

# load file
file = wave.open("file_name.wav", 'rb')
rt = file.getframerate() # sample rate of audio
samples = file.getnframes() # number of samples
bd = file.getsampwidth() # bit depth
contents = file.readframes(samples)
file.close()

print(samp)
plt.xlabel("sample number")
plt.ylabel("amplitude")
plt.plot(list(range(len(contents))),contents)

# Construct quantum circuit 
q = QuantumRegister(bd*8, 'q')
c = ClassicalRegister(bd*8, 'c')
circuit = QuantumCircuit(q,c)

qcs = prepare_circuit_from_samples(contents, bd*8)
qft(qcs, bd*8)
qcs.measure_all()

IBMQ.load_account()
backend = IBMQ.get_provider(hub='ibm-q')

out = execute(qcs, backend, shots = 10000).result()
counts = out.get_counts()
fft = get_fft_from_counts(counts, bd*8)
plot_samples(fft[:2000])

top_indices = np.argsort(-np.array(fft))
freqs = top_indices*rt/samples
freqs[:5]
