import numpy as np
pi = np.pi
import matplotlib.pyplot as plt
import librosa
# Importing standard Qiskit libraries and configuring account
from qiskit import QuantumCircuit, execute, Aer, IBMQ, ClassicalRegister, QuantumRegister
from qiskit.compiler import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from qiskit.quantum_info import *
from qiskit.providers.aer import StatevectorSimulator
import soundfile as sf
import scipy.io.wavfile
import glob
import wave
import os
import csv
from GenerateWave import CreateWaveSound

# Loading your IBM Q account(s)
provider = IBMQ.enable_account('423d6b5737a7dc50948e1dfba8ded2b9b56317d5fceb7881cbe64d671e5c5c961385527862ee8e2d856943eb759fe5575b29f82847f9c5232911272a517ced2b')

for file in glob.glob("*.wav"):
    if os.path.exists(file):
        os.remove(file)

sounds = []

csv = open('frequencies.csv', 'r').read()
frequency = csv.split(',')

for i in range(len(frequency)):
    noise = CreateWaveSound(provider, int(frequency[i]), 1)
    # write to wave
    sf.write(str('quantum_sound' + str(i) + '.wav'), noise, 44100, 'PCM_24')
