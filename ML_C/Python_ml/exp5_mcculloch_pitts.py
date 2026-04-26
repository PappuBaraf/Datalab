# ML_Practical/exp5_mcculloch_pitts.py
# Experiment 5: To implementation of Three Input AND Gate using McCulloch-Pitts Artificial Neuron Model in Python with suitable threshold

import numpy as np

def mcculloch_pitts_3input_and(x1, x2, x3):
    # Weights and suitable threshold for 3-input AND gate
    weights = np.array([1, 1, 1])
    threshold = 3 
    
    inputs = np.array([x1, x2, x3])
    net_input = np.dot(inputs, weights)
    
    # Activation function
    return 1 if net_input >= threshold else 0

print("--- McCulloch-Pitts Model (3-Input AND Gate) ---")
print("x1 x2 x3 | Output")
print("-" * 17)
for x1 in [0, 1]:
    for x2 in [0, 1]:
        for x3 in [0, 1]:
            out = mcculloch_pitts_3input_and(x1, x2, x3)
            print(f" {x1}  {x2}  {x3} |   {out}")