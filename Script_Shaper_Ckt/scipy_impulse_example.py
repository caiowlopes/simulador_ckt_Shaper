"""Example use of scipy.signal.impulse.

This script builds a simple underdamped transfer function, computes its
impulse response, and shows the effect of different normalizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Define a second-order underdamped transfer function H(s) = 1 / (s^2 + 0.2 s + 1)
num = [1.0]
den = [1.0, 0.2, 1.0]

system = signal.TransferFunction(num, den)

# Time vector for the impulse response
T = np.linspace(0, 10, 1001)

# Compute the impulse response
T_resp, y = signal.impulse(system, T=T)

# Compute undershoot relative to the positive peak
peak = np.max(y)
valley = np.min(y)
undershoot_pct = 100.0 * valley / peak

# Normalize by the positive peak only
y_norm_peak = y / peak
undershoot_norm_peak_pct = 100.0 * np.min(y_norm_peak)

# Normalize by the absolute maximum value
abs_peak = np.max(np.abs(y))
y_norm_abs = y / abs_peak
undershoot_norm_abs_pct = 100.0 * np.min(y_norm_abs)

print("Impulse response summary:")
print(f"  peak = {peak:.6f}")
print(f"  valley = {valley:.6f}")
print(f"  undershoot (valley/peak) = {undershoot_pct:.2f}%")
print(f"  undershoot after normalization by peak = {undershoot_norm_peak_pct:.2f}%")
print(f"  undershoot after normalization by abs peak = {undershoot_norm_abs_pct:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(T_resp, y, label="Original impulse response")
plt.plot(T_resp, y_norm_peak, label="Normalized by max(y)")
plt.plot(T_resp, y_norm_abs, label="Normalized by max(abs(y))", linestyle="--")
plt.axhline(0, color="black", linewidth=0.8)
plt.title("scipy.signal.impulse Example")
plt.xlabel("Time")
plt.ylabel("Response")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
