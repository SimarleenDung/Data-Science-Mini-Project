import pandas as pd
import numpy as np
from scipy import signal, interpolate
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

df = pd.read_excel('question4_data.xlsx', sheet_name='Sheet1')
time_original = df['Age (Ma)'].values
features = ['Max (Diameter) (µm)', 'Elongation', 'Min (Diameter) (µm)', 'Shape Factor']

# Calculate time intervals between adjacent points
time_diff = np.diff(time_original)
print(f"Time interval statistics: mean={time_diff.mean():.2f} Ma, std={time_diff.std():.2f} Ma")

# Check if the time sampling is approximately uniform
if time_diff.std() > 1e-3:
    print("Warning: Time series is not uniformly sampled. Interpolation will be performed!")
    # Generate a uniform time array with the same number of points
    uniform_time = np.linspace(time_original.min(), time_original.max(), len(time_original))
else:
    # Time is already uniform
    uniform_time = time_original

#Part 1: FFT to Find Dominant Periods
plt.figure(figsize=(15, 12))

# Loop through each feature to analyze
for i, feature in enumerate(features, 1):
    y_original = df[feature].values  # Extract original data

    # Detrend: remove linear trend
    y_detrend = signal.detrend(y_original)
    # Normalize: zero mean, unit variance
    y_normalized = (y_detrend - y_detrend.mean()) / y_detrend.std()

    # Interpolate if time is non-uniform
    if time_diff.std() > 1e-3:
        f_interp = interpolate.interp1d(time_original, y_normalized, kind='linear', fill_value="extrapolate")
        y_uniform = f_interp(uniform_time)
    else:
        y_uniform = y_normalized

    # Perform FFT
    n = len(y_uniform)
    dt = np.mean(np.diff(uniform_time))  # Time step
    fft_values = np.fft.fft(y_uniform)   # Fast Fourier Transform
    freqs = np.fft.fftfreq(n, d=dt)      # Corresponding frequencies
    power = np.abs(fft_values) ** 2 / n  # Power spectrum

    # Keep only positive frequencies
    mask = freqs > 0
    freqs_pos = freqs[mask]
    power_pos = power[mask]
    periods = 1 / freqs_pos  # Period = 1 / frequency

    # Keep periods shorter than the total time span
    mask_period = periods < (uniform_time.max() - uniform_time.min())
    # Find the dominant period (highest power)
    dominant_idx = np.argmax(power_pos[mask_period])
    dominant_period = periods[mask_period][dominant_idx]
    dominant_power = power_pos[mask_period][dominant_idx]

    # Plot the power spectrum
    plt.subplot(2, 2, i)
    plt.semilogx(periods[mask_period], power_pos[mask_period], color='navy', label='Power Spectrum')
    plt.axvline(dominant_period, color='red', linestyle='--', alpha=0.7, label=f'Dominant Period: {dominant_period:.2f} Ma')
    plt.title(f'{feature}\nDominant Period (FFT): {dominant_period:.2f} Ma')
    plt.xlabel('Period (Ma)')
    plt.ylabel('Power')
    plt.grid(True, alpha=0.3)
    plt.legend()

# Adjust subplot layout
plt.tight_layout()
plt.savefig('power_spectrum.png', dpi=300)
plt.show()

# Reconstruct Signal Using Dominant Frequencies
plt.figure(figsize=(15, 12))

for i, feature in enumerate(features, 1):
    y_original = df[feature].values  # Extract original data

    # Interpolate if time is non-uniform
    if time_diff.std() > 1e-3:
        f_interp = interpolate.interp1d(time_original, y_original, kind='cubic', fill_value="extrapolate")
        y_uniform = f_interp(uniform_time)
    else:
        y_uniform = y_original

    # Remove linear trend
    time_centered = uniform_time - uniform_time.mean()
    coeffs = np.polyfit(time_centered, y_uniform, deg=1)  # Fit a line
    y_trend = np.polyval(coeffs, time_centered)           # Evaluate trend
    y_detrend = y_uniform - y_trend                       # Detrended data

    # Perform FFT
    N = len(y_detrend)
    dt = np.mean(np.diff(uniform_time))
    fft_values = np.fft.fft(y_detrend)
    freqs = np.fft.fftfreq(N, d=dt)
    power = np.abs(fft_values) ** 2 / N

    # Select Top Significant Frequencies
    idx_nonzero = np.where(np.arange(N) != 0)[0]  # Exclude DC component
    idx_sorted = np.argsort(power[idx_nonzero])   # Sort by power
    top_k = 3                                     # Choose top 3 frequencies
    top_indices = idx_nonzero[idx_sorted[-top_k:]]

    # Build New Spectrum with Only Selected Frequencies
    fft_recon = np.zeros_like(fft_values, dtype=complex)

    for idx in top_indices:
        fft_recon[idx] = fft_values[idx]  # Keep positive frequency
        neg_idx = N - idx                 # Keep conjugate symmetric negative frequency
        if neg_idx != idx:
            fft_recon[neg_idx] = fft_values[neg_idx]

    #Inverse FFT to Reconstruct the Signal and Add Back Trend
    y_recon_detrend = np.fft.ifft(fft_recon)  # Inverse FFT
    y_recon_detrend = y_recon_detrend.real    # Take real part
    y_reconstructed = y_recon_detrend + y_trend  # Add back trend

    # Visualize Reconstruction and Calculate R²
    r2_val = r2_score(y_uniform, y_reconstructed)

    plt.subplot(2, 2, i)
    plt.scatter(uniform_time, y_uniform, color='grey', s=15, alpha=0.6, label='Interpolated/Original')
    plt.plot(uniform_time, y_reconstructed, color='darkred', linewidth=2, label=f'Reconstructed (Top {top_k} freqs)')
    plt.title(f'{feature}\nReconstructed vs Original (R² = {r2_val:.2f})')
    plt.xlabel('Age (Ma)')
    plt.ylabel('Value')
    plt.legend()

plt.tight_layout()
plt.savefig('signal_reconstruction.png', dpi=300)
plt.show()
