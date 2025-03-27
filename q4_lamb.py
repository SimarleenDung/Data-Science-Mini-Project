import pandas as pd
import numpy as np
from astropy.timeseries import LombScargle
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib.ticker import LogLocator, ScalarFormatter

# Data reading and preprocessing
df = pd.read_excel('question4_data.xlsx', sheet_name='Sheet1')
time = df['Age (Ma)'].values
features = ['Max (Diameter) (µm)', 'Elongation', 'Min (Diameter) (µm)', 'Shape Factor']

# Original plot (unchanged)
plt.figure(figsize=(15, 12))
for i, feature in enumerate(features, 1):
    y = df[feature].values
    y_detrend = signal.detrend(y)
    ls = LombScargle(time, y_detrend)
    freq = ls.autofrequency()
    power = ls.power(freq)
    dominant_period = 1 / freq[np.argmax(power)]
    fap = ls.false_alarm_probability(power.max())

    plt.subplot(2, 2, i)
    plt.semilogx(1 / freq, power, color='steelblue', label='Power spectrum')
    plt.axvline(dominant_period, color='crimson', linestyle='--', alpha=0.7, label='Dominant period')
    plt.axhline(ls.false_alarm_level(0.05), color='grey', linestyle=':', linewidth=1.5, alpha=0.7,
                label='95% significance')
    plt.title(f'{feature}\nDominant Period: {dominant_period:.2f} Ma | FAP = {fap:.3f}')
    plt.xlabel('Period (Ma)')
    plt.ylabel('Normalized Power')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()

# New plot (comparison of significant periods inverse transform and original data)
plt.figure(figsize=(15, 12))
for i, feature in enumerate(features, 1):
    y = df[feature].values
    y_detrend = signal.detrend(y)
    y_trend = y - y_detrend  # Extract trend component

    # Lomb-Scargle analysis
    ls = LombScargle(time, y_detrend)
    freq = ls.autofrequency()
    power = ls.power(freq)
    dominant_freq = freq[np.argmax(power)]
    dominant_period = 1 / dominant_freq
    fap = ls.false_alarm_probability(power.max())

    # Only select significant periods (FAP < 0.05)
    if fap < 0.05:
        # Signal reconstruction
        reconstructed = ls.model(time, dominant_freq) + y_trend  # Reconstructed signal including trend

        # Create subplot
        plt.subplot(2, 2, i)

        # Plot original data and reconstructed curve
        plt.scatter(time, y, color='grey', s=15, alpha=0.6, label='Original Data')
        plt.plot(np.sort(time), reconstructed[np.argsort(time)],
                 color='darkorange', linewidth=2, label='Reconstructed')

        # Calculate fitting metrics
        r_squared = 1 - np.var(y - reconstructed) / np.var(y)
        plt.text(0.05, 0.90, f'R² = {r_squared:.2f}', transform=plt.gca().transAxes,
                 ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        # Set title and labels
        plt.title(f'{feature}\nPeriod = {dominant_period:.2f} Ma', fontsize=12)
        plt.xlabel('Age (Ma)', fontweight='bold')
        plt.ylabel('Normalized Value', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.show()