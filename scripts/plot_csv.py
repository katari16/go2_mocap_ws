#!/usr/bin/env python3
"""Plot leash CSV log with a file picker dialog.

Usage:
  python3 plot_csv.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog


def main():
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title='Select leash CSV log',
        filetypes=[('CSV files', '*.csv'), ('All files', '*.*')],
    )
    if not path:
        print('No file selected.')
        return

    with open(path, 'r') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print('Empty CSV.')
        return

    data = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in rows])
    mag = np.linalg.norm(data, axis=1)

    t = np.array([float(r['timestamp_sec']) + float(r['timestamp_nanosec']) * 1e-9
                   for r in rows])
    t -= t[0]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    for i, (ax, label, color) in enumerate(zip(
        axes[:3],
        ['X', 'Y', 'Z'],
        ['#e41a1c', '#4daf4a', '#377eb8'],
    )):
        ax.plot(t, data[:, i], color=color, linewidth=0.8)
        ax.set_ylabel(f'{label} (m)')
        ax.grid(True, alpha=0.3)

    axes[3].plot(t, mag, color='#984ea3', linewidth=0.8)
    axes[3].set_ylabel('Magnitude (m)')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(path.split('/')[-1], fontsize=12)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
