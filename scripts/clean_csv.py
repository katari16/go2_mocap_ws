#!/usr/bin/env python3
"""Clean outliers from leash CSV logs.

Usage:
  python3 clean_csv.py raw_log.csv                  # writes raw_log_clean.csv
  python3 clean_csv.py raw_log.csv -o cleaned.csv   # custom output path
  python3 clean_csv.py raw_log.csv --max-jump 0.3   # stricter outlier gate
  python3 clean_csv.py raw_log.csv --ema-alpha 0.2   # smoother EMA
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def clean(input_path, output_path, max_jump, ema_alpha):
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print('Empty CSV, nothing to clean.')
        return

    raw = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in rows])

    # Pass 1: outlier rejection — flag rows where any component jumps too far
    keep = np.ones(len(raw), dtype=bool)
    for i in range(1, len(raw)):
        if np.linalg.norm(raw[i] - raw[i - 1]) > max_jump:
            keep[i] = False

    n_rejected = int(np.sum(~keep))

    # Pass 2: EMA on kept samples, interpolate rejected ones
    filtered = np.copy(raw)
    ema = raw[0].copy()
    for i in range(len(raw)):
        if keep[i]:
            ema = ema_alpha * raw[i] + (1.0 - ema_alpha) * ema
        filtered[i] = ema

    mag = np.linalg.norm(filtered, axis=1)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'x', 'y', 'z', 'magnitude'])
        for i, r in enumerate(rows):
            writer.writerow([
                r['timestamp_sec'], r['timestamp_nanosec'],
                f'{filtered[i, 0]:.6f}', f'{filtered[i, 1]:.6f}', f'{filtered[i, 2]:.6f}',
                f'{mag[i]:.6f}',
            ])

    print(f'Input:    {len(rows)} samples')
    print(f'Rejected: {n_rejected} outliers ({100 * n_rejected / len(rows):.1f}%)')
    print(f'Output:   {output_path}')

    # Plot raw vs cleaned
    t_sec = np.array([float(r['timestamp_sec']) + float(r['timestamp_nanosec']) * 1e-9
                       for r in rows])
    t_sec -= t_sec[0]

    raw_mag = np.linalg.norm(raw, axis=1)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    labels = ['X', 'Y', 'Z']
    for i, (ax, label) in enumerate(zip(axes[:3], labels)):
        ax.plot(t_sec, raw[:, i], color='#cccccc', linewidth=0.5, label='raw')
        ax.scatter(t_sec[~keep], raw[~keep, i], color='red', s=8, zorder=3, label='rejected')
        ax.plot(t_sec, filtered[:, i], color='#1f77b4', linewidth=1.0, label='cleaned')
        ax.set_ylabel(f'{label} (m)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[3].plot(t_sec, raw_mag, color='#cccccc', linewidth=0.5, label='raw')
    axes[3].scatter(t_sec[~keep], raw_mag[~keep], color='red', s=8, zorder=3, label='rejected')
    axes[3].plot(t_sec, mag, color='#1f77b4', linewidth=1.0, label='cleaned')
    axes[3].set_ylabel('Magnitude (m)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'Leash Vector — {n_rejected} outliers rejected out of {len(rows)} samples', fontsize=12)
    fig.tight_layout()

    plot_path = os.path.splitext(output_path)[0] + '.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f'Plot:     {plot_path}')


def main():
    parser = argparse.ArgumentParser(description='Clean outliers from leash CSV logs')
    parser.add_argument('input', help='Raw CSV file from csv_logger')
    parser.add_argument('-o', '--output', help='Output CSV path (default: <input>_clean.csv)')
    parser.add_argument('--max-jump', type=float, default=0.5,
                        help='Max allowed jump between consecutive samples in meters (default: 0.5)')
    parser.add_argument('--ema-alpha', type=float, default=0.3,
                        help='EMA smoothing factor, 0-1. Lower = smoother (default: 0.3)')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f'{base}_clean{ext}'

    clean(args.input, args.output, args.max_jump, args.ema_alpha)


if __name__ == '__main__':
    main()
