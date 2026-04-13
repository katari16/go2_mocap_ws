#!/usr/bin/env python3
"""Clean outliers from leash CSV logs.

Usage:
  python3 clean_csv.py raw_log.csv                    # writes raw_log_clean.csv
  python3 clean_csv.py raw_log.csv --max-jump 0.1     # stricter outlier gate
  python3 clean_csv.py raw_log.csv --median-window 7  # wider median filter
  python3 clean_csv.py raw_log.csv --ma-window 15     # wider moving average
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def moving_average(data, window):
    if window <= 1:
        return data.copy()
    out = np.copy(data)
    half = window // 2
    for i in range(len(data)):
        lo = max(0, i - half)
        hi = min(len(data), i + half + 1)
        out[i] = np.mean(data[lo:hi], axis=0)
    return out


def clean(input_path, output_path, max_jump, median_window, ma_window):
    with open(input_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print('Empty CSV, nothing to clean.')
        return

    raw = np.array([[float(r['x']), float(r['y']), float(r['z'])] for r in rows])

    # Pass 1: outlier rejection — flag rows where jump is too large
    keep = np.ones(len(raw), dtype=bool)
    for i in range(1, len(raw)):
        if np.linalg.norm(raw[i] - raw[i - 1]) > max_jump:
            keep[i] = False

    n_rejected = int(np.sum(~keep))

    # Interpolate rejected samples from neighbors
    interpolated = np.copy(raw)
    for i in range(len(raw)):
        if not keep[i]:
            prev = i - 1
            nxt = i + 1
            while nxt < len(raw) and not keep[nxt]:
                nxt += 1
            if prev >= 0 and nxt < len(raw):
                interpolated[i] = (raw[prev] + raw[nxt]) / 2.0
            elif prev >= 0:
                interpolated[i] = raw[prev]

    # Pass 2: median filter — kills remaining spikes
    half_med = median_window // 2
    median_filtered = np.copy(interpolated)
    for i in range(len(interpolated)):
        lo = max(0, i - half_med)
        hi = min(len(interpolated), i + half_med + 1)
        median_filtered[i] = np.median(interpolated[lo:hi], axis=0)

    # Pass 3: moving average — smooths the signal
    filtered = moving_average(median_filtered, ma_window)

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
    parser.add_argument('--max-jump', type=float, default=0.15,
                        help='Max allowed jump between consecutive samples in meters (default: 0.15)')
    parser.add_argument('--median-window', type=int, default=5,
                        help='Median filter window size, odd number (default: 5)')
    parser.add_argument('--ma-window', type=int, default=11,
                        help='Moving average window size (default: 11)')
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f'{base}_clean{ext}'

    clean(args.input, args.output, args.max_jump, args.median_window, args.ma_window)


if __name__ == '__main__':
    main()
