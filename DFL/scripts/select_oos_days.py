#!/usr/bin/env python3
"""
Select 19 Out-of-Sample (OOS) Days for Approach B Validation.

For each of the 19 representative (k-medoids) days in price_data_2024.csv,
find the closest day in the full Belgium.csv dataset (2024 only, excluding
the 19 representative days) based on Euclidean distance of hourly price vectors.

Outputs:
  Data/price_data_2024_oos.csv  — same format as price_data_2024.csv
"""

import os
import sys
import numpy as np
import pandas as pd

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(repo_root)


def load_representative_profiles(path="Data/price_data_2024.csv"):
    """Load the 19 representative daily price profiles."""
    df = pd.read_csv(path)
    profiles = {}
    for _, row in df.iterrows():
        date = row['date']
        prices = [float(p) for p in row['prices_hourly'].split(',')]
        assert len(prices) == 24, f"Expected 24 prices for {date}, got {len(prices)}"
        profiles[date] = np.array(prices)
    return profiles


def load_belgium_daily_profiles(path="Data/Belgium.csv", year=2024):
    """Load hourly Belgium prices and reshape into daily 24-hour profiles.

    Uses local datetime. Drops days with != 24 hours (DST transitions).
    """
    df = pd.read_csv(path)
    df['datetime_local'] = pd.to_datetime(df['Datetime (Local)'])
    df['date'] = df['datetime_local'].dt.date
    df['hour'] = df['datetime_local'].dt.hour

    # Filter to target year
    df = df[df['datetime_local'].dt.year == year]

    profiles = {}
    for date, group in df.groupby('date'):
        group = group.sort_values('hour')
        # Keep only days with exactly 24 hours
        if len(group) != 24:
            continue
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        profiles[date_str] = group['Price (EUR/MWhe)'].values.astype(float)

    return profiles


def select_closest_oos_days(representative, candidate_pool):
    """For each representative day, find its nearest neighbor in the candidate pool.

    Each candidate can only be assigned once (greedy, ordered by cluster index).
    Returns list of (rep_date, oos_date, distance) tuples.
    """
    used = set()
    matches = []

    for rep_date in sorted(representative.keys()):
        rep_vec = representative[rep_date]
        best_date = None
        best_dist = float('inf')

        for cand_date, cand_vec in candidate_pool.items():
            if cand_date in used:
                continue
            dist = np.linalg.norm(rep_vec - cand_vec)
            if dist < best_dist:
                best_dist = dist
                best_date = cand_date

        if best_date is not None:
            used.add(best_date)
            matches.append((rep_date, best_date, best_dist))
            print(f"  Representative {rep_date} -> OOS {best_date}  (dist={best_dist:.2f})")
        else:
            print(f"  WARNING: No match found for {rep_date}")

    return matches


def write_oos_price_file(matches, candidate_pool, output_path="Data/price_data_2024_oos.csv"):
    """Write out-of-sample price data in the same format as price_data_2024.csv."""
    rows = []
    for i, (rep_date, oos_date, dist) in enumerate(matches):
        prices = candidate_pool[oos_date]
        prices_hourly = ",".join(f"{p:.2f}" for p in prices)
        # Repeat each hourly price 4 times for quarterly
        prices_quarterly = ",".join(f"{p:.2f}" for p in prices for _ in range(4))
        rows.append({
            'date': oos_date,
            'type': f'oos_{i}',
            'cluster_index': i,
            'prices_hourly': prices_hourly,
            'prices_quarterly': prices_quarterly
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, quoting=1)  # QUOTE_ALL for prices
    print(f"\nWrote {len(rows)} OOS profiles to {output_path}")
    return df


def main():
    print("=" * 70)
    print("Selecting 19 Out-of-Sample Days (Approach B)")
    print("=" * 70)

    # 1. Load representative profiles
    rep_profiles = load_representative_profiles()
    rep_dates = set(rep_profiles.keys())
    print(f"\nLoaded {len(rep_profiles)} representative profiles:")
    for d in sorted(rep_dates):
        print(f"  {d}")

    # 2. Load all Belgium 2024 daily profiles
    all_profiles = load_belgium_daily_profiles()
    print(f"\nLoaded {len(all_profiles)} complete daily profiles from Belgium 2024")

    # 3. Exclude the 19 representative days from candidate pool
    candidate_pool = {d: v for d, v in all_profiles.items() if d not in rep_dates}
    print(f"Candidate pool (excluding representative days): {len(candidate_pool)} days")

    # 4. Match each representative to closest OOS day
    print("\nMatching representative -> OOS:")
    matches = select_closest_oos_days(rep_profiles, candidate_pool)

    # 5. Write output
    df = write_oos_price_file(matches, candidate_pool)

    # 6. Print summary
    print("\n" + "=" * 70)
    print("Summary of OOS Selections")
    print("=" * 70)
    print(f"{'Rep Date':<14} {'OOS Date':<14} {'Distance':>10}")
    print("-" * 40)
    for rep_date, oos_date, dist in matches:
        print(f"{rep_date:<14} {oos_date:<14} {dist:>10.2f}")

    distances = [d for _, _, d in matches]
    print(f"\nDistance stats: min={min(distances):.2f}, max={max(distances):.2f}, "
          f"mean={np.mean(distances):.2f}, median={np.median(distances):.2f}")


if __name__ == "__main__":
    main()
