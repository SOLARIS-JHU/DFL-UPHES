"""
Data loading functions for DFL training and validation.

This module provides functions to load and process data from CSV files
for both pretraining and validation workflows.
"""

import os
import torch
import pandas as pd
import numpy as np


def load_data_for_pretraining(file_path, source_name, config, device=None):
    """
    Load and process data for pretraining.

    Args:
        file_path: Path to the data CSV file
        source_name: Name of the data source (for logging)
        config: DFLConfig instance (provides MIQP file path)
        device: PyTorch device (cpu or cuda)

    Returns:
        dict: Dictionary mapping date strings to data dictionaries with keys:
              'power', 'head', 'flow', 'price', 'mode'
    """
    if device is None:
        device = torch.device("cpu")

    try:
        # Force comma separator
        df = pd.read_csv(file_path, sep=',', header=0)
        df.columns = df.columns.str.strip()

        # Check for required columns
        required_columns = ['date', 'hour', 'power', 'head', 'flow']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check if price column exists
        if 'price' not in df.columns:
            # Try to load from original MIQP file using config
            original_miqp_file = config.get_miqp_file_path()
            if os.path.exists(original_miqp_file):
                price_df = pd.read_csv(original_miqp_file)
                price_df.columns = price_df.columns.str.strip()

                try:
                    df['date_normalized'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    price_df['date_normalized'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')

                    df = df.merge(price_df[['date_normalized', 'hour', 'price']],
                                 left_on=['date_normalized', 'hour'],
                                 right_on=['date_normalized', 'hour'],
                                 how='left')
                    df.drop('date_normalized', axis=1, inplace=True)

                except Exception as e:
                    df['price'] = None

                if 'price' not in df.columns or df['price'].isna().all():
                    raise ValueError("Price merge failed. Please provide price data directly in the source file.")
            else:
                raise ValueError(f"Original MIQP file not found. Please provide price data.")

        # Convert date column
        try:
            df['Date'] = pd.to_datetime(df['date'])
        except:
            try:
                df['Date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            except:
                df['Date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

        df['Time'] = df['hour']

        # Rename columns
        df = df.rename(columns={
            'power': 'Power',
            'head': 'Head',
            'flow': 'Flow',
            'price': 'Price'
        })

        # Add Mode column
        conditions = [
            (abs(df['Power']) < 0.01),
            (df['Power'] > 0),
            (df['Power'] < 0)
        ]
        choices = ['Idle', 'Turbine', 'Pump']
        df['Mode'] = np.select(conditions, choices, default='Unknown')

        # Group data by date
        data_by_date = {}
        for date, group in df.groupby('Date'):
            group = group.sort_values('Time')

            if len(group) != 24:
                continue

            date_str = date.strftime('%Y-%m-%d')

            date_data = {
                'power': torch.tensor(group['Power'].values, dtype=torch.float32, device=device),
                'head': torch.tensor(group['Head'].values, dtype=torch.float32, device=device),
                'flow': torch.tensor(group['Flow'].values, dtype=torch.float32, device=device),
                'price': torch.tensor(group['Price'].values, dtype=torch.float32, device=device),
                'mode': group['Mode'].values
            }

            data_by_date[date_str] = date_data

        return data_by_date

    except Exception as e:
        return None


def load_new_price_data(file_path="../Data/price_data_2024.csv", device=None):
    """
    Load new price data for scheduling validation.

    Args:
        file_path: Path to the CSV file with new price data
        device: PyTorch device (cpu or cuda)

    Returns:
        dict: Dictionary with date strings as keys and price tensors as values
    """
    if device is None:
        device = torch.device("cpu")

    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check column names from the first line
        if 'date' not in df.columns or 'cluster_index' not in df.columns or 'prices_hourly' not in df.columns:
            # Try to handle the case where column headers might be different
            if len(df.columns) >= 3:
                # Assume first column is date, third column has hourly prices
                df.columns = ['date', 'cluster_index', 'prices_hourly']
            else:
                raise ValueError(f"Expected columns 'date', 'cluster_index', 'prices_hourly' but got {df.columns}")

        # Dictionary to store price data by date
        price_data = {}

        # Process each row
        for _, row in df.iterrows():
            date_str = row['date']
            prices_str = row['prices_hourly']

            # Parse the prices (attempting different delimiter formats)
            try:
                # First try splitting by comma
                prices = [float(p) for p in prices_str.split(',')]
            except:
                try:
                    # If that fails, try splitting by semicolon
                    prices = [float(p) for p in prices_str.split(';')]
                except:
                    # If that fails too, try to interpret as a list-like string
                    prices_str = prices_str.strip('[]')
                    prices = [float(p) for p in prices_str.split()]

            # Ensure we have 24 hours of data
            if len(prices) != 24:
                print(f"Warning: Date {date_str} has {len(prices)} price values instead of 24")
                # Pad or truncate as needed
                if len(prices) < 24:
                    prices.extend([prices[-1]] * (24 - len(prices)))  # Pad with last value
                else:
                    prices = prices[:24]  # Truncate

            # Convert to tensor
            price_tensor = torch.tensor(prices, dtype=torch.float32, device=device)

            # Add to dictionary
            price_data[date_str] = price_tensor

        print(f"Successfully loaded price data for {len(price_data)} days.")
        return price_data

    except Exception as e:
        print(f"Error loading new price data: {e}")
        return None


def load_data_for_validation(file_path, source_name, config, device=None):
    """
    Load historical data for finding similar price profiles.

    Args:
        file_path: Path to the data file
        source_name: Name of the source (for logging purposes)
        config: DFLConfig instance (provides MIQP file path)
        device: PyTorch device (cpu or cuda)

    Returns:
        dict: Dictionary with data grouped by date
    """
    if device is None:
        device = torch.device("cpu")

    try:
        # Read the file
        df = pd.read_csv(file_path, sep=',', header=0)

        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()

        print(f"Loading validation data from {source_name}: {list(df.columns)}")
        print(f"Data shape: {df.shape}")
        print(f"First few rows:\n{df.head(3)}")

        # Check for required columns
        required_columns = ['date', 'hour', 'power', 'head', 'flow']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check if price column already exists in the file
        if 'price' in df.columns:
            print(f"Using price data from {source_name} file.")
            # Check for missing price values
            if df['price'].isna().any():
                print("Warning: Some price values are missing. Filling with synthetic prices.")
                # Fill missing prices with synthetic data
                missing_mask = df['price'].isna()
                df.loc[missing_mask, 'price'] = 50 + 20 * np.sin(2 * np.pi * df.loc[missing_mask, 'hour'] / 24) + 5 * np.random.randn(missing_mask.sum())
        else:
            print(f"No price column found in {source_name}. Trying to load from original MIQP file...")

            # Load original MIQP data to get price information using config
            original_miqp_file = config.get_miqp_file_path()
            if os.path.exists(original_miqp_file):
                print(f"Loading price data from {original_miqp_file}...")
                price_df = pd.read_csv(original_miqp_file)
                price_df.columns = price_df.columns.str.strip()

                # Convert date formats to match for merging
                try:
                    df['date_normalized'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    price_df['date_normalized'] = pd.to_datetime(price_df['date']).dt.strftime('%Y-%m-%d')

                    # Merge price data with validation data on normalized date and hour
                    df = df.merge(price_df[['date_normalized', 'hour', 'price']],
                                 left_on=['date_normalized', 'hour'],
                                 right_on=['date_normalized', 'hour'],
                                 how='left')
                    df.drop('date_normalized', axis=1, inplace=True)

                except Exception as e:
                    print(f"Date format conversion failed: {e}")
                    df['price'] = None

                if 'price' not in df.columns or df['price'].isna().all():
                    print("Warning: Price merge failed. Using synthetic prices.")
                    df['price'] = 50 + 20 * np.sin(2 * np.pi * df['hour'] / 24) + 5 * np.random.randn(len(df))
            else:
                print(f"Warning: Original MIQP file {original_miqp_file} not found. Using synthetic prices.")
                # Generate synthetic price data
                df['price'] = 50 + 20 * np.sin(2 * np.pi * df['hour'] / 24) + 5 * np.random.randn(len(df))

        # Convert date column - handle different formats
        try:
            df['Date'] = pd.to_datetime(df['date'])
        except:
            # Try different date formats
            try:
                df['Date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
            except:
                df['Date'] = pd.to_datetime(df['date'], infer_datetime_format=True)

        df['Time'] = df['hour']

        # Rename columns to match expected format
        df = df.rename(columns={
            'power': 'Power',
            'head': 'Head',
            'flow': 'Flow',
            'price': 'Price'
        })

        # Add 'Mode' column if not present
        if 'Mode' not in df.columns:
            # Determine mode based on power and flow values
            conditions = [
                (abs(df['Power']) < 0.01),  # Idle mode (power close to zero)
                (df['Power'] > 0),          # Turbine mode (positive power)
                (df['Power'] < 0)           # Pump mode (negative power)
            ]
            choices = ['Idle', 'Turbine', 'Pump']
            df['Mode'] = np.select(conditions, choices, default='Unknown')

        # Verify price data
        print(f"Price data statistics:")
        print(f"  Min: {df['Price'].min():.2f}")
        print(f"  Max: {df['Price'].max():.2f}")
        print(f"  Mean: {df['Price'].mean():.2f}")
        print(f"  Missing values: {df['Price'].isna().sum()}")

        # Group data by date
        data_by_date = {}
        for date, group in df.groupby('Date'):
            # Sort by Time to ensure correct order
            group = group.sort_values('Time')

            # Ensure we have 24 hours of data
            if len(group) != 24:
                print(f"Warning: Date {date.strftime('%Y-%m-%d')} has {len(group)} hours instead of 24. Skipping.")
                continue

            # Convert date to string format
            date_str = date.strftime('%Y-%m-%d')

            # Create dictionary for this date
            date_data = {
                'power': torch.tensor(group['Power'].values, dtype=torch.float32, device=device),
                'head': torch.tensor(group['Head'].values, dtype=torch.float32, device=device),
                'flow': torch.tensor(group['Flow'].values, dtype=torch.float32, device=device),
                'price': torch.tensor(group['Price'].values, dtype=torch.float32, device=device),
                'mode': group['Mode'].values
            }

            data_by_date[date_str] = date_data

        print(f"Successfully loaded {source_name} data for {len(data_by_date)} days.")
        return data_by_date

    except Exception as e:
        print(f"Error loading {source_name} data: {e}")
        import traceback
        traceback.print_exc()
        return None
