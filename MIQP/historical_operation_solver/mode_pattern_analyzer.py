#%%
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

def plot_identical_patterns(csv_file_path, matching_patterns):
    """
    Create line plots for dates with identical power patterns, with background colors indicating modes
    
    Args:
        csv_file_path: Path to the training_data_piecewise.csv file
        matching_patterns: Dictionary of patterns and their corresponding dates
    """
    
    if not matching_patterns:
        print("No matching patterns to plot")
        return
    
    df = pd.read_csv(csv_file_path)
    
    # Define mode function
    def get_mode(power):
        if power > 0.5:
            return 'turbine'
        elif power < -0.5:
            return 'pump'
        else:
            return 'idle'
    
    df['mode'] = df['power'].apply(get_mode)
    
    # Define colors for different modes
    mode_colors = {
        'pump': '#FFE5E5',      # Light red background
        'idle': '#FFFFCC',      # Light yellow background  
        'turbine': '#E5F5FF'    # Light blue background
    }
    
    mode_line_colors = {
        'pump': '#FF4444',      # Red line
        'idle': '#FFA500',      # Orange line
        'turbine': '#4444FF'    # Blue line
    }
    
    # Create separate plots for each matching pattern
    num_patterns = len(matching_patterns)
    
    for idx, (pattern, dates) in enumerate(matching_patterns.items()):
        # Create a new figure for each pattern
        fig, ax = plt.subplots(1, 1, figsize=(15, 6))
        
        # Plot each date with this pattern
        for i, date in enumerate(dates):
            date_data = df[df['date'] == date].sort_values('hour')
            
            if len(date_data) != 24:
                continue
                
            hours = date_data['hour'].values
            powers = date_data['power'].values
            modes = date_data['mode'].values
            
            # Plot the power line
            line_style = '-' if i == 0 else '--'
            alpha = 1.0 if i == 0 else 0.7
            ax.plot(hours, powers, line_style, linewidth=2.5, alpha=alpha, 
                   label=f'{date}', markersize=4, marker='o')
        
        # Add background colors for modes
        current_mode = None
        start_hour = 0
        
        sample_date = dates[0]
        sample_data = df[df['date'] == sample_date].sort_values('hour')
        
        for hour in range(25):  # Go to 25 to handle the last segment
            if hour < 24:
                mode = sample_data[sample_data['hour'] == hour].iloc[0]['mode']
            else:
                mode = None  # End of data
            
            # If mode changes or we're at the end, fill the previous segment
            if mode != current_mode or hour == 24:
                if current_mode is not None:
                    ax.axvspan(start_hour, hour, alpha=0.3, 
                              color=mode_colors[current_mode], 
                              label=f'{current_mode.title()} Mode' if idx == 0 and start_hour == 0 else "")
                current_mode = mode
                start_hour = hour
        
        # Formatting
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Power (MW)', fontsize=12, fontweight='bold')
        ax.set_title(f'Pattern {idx+1}: Identical Power Schedules\n{len(dates)} dates with same mode pattern', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Add horizontal line at y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6, label='Turbine threshold')
        ax.axhline(y=-0.5, color='gray', linestyle=':', linewidth=1, alpha=0.6, label='Pump threshold')
        
        # Legend
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Add power range info
        sample_powers = sample_data['power'].values
        power_range = f"Power range: {sample_powers.min():.1f} to {sample_powers.max():.1f} MW"
        ax.text(0.02, 0.98, power_range, transform=ax.transAxes, 
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
               fontsize=10, verticalalignment='top')
        
        # Show the individual plot
        plt.tight_layout()
        plt.show()
    
    # Create a summary plot showing mode distribution
    create_mode_summary_plot(csv_file_path, matching_patterns)


def create_mode_summary_plot(csv_file_path, matching_patterns):
    """
    Create a summary visualization showing the mode patterns
    """
    df = pd.read_csv(csv_file_path)
    
    def get_mode(power):
        if power > 0.5:
            return 'turbine'
        elif power < -0.5:
            return 'pump'
        else:
            return 'idle'
    
    df['mode'] = df['power'].apply(get_mode)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mode pattern heatmap for identical patterns
    pattern_data = []
    pattern_labels = []
    
    for idx, (pattern, dates) in enumerate(matching_patterns.items()):
        # Convert modes to numbers for heatmap
        mode_numbers = []
        for mode in pattern:
            if mode == 'pump':
                mode_numbers.append(-1)
            elif mode == 'idle':
                mode_numbers.append(0)
            else:  # turbine
                mode_numbers.append(1)
        
        pattern_data.append(mode_numbers)
        pattern_labels.append(f'Pattern {idx+1}\n({len(dates)} dates)')
    
    # Create heatmap
    heatmap_data = np.array(pattern_data)
    im = ax1.imshow(heatmap_data, aspect='auto', cmap='RdYlBu', vmin=-1, vmax=1)
    
    ax1.set_xticks(range(24))
    ax1.set_xticklabels(range(24))
    ax1.set_yticks(range(len(pattern_labels)))
    ax1.set_yticklabels(pattern_labels)
    ax1.set_xlabel('Hour of Day', fontweight='bold')
    ax1.set_title('Mode Patterns Heatmap\n(Red=Pump, Yellow=Idle, Blue=Turbine)', fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.6)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_ticklabels(['Pump', 'Idle', 'Turbine'])
    
    # Plot 2: Overall mode distribution pie chart
    mode_counts = df['mode'].value_counts()
    colors = ['#FF6B6B', '#FFE66D', '#4ECDC4']  # Red, Yellow, Blue
    
    wedges, texts, autotexts = ax2.pie(mode_counts.values, labels=mode_counts.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    
    ax2.set_title('Overall Mode Distribution\n(All Dates)', fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('white')
    
    plt.tight_layout()
    plt.show()


def analyze_mode_patterns(csv_file_path):
    """
    Analyze training data to find dates with identical 24-hour mode patterns
    
    Args:
        csv_file_path: Path to the training_data_piecewise.csv file
    
    Returns:
        Dictionary with matching patterns and the dates that have them
    """
    
    print("Loading training data...")
    df = pd.read_csv(csv_file_path)
    
    print(f"Loaded {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Unique dates: {df['date'].nunique()}")
    
    # Define mode based on power
    def get_mode(power):
        if power > 0.5:
            return 'turbine'
        elif power < -0.5:
            return 'pump'
        else:
            return 'idle'
    
    # Add mode column
    df['mode'] = df['power'].apply(get_mode)
    
    # Group by date and create 24-hour mode patterns
    print("\nCreating 24-hour mode patterns for each date...")
    
    mode_patterns = {}
    incomplete_dates = []
    
    for date, group in df.groupby('date'):
        # Sort by hour to ensure correct order
        group = group.sort_values('hour')
        
        # Check if we have complete 24-hour data
        if len(group) != 24:
            incomplete_dates.append(date)
            continue
            
        # Check if hours are 0-23
        if not all(hour in group['hour'].values for hour in range(24)):
            incomplete_dates.append(date)
            continue
            
        # Create mode pattern tuple (for hashing)
        mode_pattern = tuple(group['mode'].tolist())
        mode_patterns[date] = mode_pattern
    
    if incomplete_dates:
        print(f"Warning: {len(incomplete_dates)} dates have incomplete 24-hour data")
        print(f"First few incomplete dates: {incomplete_dates[:5]}")
    
    print(f"Analyzing {len(mode_patterns)} complete dates...")
    
    # Group dates by their mode patterns
    pattern_to_dates = defaultdict(list)
    for date, pattern in mode_patterns.items():
        pattern_to_dates[pattern].append(date)
    
    # Find patterns that appear in multiple dates
    matching_patterns = {pattern: dates for pattern, dates in pattern_to_dates.items() if len(dates) > 1}
    
    print(f"\nFound {len(matching_patterns)} patterns that appear in multiple dates")
    
    # Display results
    if matching_patterns:
        print("\n" + "="*80)
        print("DATES WITH IDENTICAL 24-HOUR MODE PATTERNS")
        print("="*80)
        
        for i, (pattern, dates) in enumerate(matching_patterns.items(), 1):
            print(f"\nPattern {i}: {len(dates)} dates with identical modes")
            print(f"Dates: {', '.join(dates)}")
            
            # Show the 24-hour pattern
            print("24-hour mode pattern:")
            for hour, mode in enumerate(pattern):
                print(f"  Hour {hour:2d}: {mode}")
            
            # Show power values for first date with this pattern
            sample_date = dates[0]
            sample_data = df[df['date'] == sample_date].sort_values('hour')
            print(f"\nPower values for {sample_date} (sample):")
            for _, row in sample_data.iterrows():
                print(f"  Hour {row['hour']:2d}: {row['power']:8.3f} MW ({row['mode']})")
            
            print("-" * 80)
        
        # Create visualizations for identical patterns
        print("\nCreating visualizations...")
        plot_identical_patterns(csv_file_path, matching_patterns)
        
        # Return first two matching dates
        first_pattern_dates = list(matching_patterns.values())[0]
        return first_pattern_dates[:2] if len(first_pattern_dates) >= 2 else None
        
    else:
        print("\nNo dates found with identical 24-hour mode patterns")
        
        # Show some statistics
        print("\nMode distribution summary:")
        mode_counts = df['mode'].value_counts()
        for mode, count in mode_counts.items():
            print(f"  {mode}: {count} hours ({count/len(df)*100:.1f}%)")
        
        # Show a few example patterns
        print("\nExample patterns (first 5 dates):")
        for i, (date, pattern) in enumerate(list(mode_patterns.items())[:5]):
            pattern_str = ''.join([m[0].upper() for m in pattern])  # T, P, I for short
            print(f"  {date}: {pattern_str}")
        
        return None

def compare_specific_dates(csv_file_path, date1, date2):
    """
    Compare mode patterns between two specific dates
    
    Args:
        csv_file_path: Path to the training_data_piecewise.csv file
        date1, date2: Date strings to compare
    """
    df = pd.read_csv(csv_file_path)
    
    # Define mode function
    def get_mode(power):
        if power > 0.5:
            return 'turbine'
        elif power < -0.5:
            return 'pump'
        else:
            return 'idle'
    
    df['mode'] = df['power'].apply(get_mode)
    
    # Get data for both dates
    data1 = df[df['date'] == date1].sort_values('hour')
    data2 = df[df['date'] == date2].sort_values('hour')
    
    if len(data1) == 0:
        print(f"Date {date1} not found in dataset")
        return
    if len(data2) == 0:
        print(f"Date {date2} not found in dataset")
        return
    
    print(f"\nComparing {date1} vs {date2}:")
    print("Hour | Date1 Power | Mode1 | Date2 Power | Mode2 | Match")
    print("-" * 65)
    
    matches = 0
    for hour in range(24):
        row1 = data1[data1['hour'] == hour]
        row2 = data2[data2['hour'] == hour]
        
        if len(row1) > 0 and len(row2) > 0:
            power1 = row1.iloc[0]['power']
            mode1 = row1.iloc[0]['mode']
            power2 = row2.iloc[0]['power']
            mode2 = row2.iloc[0]['mode']
            match = "✓" if mode1 == mode2 else "✗"
            if mode1 == mode2:
                matches += 1
            
            print(f"{hour:4d} | {power1:10.3f} | {mode1:7s} | {power2:10.3f} | {mode2:7s} | {match}")
        else:
            print(f"{hour:4d} | Missing data")
    
    print(f"\nMode matches: {matches}/24 hours ({matches/24*100:.1f}%)")
    
    if matches == 24:
        print("🎉 IDENTICAL MODE PATTERNS!")
    else:
        print("Different mode patterns")

# Example usage:
if __name__ == "__main__":
    # Replace with your actual file path
    csv_file = "extended_training_data/training_data_piecewise.csv"
    
    # Set matplotlib style for better plots
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10
    
    # Find dates with identical patterns
    print("Searching for dates with identical 24-hour mode patterns...")
    matching_dates = analyze_mode_patterns(csv_file)
    
    if matching_dates:
        print(f"\n🎯 Found matching dates: {matching_dates[0]} and {matching_dates[1]}")
        
        # Detailed comparison
        compare_specific_dates(csv_file, matching_dates[0], matching_dates[1])
    else:
        print("\nNo identical patterns found. You may want to check specific dates manually.")
        
        # Example of manual comparison (replace with actual dates from your data)
        # compare_specific_dates(csv_file, "2018-12-03", "2019-01-15")