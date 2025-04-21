import pandas as pd
from dateutil.parser import parse
import os
import numpy as np
from datetime import datetime, timedelta

def convert_date_format(input_file, output_file=None, imputation_method='forward_fill'):
    """
    Convert dates in the 'Date Posted' column to YYYY-MM-DD format and handle missing values.
    
    Parameters:
    input_file (str): Path to the input CSV file
    output_file (str, optional): Path to the output CSV file. If None, 
                                will add '_formatted' to the input file name.
    imputation_method (str): Method to handle missing dates:
                            'forward_fill' - Use the previous valid date
                            'backward_fill' - Use the next valid date
                            'mean' - Use the mean date (midpoint between surrounding dates)
                            'median' - Use median date from the column
                            'most_frequent' - Use the most common date
                            'fixed' - Use a fixed date (current date)
    
    Returns:
    str: Path to the output file
    """
    # Generate output filename if not provided
    if output_file is None:
        file_name, file_ext = os.path.splitext(input_file)
        output_file = f"{file_name}_formatted{file_ext}"
    
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check if 'Date Posted' column exists
    if 'Date Posted' not in df.columns:
        raise ValueError("The 'Date Posted' column was not found in the CSV file.")
    
    # Create a function to parse dates that returns a datetime object or None
    def parse_date(date_str):
        try:
            if pd.isna(date_str) or date_str == '':
                return None
            return parse(str(date_str))
        except:
            print(f"Warning: Could not parse date '{date_str}'")
            return None
    
    # First pass: Parse all dates and identify missing ones
    parsed_dates = [parse_date(date_str) for date_str in df['Date Posted']]
    df['Parsed_Date'] = parsed_dates
    
    # Identify indices with missing dates
    missing_indices = df.index[df['Parsed_Date'].isna()].tolist()
    
    if missing_indices:
        print(f"Found {len(missing_indices)} missing or invalid dates. Applying imputation method: {imputation_method}")
        
        # Create a series of parsed dates (excluding None)
        valid_dates = [date for date in parsed_dates if date is not None]
        
        if imputation_method == 'forward_fill':
            # For each missing index, find the previous valid date
            for idx in missing_indices:
                prev_valid_idx = idx - 1
                while prev_valid_idx >= 0 and df.loc[prev_valid_idx, 'Parsed_Date'] is None:
                    prev_valid_idx -= 1
                
                if prev_valid_idx >= 0:  # If found a previous valid date
                    df.loc[idx, 'Parsed_Date'] = df.loc[prev_valid_idx, 'Parsed_Date']
                elif valid_dates:  # If no previous date, use the first valid date
                    df.loc[idx, 'Parsed_Date'] = valid_dates[0]
        
        elif imputation_method == 'backward_fill':
            # For each missing index, find the next valid date
            for idx in missing_indices:
                next_valid_idx = idx + 1
                while next_valid_idx < len(df) and df.loc[next_valid_idx, 'Parsed_Date'] is None:
                    next_valid_idx += 1
                
                if next_valid_idx < len(df):  # If found a next valid date
                    df.loc[idx, 'Parsed_Date'] = df.loc[next_valid_idx, 'Parsed_Date']
                elif valid_dates:  # If no next date, use the last valid date
                    df.loc[idx, 'Parsed_Date'] = valid_dates[-1]
        
        elif imputation_method == 'mean':
            if valid_dates:
                # Calculate the mean (average) date
                min_date = min(valid_dates)
                max_date = max(valid_dates)
                mean_days = (max_date - min_date).days // 2
                mean_date = min_date + timedelta(days=mean_days)
                
                # Apply mean date to all missing values
                for idx in missing_indices:
                    df.loc[idx, 'Parsed_Date'] = mean_date
        
        elif imputation_method == 'median':
            if valid_dates:
                # Calculate the median date
                valid_dates.sort()
                median_date = valid_dates[len(valid_dates) // 2]
                
                # Apply median date to all missing values
                for idx in missing_indices:
                    df.loc[idx, 'Parsed_Date'] = median_date
        
        elif imputation_method == 'most_frequent':
            if valid_dates:
                # Find the most frequent date
                from collections import Counter
                date_counts = Counter([d.strftime('%Y-%m-%d') for d in valid_dates])
                most_common_date_str = date_counts.most_common(1)[0][0]
                most_common_date = datetime.strptime(most_common_date_str, '%Y-%m-%d')
                
                # Apply most frequent date to all missing values
                for idx in missing_indices:
                    df.loc[idx, 'Parsed_Date'] = most_common_date
        
        elif imputation_method == 'fixed':
            # Use current date for all missing values
            today = datetime.now()
            for idx in missing_indices:
                df.loc[idx, 'Parsed_Date'] = today
    
    # Final formatting of all dates to YYYY-MM-DD
    def format_parsed_date(parsed_date):
        if parsed_date is None:
            return None
        return parsed_date.strftime('%Y-%m-%d')
    
    # Apply formatting to all dates
    df['Date Posted'] = df['Parsed_Date'].apply(format_parsed_date)
    
    # Remove the temporary column
    df = df.drop('Parsed_Date', axis=1)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert date formats in CSV files with missing value handling.')
    parser.add_argument('input_file', help='Path to the input CSV file')
    parser.add_argument('--output_file', help='Path to the output CSV file (optional)')
    parser.add_argument('--imputation', choices=['forward_fill', 'backward_fill', 'mean', 'median', 'most_frequent', 'fixed'],
                        default='forward_fill', help='Method to handle missing dates (default: forward_fill)')
    
    args = parser.parse_args()
    
    output = convert_date_format(args.input_file, args.output_file, args.imputation)
    print(f"Date conversion complete. Output saved to: {output}")
    