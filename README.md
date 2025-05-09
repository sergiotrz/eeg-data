# EEG Data Processor

## Overview

The EEG Data Processor is a Streamlit-based application designed for processing and analyzing EEG data collected with Muse headbands via the Mind Monitor application. This tool simplifies data cleanup, standardization, and aggregation to support EEG data analysis workflows.

## Features

- **Single User Data Processing**
  - Clean and preprocess raw EEG data
  - Remove noise and zero-value readings
  - Resample data to 1-second intervals
  - Add timestamps and session markers
  - Define custom experimental sections/phases

- **Multi-User Data Combination**
  - Merge preprocessed datasets from multiple users
  - Sort combined data by user, timestamp, and time
  - Create standardized datasets for group analysis

## Requirements

- Python 3.7+
- Required packages:
  - streamlit
  - pandas
  - numpy

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/sergiotrz/eeg-data.git
   cd eeg-data
   ```

2. Install required packages:
   ```
   pip install streamlit pandas numpy
   ```

3. Run the application:
   ```
   streamlit run eeg_processor.py
   ```

## Usage

### Single User Processing

1. Enter the user number (1-99)
2. Upload a raw EEG CSV file from Mind Monitor
3. Click "Detect Timestamp Range" to automatically identify the data time range
4. Adjust the start and end timestamps if needed
5. (Optional) Define experiment sections:
   - Check the "Define experiment sections" box
   - Set the number of sections
   - Enter labels and timestamp ranges for each section
6. Click "Process Data" to clean and preprocess the data
7. Review the data summary
8. Download the processed data as a CSV file

### Combining Multiple Users

1. Process individual user data first using the Single User Processing tab
2. Navigate to the "Combine Multiple Users" tab
3. Upload multiple preprocessed CSV files
4. Click "Show Files Info" to verify your uploaded files
5. Click "Combine Data" to merge all datasets
6. Review the combined dataset summary
7. Download the combined data as a CSV file

## Data Processing Steps

The application performs the following processing on your EEG data:

1. Filters out rows containing implementation artifacts
2. Removes rows where all brainwave values are zero
3. Standardizes user identification
4. Trims timestamp data to remove milliseconds
5. Adds a Time column relative to the start time
6. Resamples data to 1-second intervals for consistency
7. Adds section markers based on defined experiment phases (if specified)
8. Removes rows with missing values

## File Format

### Input Format
The application expects CSV files from Mind Monitor containing EEG data with columns for:
- TimeStamp
- Delta, Theta, Alpha, Beta, and Gamma waves for each electrode (TP9, AF7, AF8, TP10)
- Other Muse data columns

### Output Format
The processed files include:
- User identifier
- TimeStamp
- Time (relative to session start)
- Section (if defined)
- Delta, Theta, Alpha, Beta, and Gamma waves for each electrode
- Other relevant columns from the original data

## Tips for Best Results

- For large files, be patient during upload and processing
- Define timestamp ranges carefully to focus only on valid experimental data
- Use the section definition feature to mark different phases of your experiment
- Process all individual files before attempting to combine multiple users
- Use the "Clear All" button if you encounter any issues during processing

## Troubleshooting

- **Out of Memory Error**: Try processing a smaller file or reducing the time range
- **No Valid Data Error**: Check that your timestamp range contains valid readings
- **Processing Errors**: Ensure your CSV file follows the expected Mind Monitor format

## Author

[Sergio Noé Torres-Rodríguez](www.linkedin.com/in/sergiotrz)