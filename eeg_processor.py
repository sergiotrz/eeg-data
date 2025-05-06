import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
from datetime import datetime
import io

# REMOVE THIS SECTION - Not needed for cloud deployment and can cause crashes
# if not os.path.exists('.streamlit'):
#     os.makedirs('.streamlit')
#     with open('.streamlit/config.toml', 'w') as f:
#         f.write('[server]\n')
#         f.write('maxUploadSize = 5000\n')

# Instead, use Streamlit's built-in server settings
# This is done through a separate .streamlit/config.toml file in your repo
# Or through environment variables in your cloud provider

def preprocess_single_user(df, user_num, start_timestamp, end_timestamp, section_timestamps):
    """
    Preprocess data for a single user.
    """
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Display noise information
    status_text.text("Step 1/9: Analyzing noise data")
    if 'Elements' in df.columns:
        if df['Elements'].nunique() > 0:
            st.write(f"Blink count: {df['Elements'].value_counts().iloc[0] if len(df['Elements'].value_counts()) > 0 else 0}")
        if df['Elements'].nunique() > 1:
            st.write(f"Jaw clench count: {df['Elements'].value_counts().iloc[1] if len(df['Elements'].value_counts()) > 1 else 0}")
        if df['Elements'].nunique() > 2:
            st.write(f"Muse Connected count: {df['Elements'].value_counts().iloc[2] if len(df['Elements'].value_counts()) > 2 else 0}")
        if df['Elements'].nunique() > 3:
            st.write(f"Muse Disconnected count: {df['Elements'].value_counts().iloc[3] if len(df['Elements'].value_counts()) > 3 else 0}")
        
        total_noise = df['Elements'].value_counts().sum()
        st.write(f"Total noise: {total_noise}")
    
    # Count rows with all zeros in brainwave data
    zero_rows = df[(df['Delta_TP9'] == 0) & (df['Delta_AF7'] == 0) & 
                   (df['Delta_AF8'] == 0) & (df['Delta_TP10'] == 0) & 
                   (df['Theta_TP9'] == 0) & (df['Theta_AF7'] == 0) & 
                   (df['Theta_AF8'] == 0) & (df['Theta_TP10'] == 0) & 
                   (df['Alpha_TP9'] == 0) & (df['Alpha_AF7'] == 0) & 
                   (df['Alpha_AF8'] == 0) & (df['Alpha_TP10'] == 0) & 
                   (df['Beta_TP9'] == 0) & (df['Beta_AF7'] == 0) & 
                   (df['Beta_AF8'] == 0) & (df['Beta_TP10'] == 0) & 
                   (df['Gamma_TP9'] == 0) & (df['Gamma_AF7'] == 0) & 
                   (df['Gamma_AF8'] == 0) & (df['Gamma_TP10'] == 0)].shape[0]
    st.write(f"Number of rows with all zeros in brainwave data: {zero_rows}")
    progress_bar.progress(10)
    
    # Step 2: Keep only rows where 'Elements' is NaN and first 25 columns
    status_text.text("Step 2/9: Removing noise")
    if 'Elements' in df.columns:
        df = df[df['Elements'].isna()]
    
    # Keep only the first 25 columns
    df = df.iloc[:, :25]
    
    # Verify that there is no NaN values in the DataFrame
    st.write(f"NaNs in cleaned data: {df.isna().sum().sum()}")
    progress_bar.progress(20)
    
    # Step 3: Remove rows with all zeros
    status_text.text("Step 3/9: Removing zero rows")
    df = df[~((df['Delta_TP9'] == 0) & (df['Delta_AF7'] == 0) & 
              (df['Delta_AF8'] == 0) & (df['Delta_TP10'] == 0) & 
              (df['Theta_TP9'] == 0) & (df['Theta_AF7'] == 0) & 
              (df['Theta_AF8'] == 0) & (df['Theta_TP10'] == 0) & 
              (df['Alpha_TP9'] == 0) & (df['Alpha_AF7'] == 0) & 
              (df['Alpha_AF8'] == 0) & (df['Alpha_TP10'] == 0) & 
              (df['Beta_TP9'] == 0) & (df['Beta_AF7'] == 0) & 
              (df['Beta_AF8'] == 0) & (df['Beta_TP10'] == 0) & 
              (df['Gamma_TP9'] == 0) & (df['Gamma_AF7'] == 0) & 
              (df['Gamma_AF8'] == 0) & (df['Gamma_TP10'] == 0))]
    progress_bar.progress(30)
    
    # Step 4: Add user column
    status_text.text("Step 4/9: Adding user information")
    if user_num < 10:
        df.insert(0, "User", '0' + str(user_num))
    else:
        df.insert(0, "User", str(user_num))
    progress_bar.progress(40)
    
    # Step 5: Format timestamps
    status_text.text("Step 5/9: Formatting timestamps")
    if 'TimeStamp' in df.columns:
        df['TimeStamp'] = df['TimeStamp'].str[:-4] if df['TimeStamp'].iloc[0].find('.') != -1 else df['TimeStamp']
    progress_bar.progress(50)
    
    # Step 6: Filter by timestamp range
    status_text.text("Step 6/9: Filtering by timestamp range")
    df = df[(df['TimeStamp'] >= start_timestamp) & (df['TimeStamp'] <= end_timestamp)]
    progress_bar.progress(60)
    
    # Step 7: Add Time column
    status_text.text("Step 7/9: Adding Time column")
    df.insert(2, "Time", (pd.to_datetime(df['TimeStamp']) - 
                         pd.to_datetime(df['TimeStamp'].iloc[0]) + 
                         pd.to_datetime('0:00:01')).dt.strftime('%H:%M:%S.%f').str[:-3])
    progress_bar.progress(70)
    
    # Step 8: Resample to one row per second
    status_text.text("Step 8/9: Resampling data")
    
    # Create a copy
    df_copy = df.copy()
    
    # Convert Time to datetime and set as index
    df_copy['Time'] = pd.to_datetime(df_copy['Time'], format='%H:%M:%S.%f')
    df_copy.set_index('Time', inplace=True)
    
    # Separate numeric and non-numeric columns
    numeric_df = df_copy.select_dtypes(include=['float64', 'int64'])
    numeric_df = numeric_df.resample('s').mean()
    numeric_df.reset_index(inplace=True)
    
    non_numeric_df = df_copy.select_dtypes(exclude=['float64', 'int64'])
    non_numeric_df = non_numeric_df.resample('s').first()
    non_numeric_df.reset_index(inplace=True)
    
    # Merge DataFrames
    df_final = pd.merge(non_numeric_df, numeric_df, on='Time')
    df_final['Time'] = df_final['Time'].dt.time
    
    # Organize columns
    df_final = df_final[['User', 'TimeStamp', 'Time'] + 
                      [col for col in df_final.columns if col not in ['User', 'TimeStamp', 'Time']]]
    progress_bar.progress(80)
    
    # Step 9: Add Section column
    status_text.text("Step 9/9: Adding Section column")
    
    # Add Section column based on timestamps
    for i, (section_start, section_end) in enumerate(section_timestamps):
        df_final.loc[(df_final['TimeStamp'] >= section_start) & 
                   (df_final['TimeStamp'] <= section_end), 'Section'] = i
    
    # Reorder columns
    cols = df_final.columns.tolist()
    if 'Section' in cols:
        cols.remove('Section')
        cols.insert(3, 'Section')
        df_final = df_final[cols]
    
    # Remove NaNs
    df_final = df_final.dropna()
    progress_bar.progress(100)
    
    status_text.text("Processing complete!")
    
    return df_final

def combine_multiple_users(dataframes):
    """
    Combine data from multiple users.
    """
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Step 1/4: Combining dataframes")
    df_final = pd.concat(dataframes, ignore_index=True)
    progress_bar.progress(25)
    
    status_text.text("Step 2/4: Sorting data")
    df_final.sort_values(['User', 'TimeStamp', 'Time'], inplace=True)
    df_final.reset_index(drop=True, inplace=True)
    progress_bar.progress(50)
    
    status_text.text("Step 3/4: Checking for -inf values")
    # Remove -inf values
    inf_count = df_final.isin([-np.inf]).sum().sum()
    st.write(f"Number of -inf values: {inf_count}")
    if inf_count > 0:
        df_final = df_final[~df_final.isin([-np.inf]).any(axis=1)]
    progress_bar.progress(75)
    
    status_text.text("Step 4/4: Checking for NaN values")
    # Verify no NaNs
    nan_count = df_final.isna().sum().sum()
    st.write(f"Number of NaN values: {nan_count}")
    if nan_count > 0:
        df_final = df_final.dropna()
    progress_bar.progress(100)
    
    status_text.text("Processing complete!")
    
    return df_final

def main():
    st.set_page_config(
        page_title="EEG Data Processor",
        page_icon="🧠",
    )
    
    st.title("EEG Data Processor")
    
        # Add author attribution
    st.markdown("""
    *Created by [Sergio Noé Torres-Rodríguez](https://github.com/sergiotrz)*
    
    ---
    """)

    # Initialize session state for storing dataframe
    if 'df' not in st.session_state:
        st.session_state.df = None
        
    # Add a session state variable to track the uploader key
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
        
    st.markdown("""
    ## Welcome to the EEG Data Processor
    
    This application helps you process EEG data from Mind Monitor. You can:
    
    1. **Preprocess a single user's data**: Clean, format and organize data for one participant
    2. **Combine multiple preprocessed datasets**: Merge several already cleaned datasets into one consolidated file
    
    The preprocessing steps include:
    - Removing noise entries (blinks, jaw clenches, etc.)
    - Removing rows with all zeros in brainwave data
    - Formatting timestamps and adding a Time column
    - Resampling data to one row per second
    - Adding section markers based on timestamps
    
    Please select an option below to begin.
    """)
    
    option = st.radio(
        "Choose an operation:",
        ("Preprocess Single User Data", "Combine Multiple Preprocessed Datasets")
    )
    
    if option == "Preprocess Single User Data":
        st.header("Preprocess Single User Data")
        
        # Upload file
        uploaded_file = st.file_uploader("Upload CSV file with raw EEG data", type=["csv"])
        
        # Add Clear All Files button with automatic refresh
        if st.button("Clear All Files"):
            # Clear all session state variables
            for key in list(st.session_state.keys()):
                if key != 'uploader_key':  # Don't delete the uploader key
                    del st.session_state[key]
            
            # Increment the uploader key to force a new uploader widget
            st.session_state.uploader_key += 1
            st.success("Files cleared successfully!")
            
            # Force Streamlit to rerun the script
            st.rerun()
        
        if uploaded_file is not None:
            # Rest of your code remains the same
            # ...
            # Only read the data once and store in session_state
            # Update this line in your main() function, inside the Preprocess Single User Data section
            if st.session_state.df is None:
                with st.spinner("Loading data... (this may take a while for large files)"):
                    try:
                        # Add low_memory=False to handle mixed data types
                        st.session_state.df = pd.read_csv(uploaded_file, low_memory=False)
                    except Exception as e:
                        st.error(f"Error loading file: {str(e)}")
                        st.stop()  # Stop execution if file loading fails
            df = st.session_state.df
            
            # Show sample of raw data
            st.subheader("Sample of Raw Data")
            st.dataframe(df.head())
            
            # Get timestamp range from data for reference
            min_timestamp = df['TimeStamp'].min() if 'TimeStamp' in df.columns else "N/A"
            max_timestamp = df['TimeStamp'].max() if 'TimeStamp' in df.columns else "N/A"
            
            st.write(f"Available timestamp range: {min_timestamp} to {max_timestamp}")
            
            user_num = st.number_input("Enter user number", min_value=1, step=1, value=1)
            
            # Create a form for all timestamp inputs to prevent reruns
            with st.form("timestamp_form"):
                st.subheader("Timestamp Range")
                col1, col2 = st.columns(2)
                with col1:
                    start_timestamp = st.text_input(
                        "Enter start timestamp (YYYY-MM-DD HH:MM:SS)", 
                        value=str(min_timestamp).split('.')[0] if min_timestamp != "N/A" else ""
                    )
                with col2:
                    end_timestamp = st.text_input(
                        "Enter end timestamp (YYYY-MM-DD HH:MM:SS)", 
                        value=str(max_timestamp).split('.')[0] if max_timestamp != "N/A" else ""
                    )
                
                st.subheader("Define Sections")
                st.write("Define the sections in your data based on timestamps.")
                
                num_sections = st.number_input("Number of sections", min_value=1, step=1, value=4)
                
                # Create organized grid for section timestamps
                st.write("Section timestamp ranges:")
                
                section_starts = []
                section_ends = []
                
                # Use columns for better organization
                cols = st.columns(2)
                cols[0].write("**Section Start**")
                cols[1].write("**Section End**")
                
                for i in range(num_sections):
                    col1, col2 = st.columns(2)
                    with col1:
                        section_start = st.text_input(f"Section {i}", key=f"section_{i}_start")
                        section_starts.append(section_start)
                    with col2:
                        section_end = st.text_input(" ", key=f"section_{i}_end")
                        section_ends.append(section_end)
                
                # Submit button for processing
                submit_button = st.form_submit_button("Process Data")
            
            # Only process after form submission
            if submit_button:
                section_timestamps = []
                for i in range(num_sections):
                    if section_starts[i] and section_ends[i]:
                        section_timestamps.append((section_starts[i], section_ends[i]))
                
                if len(section_timestamps) != num_sections:
                    st.error(f"Please define all {num_sections} sections with valid timestamps.")
                else:
                    try:
                        with st.spinner("Processing data..."):
                            processed_df = preprocess_single_user(
                                df, user_num, start_timestamp, end_timestamp, section_timestamps
                            )
                        
                        st.success("Processing complete!")
                        
                        st.subheader("Processed Data")
                        st.dataframe(processed_df)
                        
                        # Statistics
                        st.subheader("Data Statistics")
                        st.dataframe(processed_df.describe())
                        
                        # Check for NaNs
                        nan_count = processed_df.isna().sum().sum()
                        st.write(f"NaN values in processed data: {nan_count}")
                        
                        # Download option
                        output_filename = st.text_input("Enter filename for download", value=f"user_{user_num}_processed.csv")
                        csv = processed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Processed Data as CSV",
                            data=csv,
                            file_name=output_filename,
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
        
    else:  # Combine Multiple Preprocessed Datasets
        st.header("Combine Multiple Preprocessed Datasets")
        
        st.markdown("""
        This option will combine multiple preprocessed datasets into a single file.
        
        Upload all preprocessed CSV files (the ones with "_cleaned.csv" suffix) to combine them into one dataset.
        """)
        
        # Add Clear All Files button with automatic refresh
        if st.button("Clear All Files"):
            # Clear all session state variables
            for key in list(st.session_state.keys()):
                if key != 'uploader_key':  # Don't delete the uploader key
                    del st.session_state[key]
            
            # Increment the uploader key to force a new uploader widget
            st.session_state.uploader_key += 1
            st.success("Files cleared successfully!")
            
            # Force Streamlit to rerun the script
            st.rerun()
        
        # Upload multiple files with dynamic key
        uploaded_files = st.file_uploader(
            "Upload processed CSV files", 
            type=["csv"], 
            accept_multiple_files=True,
            key=f"file_uploader_{st.session_state.uploader_key}"  # Dynamic key changes when cleared
        )

        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} files")
            
            # Show file names
            for file in uploaded_files:
                st.write(f"- {file.name}")
            
            if st.button("Combine Datasets"):
                try:
                    with st.spinner("Combining datasets..."):
                        dataframes = []
                        # Update in the Combine Datasets button handler as well
                        for file in uploaded_files:
                            try:
                                df = pd.read_csv(file, low_memory=False)  # Add low_memory=False
                                dataframes.append(df)
                            except Exception as e:
                                st.error(f"Error loading {file.name}: {str(e)}")
                                continue  # Skip problematic files
                        combined_df = combine_multiple_users(dataframes)
                    
                    st.success("Combination complete!")
                    
                    st.subheader("Combined Data")
                    st.dataframe(combined_df.head(50))
                    
                    # Statistics
                    st.subheader("Data Statistics")
                    st.write(f"Total rows: {combined_df.shape[0]}")
                    # In the "Combine Multiple Preprocessed Datasets" section, update this line:
                    st.write(f"Users included: {', '.join(str(u) for u in sorted(combined_df['User'].unique()))}")
                    
                    # Download option
                    output_filename = st.text_input("Enter filename for download", value="combined_eeg_dataset.csv")
                    csv = combined_df.to_csv(index=False)
                    st.download_button(
                        label="Download Combined Data as CSV",
                        data=csv,
                        file_name=output_filename,
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
