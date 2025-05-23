import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import datetime
from io import StringIO

# Configure page and increase file size limit
st.set_page_config(
    page_title="EEG Data Processor",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

def main():
    st.title("EEG Data Processor")
    
    # Welcome message
    st.markdown("""
    ## Welcome to the EEG Data Processor!
    
    This application processes EEG data collected with Muse headbands through Mind Monitor. The app can:
    
    - **Clean and preprocess** a single user's EEG data by removing noise, resampling data to 1-second intervals, and adding session markers
    - **Combine multiple preprocessed datasets** into one comprehensive dataset for group analysis
    
    Choose an option from the tabs below to get started.
    """)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Single User Processing", "Combine Multiple Users"])
    
    # Tab 1: Single User Processing
    with tab1:
        process_single_user()
    
    # Tab 2: Combine Multiple Users
    with tab2:
        combine_multiple_users()

def normalize_timestamp(timestamp_str):
    """
    Normalize timestamp format to ensure days, months, hours, minutes, seconds 
    all have leading zeros when needed (i.e., 2025-5-4 9:10:34 -> 2025-05-04 09:10:34)
    """
    if not timestamp_str or not isinstance(timestamp_str, str):
        return timestamp_str
    
    # Strip any extra whitespace
    timestamp_str = timestamp_str.strip()
    
    # Split into date and time parts
    parts = timestamp_str.split(' ')
    if len(parts) != 2:
        return timestamp_str  # Return as-is if format doesn't match expected pattern
        
    date_part, time_part = parts
    
    # Normalize date (yyyy-mm-dd)
    date_components = date_part.split('-')
    if len(date_components) == 3:
        year, month, day = date_components
        # Ensure month and day are 2 digits
        month = month.zfill(2)
        day = day.zfill(2)
        date_part = f"{year}-{month}-{day}"
    
    # Normalize time (hh:mm:ss)
    time_components = time_part.split(':')
    if len(time_components) == 3:
        hour, minute, second = time_components
        # Ensure hour, minute, second are 2 digits
        hour = hour.zfill(2)
        minute = minute.zfill(2)
        second = second.zfill(2)
        time_part = f"{hour}:{minute}:{second}"
    
    return f"{date_part} {time_part}"

def process_single_user():
    st.header("Single User Data Processing")
    
    # User number input
    user_num = st.number_input("User Number", min_value=1, max_value=99, value=1)

    # Clear All button
    if st.button("Clear All", key="clear_all"):
        st.session_state.clear()
        st.rerun()
    
    # File upload
    st.write("Upload your raw EEG CSV file from Mind Monitor (files can be very large, please be patient)")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv", 
        accept_multiple_files=False,
        help="Upload the raw EEG data CSV file from Mind Monitor"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {"Filename": uploaded_file.name, 
                       "FileType": uploaded_file.type, 
                       "FileSize": f"{uploaded_file.size / (1024 * 1024):.2f} MB"}
        st.write(file_details)
        
        try:
            # Create a temporary file to store the uploaded data - do this only once
            if 'tmp_path' not in st.session_state:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    st.session_state.tmp_path = tmp_file.name
            
            # Initialize timestamp variables if not in session state
            if 'min_timestamp' not in st.session_state:
                st.session_state.min_timestamp = ""
            if 'max_timestamp' not in st.session_state:
                st.session_state.max_timestamp = ""
            
            # Extract timestamp range with a button
            if st.button("Detect Timestamp Range"):
                with st.spinner("Detecting timestamp range..."):
                    try:
                        # Read first and last rows to get timestamp range
                        df_head = pd.read_csv(st.session_state.tmp_path, nrows=1)
                        df_tail = pd.read_csv(st.session_state.tmp_path, skiprows=lambda x: x > 0 and x < os.path.getsize(st.session_state.tmp_path) - 10000)
                        
                        min_timestamp = df_head["TimeStamp"].iloc[0] if "TimeStamp" in df_head.columns else ""
                        max_timestamp = df_tail["TimeStamp"].iloc[-1] if "TimeStamp" in df_tail.columns else ""
                        
                        # Remove milliseconds if present
                        min_timestamp = min_timestamp[:19] if isinstance(min_timestamp, str) else ""
                        max_timestamp = max_timestamp[:19] if isinstance(max_timestamp, str) else ""
                        
                        # Store in session state
                        st.session_state.min_timestamp = min_timestamp
                        st.session_state.max_timestamp = max_timestamp
                        
                        st.success("Timestamp range detected!")
                    except Exception as e:
                        st.warning(f"Couldn't detect timestamp range: {e}")
            
            # Timestamp range inputs
            st.subheader("Timestamp Range")
            if st.session_state.min_timestamp and st.session_state.max_timestamp:
                st.write(f"Detected timestamp range: {st.session_state.min_timestamp} to {st.session_state.max_timestamp}")
            
            # Add normalization to timestamp inputs
            col1, col2 = st.columns(2)
            with col1:
                start_timestamp_input = st.text_input("Start timestamp (YYYY-MM-DD HH:MM:SS)", value=st.session_state.min_timestamp)
                start_timestamp = normalize_timestamp(start_timestamp_input)
                if start_timestamp != start_timestamp_input and start_timestamp_input:
                    st.info(f"Normalized timestamp: {start_timestamp}")
            with col2:
                end_timestamp_input = st.text_input("End timestamp (YYYY-MM-DD HH:MM:SS)", value=st.session_state.max_timestamp)
                end_timestamp = normalize_timestamp(end_timestamp_input)
                if end_timestamp != end_timestamp_input and end_timestamp_input:
                    st.info(f"Normalized timestamp: {end_timestamp}")
            
            # Option to define sections
            define_sections = st.checkbox("Define experiment sections", value=False)
            
            # Section definition with normalized timestamps
            section_data = []
            if define_sections:
                st.subheader("Define Sections")
                st.write("Define the different phases/sections of your experiment with their timestamp ranges:")
                num_sections = st.number_input("Number of sections", min_value=1, max_value=10, value=3)

                # Create inputs for each section using timestamp values
                for i in range(int(num_sections)):
                    st.markdown(f"**Section {i+1}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        section_label = st.text_input(f"Label for Section {i+1}", value=f"{i+1}", key=f"label_{i}")
                    with col2:
                        section_start_input = st.text_input(
                            f"Start time (YYYY-MM-DD HH:MM:SS)",
                            value=start_timestamp_input,
                            key=f"start_{i}"
                        )
                        section_start = normalize_timestamp(section_start_input)
                    with col3:
                        section_end_input = st.text_input(
                            f"End time (YYYY-MM-DD HH:MM:SS)",
                            value=end_timestamp_input,
                            key=f"end_{i}"
                        )
                        section_end = normalize_timestamp(section_end_input)
                    
                    section_data.append({
                        "label": section_label,
                        "start": section_start,
                        "end": section_end
                    })
            
            # Process button
            if st.button("Process Data"):
                with st.spinner('Processing data... Please wait'):
                    progress_bar = st.progress(0)
                    
                    try:
                        # Process the file
                        result_df = process_eeg_data(
                            st.session_state.tmp_path, 
                            user_num,
                            start_timestamp, 
                            end_timestamp, 
                            section_data,
                            include_sections=define_sections,
                            progress_callback=lambda x: progress_bar.progress(x)
                        )
                        
                        if result_df.empty:
                            st.error("No valid data found after processing. Check your timestamp range and filters.")
                            return
                        
                        # Display result
                        st.success('Processing complete!')
                        
                        # Store processed data in session state
                        st.session_state.processed_data = result_df
                        
                        # Display summary statistics
                        st.subheader("Data Summary")
                        st.write(f"Total rows: {len(result_df)}")
                        st.write(f"Time range: {result_df['Time'].iloc[0]} to {result_df['Time'].iloc[-1]}")
                        
                        # Only display section info if sections were defined
                        if define_sections:
                            st.write(f"Sections: {result_df['Section'].nunique()}")
                
                        output_filename = f"user_{user_num}_processed.csv"
                        
                        # Allow user to download the processed file
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="Download Processed Data",
                            data=csv,
                            file_name=output_filename,
                            mime="text/csv"
                        )
                    
                    except MemoryError:
                        st.error("Out of memory. Try processing a smaller file or reduce the time range.")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                        import traceback
                        st.code(traceback.format_exc())
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            import traceback
            st.code(traceback.format_exc())
            
            # Clean up temp file in case of error
            if 'tmp_path' in st.session_state and os.path.exists(st.session_state.tmp_path):
                try:
                    os.unlink(st.session_state.tmp_path)
                    del st.session_state.tmp_path
                except:
                    pass

def combine_multiple_users():
    st.header("Combine Multiple Preprocessed Datasets")
    
    st.write("""
    Upload multiple preprocessed EEG CSV files (created with the Single User Processing tab) 
    to combine them into a single dataset for group analysis.
    """)

    # Clear All button
    if st.button("Clear All", key="clear_all_combined"):
        st.session_state.clear()
        st.rerun()
    
    # Multiple file upload
    uploaded_files = st.file_uploader(
        "Upload preprocessed EEG CSV files", 
        type="csv", 
        accept_multiple_files=True,
        help="Upload the CSV files that were processed with the Single User Processing tab"
    )
    
    if uploaded_files:
        # Display file info button
        if st.button("Show Files Info"):
            st.write(f"Uploaded {len(uploaded_files)} files:")
            for file in uploaded_files:
                file_details = {"Filename": file.name, 
                               "FileSize": f"{file.size / (1024 * 1024):.2f} MB"}
                st.write(file_details)
        
        # Process button
        if st.button("Combine Data"):
            with st.spinner('Combining data...'):
                progress_bar = st.progress(0)
                
                try:
                    # Combine the data
                    all_dfs = []
                    temp_files = []
                    
                    for i, file in enumerate(uploaded_files):
                        # Create a temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                            tmp_file.write(file.getvalue())
                            tmp_path = tmp_file.name
                            temp_files.append(tmp_path)
                        
                        # Read the file directly
                        df = pd.read_csv(tmp_path)
                        all_dfs.append(df)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Concatenate all DataFrames
                    if all_dfs:
                        df_final = pd.concat(all_dfs, ignore_index=True)
                        
                        # Sort by 'User', 'TimeStamp', and 'Time'
                        df_final.sort_values(['User', 'TimeStamp', 'Time'], inplace=True)
                        
                        # Reset the index
                        df_final.reset_index(drop=True, inplace=True)
                        
                        # Remove any -inf or inf values
                        df_final = df_final[~df_final.isin([np.inf, -np.inf]).any(axis=1)]
                        
                        # Drop NaN values
                        df_final = df_final.dropna()
                        
                        # Store in session state
                        st.session_state.combined_data = df_final
                        
                        st.success('Combining complete!')
                        
                        # Display summary info
                        st.subheader("Dataset Summary")
                        users = sorted(df_final['User'].unique())
                        st.write(f"Unique Users in combined dataset: {', '.join(str(u) for u in users)}")
                        st.write(f"Total rows: {len(df_final)}")
                        
                        # Allow user to download the combined file
                        output_filename = "combined_eeg_data.csv"
                        
                        csv = df_final.to_csv(index=False)
                        st.download_button(
                            label="Download Combined Data",
                            data=csv,
                            file_name=output_filename,
                            mime="text/csv"
                        )
                    else:
                        st.error("No valid data found in the uploaded files.")
                    
                    # Clean up temp files
                    for tmp_path in temp_files:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                            
                except Exception as e:
                    st.error(f"An error occurred during combining: {e}")
                    import traceback
                    st.code(traceback.format_exc())

def process_eeg_data(file_path, user_num, start_timestamp, end_timestamp, section_data, include_sections=True, progress_callback=None):
    """
    Process EEG data all at once (faster but requires more memory)
    """
    try:
        # Update progress
        if progress_callback:
            progress_callback(0.1)  # 10% - Starting to read file
        
        # Read the entire file at once
        df = pd.read_csv(file_path, low_memory=False)
        
        # Update progress
        if progress_callback:
            progress_callback(0.3)  # 30% - File loaded
        
        # Filter data
        # Keep only the rows where 'Elements' is NaN
        df = df[df['Elements'].isna()]
        
        # Keep only the first 25 columns if there are more
        if df.shape[1] > 25:
            df = df.iloc[:, :25]
        
        # Remove rows where all brainwave data values are zero
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
        
        # Update progress
        if progress_callback:
            progress_callback(0.4)  # 40% - Initial filtering done
        
        # Format user number
        if user_num < 10:
            df.insert(0, "User", '0' + str(user_num))
        else:
            df.insert(0, "User", str(user_num))

        # Normalize timestamps
        start_timestamp = normalize_timestamp(start_timestamp)
        end_timestamp = normalize_timestamp(end_timestamp)
        
        # Normalize section timestamps
        for section in section_data:
            section['start'] = normalize_timestamp(section['start'])
            section['end'] = normalize_timestamp(section['end'])
                
        # Remove the milliseconds from the 'TimeStamp' column
        df['TimeStamp'] = df['TimeStamp'].str[:-4]
        
        # Filter by timestamp range
        df = df[(df['TimeStamp'] >= start_timestamp) & (df['TimeStamp'] <= end_timestamp)]
        
        # Update progress
        if progress_callback:
            progress_callback(0.6)  # 60% - Time filtering complete
        
        # Add Time column with explicit format
        if not df.empty:
            df.insert(2, "Time", (pd.to_datetime(df['TimeStamp'], format="%Y-%m-%d %H:%M:%S") - 
                                pd.to_datetime(df['TimeStamp'].iloc[0], format="%Y-%m-%d %H:%M:%S") + 
                                pd.to_datetime('0:00:01')).dt.strftime('%H:%M:%S.%f').str[:-3])
        else:
            return pd.DataFrame()  # Return empty DataFrame if filtered data is empty
        
        # Create a copy for resampling
        df_copy = df.copy()
        
        # Update progress
        if progress_callback:
            progress_callback(0.7)  # 70% - Prepared for resampling
        
        # Convert Time to datetime and set as index with explicit format
        df_copy['Time'] = pd.to_datetime(df_copy['Time'], format="%H:%M:%S.%f")
        df_copy.set_index('Time', inplace=True)
        
        # Select numeric columns for resampling
        numeric_df = df_copy.select_dtypes(include=['float64', 'int64'])
        numeric_df = numeric_df.resample('s').mean()
        numeric_df.reset_index(inplace=True)
        
        # Select non-numeric columns for resampling
        non_numeric_df = df_copy.select_dtypes(exclude=['float64', 'int64'])
        non_numeric_df = non_numeric_df.resample('s').first()
        non_numeric_df.reset_index(inplace=True)
        
        # Merge the resampled DataFrames
        df_final = pd.merge(non_numeric_df, numeric_df, on='Time')
        df_final['Time'] = df_final['Time'].dt.time
        
        # Update progress
        if progress_callback:
            progress_callback(0.8)  # 80% - Resampling complete
        
        # Organize columns
        df_final = df_final[['User', 'TimeStamp', 'Time'] + 
                           [col for col in df_final.columns if col not in ['User', 'TimeStamp', 'Time']]]
        
        # Add Section column based on user input (only if sections are included)
        if include_sections and section_data:
            df_final['Section'] = None
            for section in section_data:
                df_final.loc[(df_final['TimeStamp'] >= section['start']) & 
                             (df_final['TimeStamp'] <= section['end']), 'Section'] = section['label']
            
            # Reorder columns to place 'Section' in the 3rd position
            cols = df_final.columns.tolist()
            cols.remove('Section')
            cols.insert(3, 'Section')
            df_final = df_final[cols]
        
        # Drop NaN values
        df_final = df_final.dropna()
        
        # Update progress
        if progress_callback:
            progress_callback(1.0)  # 100% - Processing complete
        
        return df_final
    
    except Exception as e:
        # Clean up in case of error
        if progress_callback:
            progress_callback(1.0)  # Complete the progress bar
        raise e

if __name__ == "__main__":
    main()