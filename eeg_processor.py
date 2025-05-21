import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import datetime
import json
import uuid
from io import StringIO
import re

# Configure page and increase file size limit
st.set_page_config(
    page_title="EEG Data Processor",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Create a session ID if one doesn't exist yet
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize the uploads directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def main():
    st.title("EEG Data Processor")
    
    # Welcome message
    st.markdown("""
    ## Welcome to the EEG Data Processor!
    
    This application processes EEG data collected with Muse headbands through Mind Monitor.
    
    **Note about large files:** For files larger than 200MB, consider processing in smaller time segments.
    """)
    
    # Display session info in sidebar
    with st.sidebar:
        st.info(f"Session ID: {st.session_state.session_id[:8]}...")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Single User Processing", "Combine Multiple Users"])
    
    # Tab 1: Single User Processing
    with tab1:
        process_single_user()
    
    # Tab 2: Combine Multiple Users
    with tab2:
        combine_multiple_users()


def get_user_upload_path(filename=None):
    """Generate a consistent path for user uploads"""
    user_dir = os.path.join(UPLOAD_DIR, st.session_state.session_id)
    os.makedirs(user_dir, exist_ok=True)
    
    if filename:
        return os.path.join(user_dir, filename)
    return user_dir


def process_single_user():
    st.header("Single User Data Processing")
    
    # Add upload options
    upload_option = st.radio(
        "Choose upload method:",
        ["Direct Upload", "URL Reference (for large files)"],
        index=0
    )
    
    user_num = st.number_input("User Number", min_value=1, max_value=99, value=1)

    # Clear All button with improved cleanup
    if st.button("Clear All", key="clear_all"):
        user_dir = get_user_upload_path()
        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                try:
                    os.unlink(os.path.join(user_dir, file))
                except:
                    pass
        # Keep the session ID but clear other state
        session_id = st.session_state.session_id
        st.session_state.clear()
        st.session_state.session_id = session_id
        st.rerun()

    # Handle different upload methods
    if upload_option == "Direct Upload":
        st.write("Upload your raw EEG CSV file from Mind Monitor")
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type="csv", 
            accept_multiple_files=False,
            key="direct_upload",
            help="Upload the raw EEG data CSV file from Mind Monitor"
        )
        
        if uploaded_file is not None:
            # Save uploaded file with consistent name
            file_path = get_user_upload_path("raw_eeg_data.csv")
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
                
            st.success(f"File saved successfully: {os.path.basename(file_path)}")
            file_details = {
                "Filename": uploaded_file.name, 
                "FileSize": f"{uploaded_file.size / (1024 * 1024):.2f} MB"
            }
            st.write(file_details)
            
            # Process the saved file
            process_saved_eeg_file(file_path, user_num)
    else:
        # URL-based processing
        url_input = st.text_input(
            "Enter URL to EEG data file",
            help="Enter a publicly accessible URL to your EEG data file"
        )
        
        if url_input and st.button("Load from URL"):
            with st.spinner("Downloading file from URL..."):
                try:
                    import requests
                    response = requests.get(url_input)
                    if response.status_code == 200:
                        file_path = get_user_upload_path("raw_eeg_data_url.csv")
                        with open(file_path, "wb") as f:
                            f.write(response.content)
                            
                        st.success(f"File downloaded successfully: {os.path.basename(file_path)}")
                        file_size = os.path.getsize(file_path) / (1024 * 1024)
                        st.write(f"File size: {file_size:.2f} MB")
                        
                        # Process the downloaded file
                        process_saved_eeg_file(file_path, user_num)
                    else:
                        st.error(f"Failed to download file: HTTP {response.status_code}")
                except Exception as e:
                    st.error(f"Error downloading file: {str(e)}")


def process_saved_eeg_file(file_path, user_num):
    """Process a saved EEG file"""
    
    try:
        # Detect timestamps
        with st.spinner("Detecting timestamp range..."):
            min_ts, max_ts = detect_min_max_timestamp(file_path)
        
        # Timestamp range inputs with normalization
        st.subheader("Timestamp Range")
        if min_ts and max_ts:
            st.write(f"Detected timestamp range: {min_ts} to {max_ts}")

        col1, col2 = st.columns(2)
        with col1:
            start_timestamp = st.text_input(
                "Start timestamp (YYYY-MM-DD HH:MM:SS)",
                value=min_ts
            )
        with col2:
            end_timestamp = st.text_input(
                "End timestamp (YYYY-MM-DD HH:MM:SS)",
                value=max_ts
            )

        # Add sampling option for large files
        sampling_enabled = st.checkbox("Enable data sampling (recommended for large files)", value=False)
        sample_rate = st.slider("Sample rate (%)", 1, 100, 20, disabled=not sampling_enabled)

        # Normalize timestamps
        start_timestamp = normalize_timestamp(start_timestamp)
        end_timestamp = normalize_timestamp(end_timestamp)

        # Section definition
        define_sections = st.checkbox("Define experiment sections", value=False)
        section_data = []
        if define_sections:
            st.subheader("Define Sections")
            st.write("Define the different phases/sections of your experiment:")
            num_sections = st.number_input("Number of sections", min_value=1, max_value=10, value=3)
            for i in range(int(num_sections)):
                st.markdown(f"**Section {i+1}**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    section_label = st.text_input(f"Label for Section {i+1}", value=f"{i+1}", key=f"label_{i}")
                with col2:
                    section_start = st.text_input(
                        f"Start time (YYYY-MM-DD HH:MM:SS)",
                        value=start_timestamp,
                        key=f"start_{i}"
                    )
                with col3:
                    section_end = st.text_input(
                        f"End time (YYYY-MM-DD HH:MM:SS)",
                        value=end_timestamp,
                        key=f"end_{i}"
                    )
                # Normalize timestamps
                section_data.append({
                    "label": section_label,
                    "start": normalize_timestamp(section_start),
                    "end": normalize_timestamp(section_end)
                })

        # Process button
        if st.button("Process Data"):
            with st.spinner('Processing data... This may take several minutes for large files'):
                progress_bar = st.progress(0)
                try:
                    # Process with incremental file writing to reduce memory usage
                    result_path = process_eeg_data_incremental(
                        file_path, 
                        user_num,
                        start_timestamp, 
                        end_timestamp, 
                        section_data,
                        sampling_enabled=sampling_enabled,
                        sample_rate=sample_rate,
                        include_sections=define_sections,
                        progress_callback=lambda x: progress_bar.progress(x)
                    )
                    
                    # Load result summary without loading entire file
                    summary = get_processed_file_summary(result_path)
                    
                    if summary:
                        st.success('Processing complete!')
                        st.subheader("Data Summary")
                        st.write(f"Total rows: {summary['rows']}")
                        st.write(f"Time range: {summary['start_time']} to {summary['end_time']}")
                        if define_sections:
                            st.write(f"Sections: {summary['sections']}")
                            
                        # Provide download link
                        output_filename = f"user_{user_num}_processed.csv"
                        with open(result_path, "rb") as file:
                            st.download_button(
                                label="Download Processed Data",
                                data=file,
                                file_name=output_filename,
                                mime="text/csv"
                            )
                    else:
                        st.error("No valid data found after processing. Check your timestamp range and filters.")
                except MemoryError:
                    st.error("Out of memory. Try processing a smaller file, reduce the time range, or enable sampling.")
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def normalize_timestamp(ts):
    """Normalize timestamp format"""
    import re
    if not ts:
        return ""
    match = re.match(r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})", ts.strip())
    if match:
        date, h, m, s = match.groups()
        return f"{date} {int(h):02d}:{int(m):02d}:{int(s):02d}"
    return ts


def detect_min_max_timestamp(file_path):
    """Detect timestamp range from file efficiently"""
    min_ts, max_ts = "", ""
    try:
        # Read just the first few rows for min timestamp
        first_chunk = next(pd.read_csv(file_path, chunksize=10))
        if "TimeStamp" in first_chunk.columns:
            min_ts = first_chunk["TimeStamp"].iloc[0]
            
        # For max timestamp, read the file in reverse (faster for large files)
        import subprocess
        try:
            # Try using tail command for efficiency on large files
            tail_output = subprocess.check_output(
                f"tail -n 10 {file_path}", shell=True, text=True
            )
            # Parse the output and extract the last timestamp
            lines = tail_output.strip().split("\n")
            for line in reversed(lines):
                parts = line.split(",")
                if len(parts) > 1:  # Ensure it's a valid CSV line
                    # Assuming TimeStamp is in one of the first few columns
                    for i in range(min(5, len(parts))):
                        if parts[i] and parts[i][0].isdigit():  # Check if it looks like a timestamp
                            max_ts = parts[i]
                            break
                    if max_ts:
                        break
        except:
            # Fallback: read with pandas in chunks and keep last chunk
            for chunk in pd.read_csv(file_path, chunksize=1000):
                pass
            if "TimeStamp" in chunk.columns:
                max_ts = chunk["TimeStamp"].iloc[-1]
        
        # Remove milliseconds if present
        min_ts = min_ts[:19] if isinstance(min_ts, str) else ""
        max_ts = max_ts[:19] if isinstance(max_ts, str) else ""
    except Exception:
        pass
    return min_ts, max_ts


def process_eeg_data_incremental(file_path, user_num, start_timestamp, end_timestamp, 
                                section_data, sampling_enabled=False, sample_rate=100,
                                include_sections=True, progress_callback=None):
    """
    Process EEG data with incremental file writing to minimize memory usage
    """
    # Create a temporary output file
    output_path = get_user_upload_path(f"processed_eeg_{user_num}.csv")
    
    # Calculate sampling interval
    sample_every_n = 100 // sample_rate if sampling_enabled else 1
    
    # Process in smaller chunks with minimal memory footprint
    chunk_size = 50
    total_size = os.path.getsize(file_path)
    processed_size = 0
    row_counter = 0
    
    # Track time range for summary
    first_time = None
    last_time = None
    
    # Extract only needed columns to reduce memory usage
    try:
        header = pd.read_csv(file_path, nrows=0)
        necessary_cols = ['TimeStamp', 'Elements']
        for prefix in ['Delta_', 'Theta_', 'Alpha_', 'Beta_', 'Gamma_']:
            necessary_cols.extend([col for col in header.columns if col.startswith(prefix)])
        usecols = [col for col in necessary_cols if col in header.columns]
    except:
        usecols = None
    
    # First pass: filter and write data incrementally
    try:
        # Open the output file
        with open(output_path, 'w', newline='') as f_out:
            header_written = False
            
            # Process file in small chunks
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, usecols=usecols)):
                # Update progress
                processed_size += chunk.memory_usage(deep=True).sum()
                if progress_callback:
                    progress_callback(min(processed_size / total_size, 0.5))
                
                # Apply sampling if enabled
                if sampling_enabled and sample_every_n > 1:
                    chunk = chunk.iloc[::sample_every_n]
                
                # Initial filtering
                if 'Elements' in chunk.columns:
                    chunk = chunk[chunk['Elements'].isna()]
                
                # Keep only first 25 columns
                if chunk.shape[1] > 25:
                    chunk = chunk.iloc[:, :25]
                
                # Filter timestamps
                if 'TimeStamp' in chunk.columns:
                    chunk['TimeStamp'] = chunk['TimeStamp'].astype(str).str[:-4]
                    chunk = chunk[(chunk['TimeStamp'] >= start_timestamp) & 
                                 (chunk['TimeStamp'] <= end_timestamp)]
                
                # Skip if empty
                if chunk.empty:
                    continue
                
                # Remove zero brainwave rows
                chunk = chunk[~((chunk['Delta_TP9'] == 0) & (chunk['Delta_AF7'] == 0) & 
                          (chunk['Delta_AF8'] == 0) & (chunk['Delta_TP10'] == 0) & 
                          (chunk['Theta_TP9'] == 0) & (chunk['Theta_AF7'] == 0) & 
                          (chunk['Theta_AF8'] == 0) & (chunk['Theta_TP10'] == 0) & 
                          (chunk['Alpha_TP9'] == 0) & (chunk['Alpha_AF7'] == 0) & 
                          (chunk['Alpha_AF8'] == 0) & (chunk['Alpha_TP10'] == 0) & 
                          (chunk['Beta_TP9'] == 0) & (chunk['Beta_AF7'] == 0) & 
                          (chunk['Beta_AF8'] == 0) & (chunk['Beta_TP10'] == 0) & 
                          (chunk['Gamma_TP9'] == 0) & (chunk['Gamma_AF7'] == 0) & 
                          (chunk['Gamma_AF8'] == 0) & (chunk['Gamma_TP10'] == 0))]
                
                # Skip if empty after filtering
                if chunk.empty:
                    continue
                
                # Add User column
                if user_num < 10:
                    chunk.insert(0, "User", '0' + str(user_num))
                else:
                    chunk.insert(0, "User", str(user_num))
                
                # Write to CSV incrementally
                chunk.to_csv(f_out, mode='a', header=not header_written, index=False)
                header_written = True
                row_counter += len(chunk)
        
        # If no data was written, return None
        if row_counter == 0:
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None
        
        # Second pass: Calculate time column and resample
        intermediate_path = get_user_upload_path(f"intermediate_{user_num}.csv")
        df = pd.read_csv(output_path)
        
        # Add Time column
        if not df.empty:
            df.insert(2, "Time", (pd.to_datetime(df['TimeStamp']) - pd.to_datetime(df['TimeStamp'].iloc[0]) + 
                               pd.to_datetime('0:00:01')).dt.strftime('%H:%M:%S.%f').str[:-3])
            if progress_callback:
                progress_callback(0.6)
                
            # Track time range for summary
            first_time = df['Time'].iloc[0]
            last_time = df['Time'].iloc[-1]
            
            # Resample to 1-second intervals
            df_copy = df.copy()
            df_copy['Time'] = pd.to_datetime(df_copy['Time'])
            df_copy.set_index('Time', inplace=True)
            
            # Select numeric columns for resampling
            numeric_df = df_copy.select_dtypes(include=['float64', 'int64'])
            numeric_df = numeric_df.resample('s').mean()
            numeric_df.reset_index(inplace=True)
            
            # Select non-numeric columns for resampling
            non_numeric_df = df_copy.select_dtypes(exclude=['float64', 'int64'])
            non_numeric_df = non_numeric_df.resample('s').first()
            non_numeric_df.reset_index(inplace=True)
            
            if progress_callback:
                progress_callback(0.8)
                
            # Merge and finalize
            df_final = pd.merge(non_numeric_df, numeric_df, on='Time')
            df_final['Time'] = df_final['Time'].dt.time
            
            # Organize columns
            df_final = df_final[['User', 'TimeStamp', 'Time'] + 
                               [col for col in df_final.columns if col not in ['User', 'TimeStamp', 'Time']]]
            
            # Add Section column if needed
            if include_sections and section_data:
                df_final['Section'] = None
                for section in section_data:
                    df_final.loc[(df_final['TimeStamp'] >= section['start']) & 
                                 (df_final['TimeStamp'] <= section['end']), 'Section'] = section['label']
                
                # Reorder columns to place 'Section' in the 4th position
                cols = df_final.columns.tolist()
                cols.remove('Section')
                cols.insert(3, 'Section')
                df_final = df_final[cols]
            
            # Drop NaN values
            df_final = df_final.dropna()
            
            # Save processed data
            df_final.to_csv(output_path, index=False)
            
            # Write summary file
            write_file_summary(output_path, {
                'rows': len(df_final),
                'start_time': str(first_time),
                'end_time': str(last_time),
                'sections': len(set(df_final['Section'])) if include_sections and 'Section' in df_final else 0
            })
            
            if progress_callback:
                progress_callback(1.0)
                
            return output_path
        else:
            if os.path.exists(output_path):
                os.unlink(output_path)
            return None
    except Exception as e:
        # Clean up on error
        if os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except:
                pass
        raise e


def write_file_summary(file_path, summary_data):
    """Write summary metadata for a processed file"""
    summary_path = file_path + ".meta"
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f)


def get_processed_file_summary(file_path):
    """Read summary metadata for a processed file"""
    summary_path = file_path + ".meta"
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def combine_multiple_users():
    """Function to combine multiple preprocessed datasets"""
    st.header("Combine Multiple Preprocessed Datasets")
    
    # Clear All button
    if st.button("Clear All", key="clear_all_combined"):
        user_dir = get_user_upload_path()
        if os.path.exists(user_dir):
            for file in os.listdir(user_dir):
                if file.startswith("combined_"):
                    try:
                        os.unlink(os.path.join(user_dir, file))
                    except:
                        pass
        # Keep session ID but clear other state
        session_id = st.session_state.session_id
        st.session_state.clear()
        st.session_state.session_id = session_id
        st.rerun()
    
    # Multiple file upload with better memory management
    st.write("Upload preprocessed CSV files (created with the Single User Processing tab)")
    uploaded_files = st.file_uploader(
        "Choose CSV files", 
        type="csv", 
        accept_multiple_files=True,
        key="combine_upload",
        help="Upload preprocessed CSV files (smaller files work better)"
    )
    
    if uploaded_files:
        # Display file info
        if st.button("Show Files Info"):
            st.write(f"Uploaded {len(uploaded_files)} files:")
            for file in uploaded_files:
                file_details = {"Filename": file.name, 
                               "FileSize": f"{file.size / (1024 * 1024):.2f} MB"}
                st.write(file_details)
        
        # Combine button
        if st.button("Combine Data"):
            with st.spinner('Combining data...'):
                progress_bar = st.progress(0)
                
                try:
                    # Save each file first
                    temp_files = []
                    for i, file in enumerate(uploaded_files):
                        file_path = get_user_upload_path(f"combine_input_{i}.csv")
                        with open(file_path, "wb") as f:
                            f.write(file.getvalue())
                        temp_files.append(file_path)
                        # Update progress
                        progress_bar.progress((i + 1) / (len(uploaded_files) * 2))
                    
                    # Combine files incrementally to avoid memory issues
                    combined_path = get_user_upload_path("combined_eeg_data.csv")
                    with open(combined_path, 'w', newline='') as f_out:
                        header_written = False
                        
                        for i, file_path in enumerate(temp_files):
                            # Process in chunks for large files
                            chunk_size = 1000
                            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                                # Write to combined file
                                chunk.to_csv(f_out, mode='a', header=not header_written, index=False)
                                header_written = True
                            
                            # Update progress
                            progress_bar.progress(0.5 + (i + 1) / (len(temp_files) * 2))
                            
                            # Clean up intermediate file
                            try:
                                os.unlink(file_path)
                            except:
                                pass
                    
                    # Read back for sorting and final processing
                    df_final = pd.read_csv(combined_path)
                    
                    # Sort by 'User', 'TimeStamp', and 'Time'
                    df_final.sort_values(['User', 'TimeStamp', 'Time'], inplace=True)
                    
                    # Reset index
                    df_final.reset_index(drop=True, inplace=True)
                    
                    # Clean data
                    df_final = df_final[~df_final.isin([np.inf, -np.inf]).any(axis=1)]
                    df_final = df_final.dropna()
                    
                    # Save final sorted version
                    df_final.to_csv(combined_path, index=False)
                    
                    # Display summary info
                    st.success('Combining complete!')
                    st.subheader("Dataset Summary")
                    users = sorted(df_final['User'].unique())
                    st.write(f"Unique Users in combined dataset: {', '.join(str(u) for u in users)}")
                    st.write(f"Total rows: {len(df_final)}")
                    
                    # Provide download button
                    with open(combined_path, "rb") as file:
                        st.download_button(
                            label="Download Combined Data",
                            data=file,
                            file_name="combined_eeg_data.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"An error occurred during combining: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
