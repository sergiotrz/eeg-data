import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import datetime
from io import StringIO
import re

# Configure page and increase file size limit
st.set_page_config(
    page_title="EEG Data Processor",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

def standardize_timestamp(timestamp_str):
    """
    Standardize timestamp format to ensure HH:MM:SS format with leading zeros
    
    Examples:
    2025-05-02 9:32:30 → 2025-05-02 09:32:30
    2025-5-2 9:32:30 → 2025-05-02 09:32:30
    """
    if not timestamp_str or not isinstance(timestamp_str, str):
        return timestamp_str
        
    # Use regex to match timestamp components
    match = re.match(r'(\d{4})-(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})', timestamp_str)
    if match:
        year, month, day, hour, minute, second = match.groups()
        # Pad with leading zeros where needed
        return f"{year}-{int(month):02d}-{int(day):02d} {int(hour):02d}:{int(minute):02d}:{int(second):02d}"
    return timestamp_str


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

@st.cache_resource
def get_chunk_iterator(file_path, chunksize=500):  # Reduced from 5000 to 500
    """Get a chunk iterator for large CSV files"""
    return pd.read_csv(file_path, chunksize=chunksize, low_memory=False)
# ...existing code...

def process_single_user():
    st.header("Single User Data Processing")
    
    # User number input
    user_num = st.number_input("User Number", min_value=1, max_value=99, value=1)

    # Clear All button - improved to clean temporary files
    if st.button("Clear All", key="clear_all"):
        # Clean up temp file if it exists
        if 'tmp_path' in st.session_state and os.path.exists(st.session_state.tmp_path):
            try:
                os.unlink(st.session_state.tmp_path)
            except Exception:
                pass
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
                    
                # Automatically detect timestamp range when file is uploaded
                with st.spinner("Detecting timestamp range (this may take a moment)..."):
                    try:
                        # Read first chunk to get min timestamp
                        first_chunk = next(get_chunk_iterator(st.session_state.tmp_path, chunksize=500))
                        min_timestamp = first_chunk["TimeStamp"].iloc[0] if "TimeStamp" in first_chunk.columns else ""
                        
                        # Read chunks incrementally to find the last one (more efficient)
                        max_timestamp = min_timestamp
                        chunk_count = 0
                        total_chunks = (os.path.getsize(st.session_state.tmp_path) // (500 * 100)) + 1  # Estimate
                        
                        # Create a progress bar for timestamp detection
                        timestamp_progress = st.progress(0)
                        
                        for i, chunk in enumerate(get_chunk_iterator(st.session_state.tmp_path, chunksize=500)):
                            chunk_count += 1
                            if "TimeStamp" in chunk.columns and not chunk["TimeStamp"].empty:
                                chunk_max = chunk["TimeStamp"].iloc[-1]
                                if chunk_max > max_timestamp:
                                    max_timestamp = chunk_max
                            
                            # Update progress every 10 chunks
                            if i % 10 == 0:
                                timestamp_progress.progress(min(i / total_chunks, 1.0))
                        
                        # Remove milliseconds if present
                        min_timestamp = min_timestamp[:19] if isinstance(min_timestamp, str) else ""
                        max_timestamp = max_timestamp[:19] if isinstance(max_timestamp, str) else ""
                        
                        # Standardize timestamps
                        min_timestamp = standardize_timestamp(min_timestamp)
                        max_timestamp = standardize_timestamp(max_timestamp)
                        
                        # Store in session state
                        st.session_state.min_timestamp = min_timestamp
                        st.session_state.max_timestamp = max_timestamp
                        
                        # Clear progress bar
                        timestamp_progress.empty()
                        
                        st.success("Timestamp range detected!")
                    except Exception as e:
                        st.warning(f"Couldn't detect timestamp range: {e}")
            
            # Initialize timestamp variables if not in session state
            if 'min_timestamp' not in st.session_state:
                st.session_state.min_timestamp = ""
            if 'max_timestamp' not in st.session_state:
                st.session_state.max_timestamp = ""
                
            # Timestamp range inputs with standardized format
            st.subheader("Timestamp Range")
            if st.session_state.min_timestamp and st.session_state.max_timestamp:
                st.write(f"Detected timestamp range: {st.session_state.min_timestamp} to {st.session_state.max_timestamp}")
            
            col1, col2 = st.columns(2)
            with col1:
                start_timestamp_input = st.text_input("Start timestamp (YYYY-MM-DD HH:MM:SS)", value=st.session_state.min_timestamp)
                # Standardize timestamp format when user inputs it
                start_timestamp = standardize_timestamp(start_timestamp_input)
                if start_timestamp != start_timestamp_input and start_timestamp_input:
                    st.info(f"Timestamp standardized to: {start_timestamp}")
            with col2:
                end_timestamp_input = st.text_input("End timestamp (YYYY-MM-DD HH:MM:SS)", value=st.session_state.max_timestamp)
                # Standardize timestamp format when user inputs it
                end_timestamp = standardize_timestamp(end_timestamp_input)
                if end_timestamp != end_timestamp_input and end_timestamp_input:
                    st.info(f"Timestamp standardized to: {end_timestamp}")
            
            # Option to define sections
            define_sections = st.checkbox("Define experiment sections", value=False)
            
            # Section definition (only shown if checkbox is checked)
# ...existing code...

            # Section definition (only shown if checkbox is checked)
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
                            value=start_timestamp,
                            key=f"start_{i}"
                        )
                        # Standardize section start timestamp
                        section_start = standardize_timestamp(section_start_input)
                    with col3:
                        section_end_input = st.text_input(
                            f"End time (YYYY-MM-DD HH:MM:SS)",
                            value=end_timestamp,
                            key=f"end_{i}"
                        )
                        # Standardize section end timestamp
                        section_end = standardize_timestamp(section_end_input)
                    
                    section_data.append({
                        "label": section_label,
                        "start": section_start,
                        "end": section_end
                    })
            
            # Process button
            if st.button("Process Data"):
                with st.spinner('Processing data... This may take several minutes for large files'):
                    progress_bar = st.progress(0)
                    
                    try:
                        # Process the file in chunks
                        result_df = process_eeg_data_in_chunks(
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
                        
                        """                        
                        # Show preview button
                        if st.button("View Processed Data Preview"):
                            st.subheader("Processed Data Preview")
                            head_df = result_df.head(5)
                            tail_df = result_df.tail(5)
                            ellipsis_row = {col: "..." for col in result_df.columns}
                            ellipsis_df = pd.DataFrame([ellipsis_row])
                            preview_df = pd.concat([head_df, ellipsis_df, tail_df], ignore_index=True)
                            st.write(preview_df)
                        """

                        
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

    # Clear All button - improved to clean temporary files
    if st.button("Clear All", key="clear_all_combined"):
        # Clean up any temporary files that might be in session state
        if 'temp_files' in st.session_state:
            for tmp_path in st.session_state.temp_files:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass
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
                        
                        # Read the file in smaller chunks if it's large
                        if os.path.getsize(tmp_path) > 50 * 1024 * 1024:  # If file > 50MB
                            chunks = []
                            for chunk in pd.read_csv(tmp_path, chunksize=500):  # Smaller chunk size
                                chunks.append(chunk)
                            df = pd.concat(chunks, ignore_index=True)
                        else:
                            df = pd.read_csv(tmp_path)
                            
                        all_dfs.append(df)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Store temp files in session state for later cleanup
                    st.session_state.temp_files = temp_files

                    
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
                        
                        """
                        # Show preview button 
                        if st.button("View Combined Data Preview"):
                            st.subheader("Combined Data Preview")
                            head_df = df_final.head(5)
                            tail_df = df_final.tail(5)
                            ellipsis_row = {col: "..." for col in df_final.columns}
                            ellipsis_df = pd.DataFrame([ellipsis_row])
                            preview_df = pd.concat([head_df, ellipsis_df, tail_df], ignore_index=True)
                            st.write(preview_df)
                        """

                        
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

                    # Clean up temp files after processing
                    for tmp_path in temp_files:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                    # Remove from session state
                    if 'temp_files' in st.session_state:
                        del st.session_state.temp_files
                            
                except Exception as e:
                    st.error(f"An error occurred during combining: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                                    # Clean up temp files in case of error
                    for tmp_path in temp_files:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                    # Remove from session state
                    if 'temp_files' in st.session_state:
                        del st.session_state.temp_files

# ...existing code...

def process_eeg_data_in_chunks(file_path, user_num, start_timestamp, end_timestamp, section_data, include_sections=True, progress_callback=None):
    """
    Process EEG data in chunks to handle large files
    """
    # Standardize input timestamps
    start_timestamp = standardize_timestamp(start_timestamp)
    end_timestamp = standardize_timestamp(end_timestamp)
    
    # Standardize section timestamps
    if include_sections and section_data:
        for section in section_data:
            section['start'] = standardize_timestamp(section['start'])
            section['end'] = standardize_timestamp(section['end'])
    
    # Initialize an empty list to store processed chunks
    processed_chunks = []
    
    # Get the total file size for progress tracking
    total_size = os.path.getsize(file_path)
    processed_size = 0
    
    try:
        # Process the file in chunks with smaller chunk size
        chunk_iter = get_chunk_iterator(file_path, chunksize=500)
        
        # Rest of function remains the same...
        
        for i, chunk in enumerate(chunk_iter):
            # Update progress
            processed_size += chunk.memory_usage(deep=True).sum()
            if progress_callback:
                progress_callback(min(processed_size / total_size, 0.5))  # First 50% for reading
            
            # Keep only the rows where 'Elements' is NaN
            chunk = chunk[chunk['Elements'].isna()]
            
            # Keep only the first 25 columns if there are more
            if chunk.shape[1] > 25:
                chunk = chunk.iloc[:, :25]
            
            # Remove rows where all brainwave data values are zero
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
            
            # Add to processed chunks list if not empty
            if not chunk.empty:
                processed_chunks.append(chunk)
        
        # Concatenate all processed chunks
        if processed_chunks:
            df = pd.concat(processed_chunks, ignore_index=True)
        else:
            return pd.DataFrame()  # Return empty DataFrame if no valid data
        
        # Format user number
        if user_num < 10:
            df.insert(0, "User", '0' + str(user_num))
        else:
            df.insert(0, "User", str(user_num))
        
        # Remove the milliseconds from the 'TimeStamp' column
        df['TimeStamp'] = df['TimeStamp'].str[:-4]
        
        # Filter by timestamp range
        df = df[(df['TimeStamp'] >= start_timestamp) & (df['TimeStamp'] <= end_timestamp)]
        
        # Update progress
        if progress_callback:
            progress_callback(0.6)  # 60% complete
        
        # Add Time column
        if not df.empty:
            df.insert(2, "Time", (pd.to_datetime(df['TimeStamp']) - pd.to_datetime(df['TimeStamp'].iloc[0]) + 
                                 pd.to_datetime('0:00:01')).dt.strftime('%H:%M:%S.%f').str[:-3])
        else:
            return pd.DataFrame()  # Return empty DataFrame if filtered data is empty
        
        # Create a copy for resampling
        df_copy = df.copy()
        
        # Update progress
        if progress_callback:
            progress_callback(0.7)  # 70% complete
        
        # Convert Time to datetime and set as index
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
        
        # Merge the resampled DataFrames
        df_final = pd.merge(non_numeric_df, numeric_df, on='Time')
        df_final['Time'] = df_final['Time'].dt.time
        
        # Update progress
        if progress_callback:
            progress_callback(0.8)  # 80% complete
        
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
            progress_callback(1.0)  # 100% complete
        
        return df_final
    
    except Exception as e:
        # Clean up in case of error
        if progress_callback:
            progress_callback(1.0)  # Complete the progress bar
        raise e

if __name__ == "__main__":
    main()
