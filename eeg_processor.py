import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import datetime
from io import StringIO
import re
import gc




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
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Single User Processing", "Combine Multiple Users", "Memory Diagnostics"])
    
    # Tab 1: Single User Processing
    with tab1:
        process_single_user()
    
    # Tab 2: Combine Multiple Users
    with tab2:
        combine_multiple_users()
    
    # Tab 3: Memory diagnostics 
    with tab3:
        memory_diagnostics()



# Modify function to get only needed columns
@st.cache_resource(max_entries=1)  # Limit to one cached item to prevent accumulation
def get_chunk_iterator(file_path, chunksize=50):  # Even smaller chunks
    """Get a chunk iterator for large CSV files"""
    # Detect columns we actually need to reduce memory usage
    usecols = None
    try:
        # Read only the header to determine columns
        header = pd.read_csv(file_path, nrows=0)
        necessary_cols = ['TimeStamp', 'Elements']
        # Add brain wave columns if they exist
        for prefix in ['Delta_', 'Theta_', 'Alpha_', 'Beta_', 'Gamma_']:
            necessary_cols.extend([col for col in header.columns if col.startswith(prefix)])
        # Only keep columns that exist in the file
        usecols = [col for col in necessary_cols if col in header.columns]
    except:
        pass
    
    # Use dtype optimization for common columns to reduce memory usage
    dtypes = {'TimeStamp': 'str'}
    
    return pd.read_csv(file_path, chunksize=chunksize, low_memory=False, 
                      usecols=usecols if usecols else None,
                      dtype=dtypes)


def init_tracking_object():
    tracemalloc.start(10)
    return {
        "runs": 0,
        "tracebacks": {},
        "snapshot": None
    }

# Memory diagnostics tab implementation
def memory_diagnostics():
    st.header("Memory Diagnostics")
    
    # Import memory tracking module
    try:
        from memory_tracking import compare_snapshots, find_leak_sources
        
        st.write("""This tab helps diagnose memory issues in the application. 
                 Click the buttons below to run diagnostics.""")
        
        if st.button("Check Memory Usage"):
            current_usage = get_current_memory_usage()
            st.metric("Current Memory Usage", f"{current_usage:.1f} MB")
        
        if st.button("Detect Memory Leaks"):
            compare_snapshots()
        
        if st.button("Find Leak Sources"):
            find_leak_sources()
            
        if st.button("Force Garbage Collection"):
            before = get_current_memory_usage()
            gc.collect()
            after = get_current_memory_usage()
            st.write(f"Memory before GC: {before:.1f} MB")
            st.write(f"Memory after GC: {after:.1f} MB")
            st.write(f"Memory freed: {(before - after):.1f} MB")
            
    except ImportError:
        st.error("""Memory tracking module not found. 
                 Please create the memory_tracking.py file using the code provided.""")

def get_current_memory_usage():
    """Get current memory usage of the Python process in MB"""
    import psutil
    import os
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def compare_snapshots():
    """Compare memory snapshots to detect leaks"""
    _TRACES = init_tracking_object()
    
    # Force garbage collection
    gc.collect()
    
    # Take new snapshot
    snapshot = tracemalloc.take_snapshot()
    
    if _TRACES["snapshot"] is not None:
        diff = snapshot.compare_to(_TRACES["snapshot"], "lineno")
        diff = [d for d in diff if d.count_diff > 0]
        
        # Track potential leaks
        _TRACES["runs"] = _TRACES["runs"] + 1
        tracebacks = set()
        
        for sd in diff:
            for t in sd.traceback:
                tracebacks.add(t)
        
        # Update traceback counts
        if "tracebacks" not in _TRACES or len(_TRACES["tracebacks"]) == 0:
            for t in tracebacks:
                _TRACES["tracebacks"][str(t)] = 1
        else:
            oldTracebacks = set(_TRACES["tracebacks"].keys())
            intersection = tracebacks.intersection([str(t) for t in oldTracebacks])
            
            # Update counts for repeated tracebacks
            evictions = set()
            for t in _TRACES["tracebacks"]:
                if str(t) not in intersection:
                    evictions.add(t)
                else:
                    _TRACES["tracebacks"][str(t)] = _TRACES["tracebacks"][str(t)] + 1
            
            # Remove non-repeated traces
            for t in evictions:
                del _TRACES["tracebacks"][t]
        
        # Display results
        if _TRACES["runs"] > 1:
            st.write(f'After {_TRACES["runs"]} runs the following traces were collected:')
            st.write(json.dumps(_TRACES["tracebacks"], sort_keys=True, indent=4))
    
    # Update snapshot
    _TRACES["snapshot"] = snapshot

def find_leak_sources():
    """Find objects potentially causing memory leaks"""
    gc.collect()
    
    # Look for session state objects that might be leaked
    leaked_objects = []
    for obj in gc.get_objects():
        if 'SessionState' in str(type(obj)) and obj is not st.session_state:
            leaked_objects.append(obj)
        
        # Also check for DataFrame objects not being released
        if 'pandas.core.frame.DataFrame' in str(type(obj)):
            if not hasattr(obj, '_is_view') and len(obj) > 1000:  # Only track large dataframes
                leaked_objects.append(obj)
    
    if leaked_objects:
        st.write(f"Found {len(leaked_objects)} potential leak sources")
        for i, obj in enumerate(leaked_objects[:5]):  # Limit to first 5
            st.write(f"Object {i+1}: {type(obj)}")
            st.write(f"Size: ~{len(obj) if hasattr(obj, '__len__') else 'unknown'}")
    else:
        st.write("No obvious leak sources found")

# Helper function for proper resource cleanup
def cleanup_resources():
    """Clean up all temporary resources and free memory"""
    # Clean up temp files
    if 'tmp_path' in st.session_state and os.path.exists(st.session_state.tmp_path):
        try:
            os.unlink(st.session_state.tmp_path)
        except Exception:
            pass
    
    # Clean up any stored DataFrames
    if 'processed_data' in st.session_state:
        del st.session_state.processed_data
    
    # Clean up any temporary files in combine function
    if 'temp_files' in st.session_state:
        for tmp_path in st.session_state.temp_files:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        del st.session_state.temp_files
    
    # Clear the session state
    st.session_state.clear()
    
    # Force garbage collection
    gc.collect()

def process_single_user():
    # Add data sampling option for very large files
    st.header("Single User Data Processing")
    
    # Add sampling options
    col1, col2 = st.columns(2)
    with col1:
        enable_sampling = st.checkbox("Enable data sampling (for large files)", value=False)
    with col2:
        sample_rate = st.slider("Sampling rate (%)", 1, 100, 20, disabled=not enable_sampling)
    
    # User number input
    user_num = st.number_input("User Number", min_value=1, max_value=99, value=1)

    # Clear All button with improved cleanup
    if st.button("Clear All", key="clear_all"):
        cleanup_resources()
        st.rerun()
    
    # File upload with memory warning
    st.write("Upload your raw EEG CSV file from Mind Monitor (files can be very large, please be patient)")
    st.warning("Processing files larger than 500MB may take several minutes and could fail if memory is limited.")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv", 
        accept_multiple_files=False,
        help="Upload the raw EEG data CSV file from Mind Monitor"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        file_details = {"Filename": uploaded_file.name, 
                       "FileType": uploaded_file.type, 
                       "FileSize": f"{file_size_mb:.2f} MB"}
        st.write(file_details)
        
        # Show additional warning for very large files
        if file_size_mb > 500:
            st.warning(f"This file is very large ({file_size_mb:.1f} MB). Consider using a more restrictive timestamp range to reduce processing time and memory usage.")
        
        try:
            # Create a temporary file to store the uploaded data - do this only once
            if 'tmp_path' not in st.session_state:
                with st.spinner("Saving uploaded file for processing..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                        # Write in chunks to reduce memory usage
                        chunk_size = 5 * 1024 * 1024  # 5MB chunks
                        uploaded_file.seek(0)
                        while True:
                            chunk = uploaded_file.read(chunk_size)
                            if not chunk:
                                break
                            tmp_file.write(chunk)
                        st.session_state.tmp_path = tmp_file.name
                
                # Detect timestamp range more efficiently
                with st.spinner("Detecting timestamp range (this may take a moment for large files)..."):
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

def process_eeg_data_in_chunks(file_path, user_num, start_timestamp, end_timestamp, section_data, 
                              include_sections=True, progress_callback=None, sampling_enabled=False, 
                              sample_rate=100):    
    """
    Process EEG data in chunks to handle large files with optimized memory usage
    """
    # Standardize input timestamps
    start_timestamp = standardize_timestamp(start_timestamp)
    end_timestamp = standardize_timestamp(end_timestamp)
    
    # Standardize section timestamps
    if include_sections and section_data:
        for section in section_data:
            section['start'] = standardize_timestamp(section['start'])
            section['end'] = standardize_timestamp(section['end'])
    
    # Initialize an empty list to store processed data (we'll use append instead of concat for memory efficiency)
    processed_data = []
    
    # Get the total file size for progress tracking
    total_size = os.path.getsize(file_path)
    processed_size = 0

    # Apply sampling logic if enabled
    sample_every_n_rows = 100 // sample_rate if sampling_enabled else 1
    
    # Write processed chunks directly to disk to avoid memory issues
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    temp_output_path = temp_output_file.name
    temp_output_file.close()
    
    # First write the header
    header_written = False
    rows_processed = 0
    total_rows = 0
    
    try:
        # Process the file in smaller chunks 
        chunk_iter = get_chunk_iterator(file_path, chunksize=100)
        
        # First pass: Process and write filtered data
        chunk_iter = get_chunk_iterator(file_path, chunksize=50)
        
        for i, chunk in enumerate(chunk_iter):
            # Apply sampling if enabled
            if sampling_enabled and sample_every_n_rows > 1:
                chunk = chunk.iloc[::sample_every_n_rows]
            
            # Filter chunk by timestamp
            if 'TimeStamp' in chunk.columns:
                chunk['TimeStamp'] = chunk['TimeStamp'].astype(str).str[:-4]
                chunk = chunk[(chunk['TimeStamp'] >= start_timestamp) & (chunk['TimeStamp'] <= end_timestamp)]
            
            # Skip if empty after filtering
            if chunk.empty:
                continue
                
            # Keep only the rows where 'Elements' is NaN
            if 'Elements' in chunk.columns:
                chunk = chunk[chunk['Elements'].isna()]
            
            # Keep only the first 25 columns if there are more (do this early to save memory)
            if chunk.shape[1] > 25:
                chunk = chunk.iloc[:, :25]
            
            # Filter out zero brain wave data
            # Use a simplified condition to save memory
            zero_mask = ((chunk.filter(like='Delta_').mean(axis=1) == 0) & 
                        (chunk.filter(like='Theta_').mean(axis=1) == 0) & 
                        (chunk.filter(like='Alpha_').mean(axis=1) == 0) & 
                        (chunk.filter(like='Beta_').mean(axis=1) == 0) & 
                        (chunk.filter(like='Gamma_').mean(axis=1) == 0))
            chunk = chunk[~zero_mask]
            
            # Add to processed data list if not empty
            if not chunk.empty:
                processed_data.append(chunk)
                
            # Explicitly delete variables to free memory
            del chunk
            if 'zero_mask' in locals():
                del zero_mask
            
            # Force garbage collection
            import gc
            gc.collect()
        
        # Check if we have any data
        if not processed_data:
            return pd.DataFrame()  # Return empty DataFrame if no valid data
            
        if progress_callback:
            progress_callback(0.5)  # 50% complete
            
        # Process the data in smaller batches to save memory
        batch_size = 5  # Process 5 chunks at a time
        final_chunks = []
        
        for i in range(0, len(processed_data), batch_size):
            batch = processed_data[i:i+batch_size]
            
            # Concatenate this batch
            df_batch = pd.concat(batch, ignore_index=True)
            
            # Format user number
            if 'User' not in df_batch.columns:
                if user_num < 10:
                    df_batch.insert(0, "User", '0' + str(user_num))
                else:
                    df_batch.insert(0, "User", str(user_num))
            
            # Add Time column if needed
            if 'Time' not in df_batch.columns and not df_batch.empty:
                try:
                    df_batch.insert(2, "Time", (pd.to_datetime(df_batch['TimeStamp']) - 
                                           pd.to_datetime(df_batch['TimeStamp'].iloc[0]) + 
                                           pd.to_datetime('0:00:01')).dt.strftime('%H:%M:%S.%f').str[:-3])
                except:
                    # If there's an error with time calculation, use a simple index
                    df_batch.insert(2, "Time", [f"00:{idx//60:02d}:{idx%60:02d}" for idx in range(len(df_batch))])
            
            final_chunks.append(df_batch)
            
            # Write to temp file
            if not header_written:
                chunk.to_csv(temp_output_path, mode='w', index=False)
                header_written = True
            else:
                chunk.to_csv(temp_output_path, mode='a', header=False, index=False)
            
            rows_processed += len(chunk)
            total_rows = i * 50  # Estimate based on chunk number
            
            # Free memory
            del chunk
            gc.collect()
            
            # Update progress
            if progress_callback:
                progress_callback(0.5 + ((i / len(processed_data)) * 0.3))
        
        # Now concatenate the processed batches
        df_final = pd.concat(final_chunks, ignore_index=True)
        
        # Clean up to free memory
        del final_chunks, processed_data
        gc.collect()
        
        if progress_callback:
            progress_callback(0.8)  # 80% complete
        
    # Read back the processed data
    if rows_processed > 0:
        # Read back in smaller chunks if file is large
        if os.path.getsize(temp_output_path) > 100 * 1024 * 1024:  # 100MB
            final_dfs = []
            for chunk in pd.read_csv(temp_output_path, chunksize=1000):
                final_dfs.append(chunk)
            df_final = pd.concat(final_dfs)
            del final_dfs
            gc.collect()
        else:
            df_final = pd.read_csv(temp_output_path)
        
        # Organize columns
        column_order = ['User', 'TimeStamp', 'Time']
        remaining_cols = [col for col in df_final.columns if col not in column_order]
        df_final = df_final[column_order + remaining_cols]
        
        # Add Section column if needed
        if include_sections and section_data:
            df_final['Section'] = None
            for section in section_data:
                mask = (df_final['TimeStamp'] >= section['start']) & (df_final['TimeStamp'] <= section['end'])
                df_final.loc[mask, 'Section'] = section['label']
                del mask  # Free memory
                gc.collect()
            
            # Reorder columns to place 'Section' in the 4th position
            cols = df_final.columns.tolist()
            if 'Section' in cols:
                cols.remove('Section')
                cols.insert(3, 'Section')
                df_final = df_final[cols]
        
            # Complete progress
            if progress_callback:
                progress_callback(1.0)
                
            return df_final
        else:
            return pd.DataFrame()

        except Exception as e:
            raise e
        finally:
            # Always clean up the temporary file
            if os.path.exists(temp_output_path):
                try:
                    os.unlink(temp_output_path)
                except:
                    pass

if __name__ == "__main__":
    main()
