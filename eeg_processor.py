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

def normalize_timestamp(ts):
    """
    Normaliza un timestamp a formato YYYY-MM-DD HH:MM:SS
    Si la hora, minuto o segundo tiene un solo dígito, agrega un cero.
    """
    import re
    if not ts:
        return ""
    # Busca el patrón de fecha y hora
    match = re.match(r"(\d{4}-\d{2}-\d{2})\s+(\d{1,2}):(\d{1,2}):(\d{1,2})", ts.strip())
    if match:
        date, h, m, s = match.groups()
        return f"{date} {int(h):02d}:{int(m):02d}:{int(s):02d}"
    return ts  # Si no hace match, regresa el original

@st.cache_resource
def get_chunk_iterator(file_path, chunksize=50000):  # Chunk size más pequeño
    """Get a chunk iterator for large CSV files"""
    return pd.read_csv(file_path, chunksize=chunksize, low_memory=False)

def detect_min_max_timestamp(file_path):
    """
    Detecta el timestamp mínimo y máximo del archivo de entrada.
    """
    min_ts, max_ts = "", ""
    try:
        # Lee el primer chunk para el mínimo
        for chunk in pd.read_csv(file_path, chunksize=1000):
            if "TimeStamp" in chunk.columns:
                min_ts = chunk["TimeStamp"].iloc[0]
                break
        # Lee el último chunk para el máximo
        for chunk in pd.read_csv(file_path, chunksize=1000):
            pass
        if "TimeStamp" in chunk.columns:
            max_ts = chunk["TimeStamp"].iloc[-1]
        # Quita milisegundos si existen
        min_ts = min_ts[:19] if isinstance(min_ts, str) else ""
        max_ts = max_ts[:19] if isinstance(max_ts, str) else ""
    except Exception:
        pass
    return min_ts, max_ts

def process_single_user():
    st.header("Single User Data Processing")
    user_num = st.number_input("User Number", min_value=1, max_value=99, value=1)

    if st.button("Clear All", key="clear_all"):
        # Clean up temp file if it exists
        if 'tmp_path' in st.session_state and os.path.exists(st.session_state.tmp_path):
            try:
                os.unlink(st.session_state.tmp_path)
            except Exception:
                pass
        st.session_state.clear()
        st.rerun()

    st.write("Upload your raw EEG CSV file from Mind Monitor (files can be very large, please be patient)")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv", 
        accept_multiple_files=False,
        help="Upload the raw EEG data CSV file from Mind Monitor"
    )

    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, 
                       "FileType": uploaded_file.type, 
                       "FileSize": f"{uploaded_file.size / (1024 * 1024):.2f} MB"}
        st.write(file_details)
        try:
            if 'tmp_path' not in st.session_state:
                # Crear archivo temporal y escribir por chunks para minimizar uso de memoria
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    # Configurar tamaño de buffer pequeño (1MB)
                    chunk_size = 1024 * 1024  # 1MB chunks
                    with st.spinner('Storing uploaded file... this may take a while for large files'):
                        # Mostrar progreso de carga
                        progress_bar = st.progress(0)
                        file_size = uploaded_file.size
                        bytes_processed = 0
                        
                        # Leer y escribir por chunks pequeños, nunca cargando todo el archivo en memoria
                        while True:
                            chunk = uploaded_file.read(chunk_size)
                            if not chunk:
                                break
                                
                            tmp_file.write(chunk)
                            bytes_processed += len(chunk)
                            progress_bar.progress(min(bytes_processed / file_size, 1.0))
                            
                    st.session_state.tmp_path = tmp_file.name

            # Detecta automáticamente los timestamps si no están en session_state
            if 'min_timestamp' not in st.session_state or 'max_timestamp' not in st.session_state:
                min_ts, max_ts = detect_min_max_timestamp(st.session_state.tmp_path)
                st.session_state.min_timestamp = min_ts
                st.session_state.max_timestamp = max_ts

            # Timestamp range inputs con normalización automática
            st.subheader("Timestamp Range")
            if st.session_state.min_timestamp and st.session_state.max_timestamp:
                st.write(f"Detected timestamp range: {st.session_state.min_timestamp} to {st.session_state.max_timestamp}")

            col1, col2 = st.columns(2)
            with col1:
                start_timestamp = st.text_input(
                    "Start timestamp (YYYY-MM-DD HH:MM:SS)",
                    value=st.session_state.min_timestamp,
                    key="start_ts"
                )
            with col2:
                end_timestamp = st.text_input(
                    "End timestamp (YYYY-MM-DD HH:MM:SS)",
                    value=st.session_state.max_timestamp,
                    key="end_ts"
                )

            # Normaliza los timestamps automáticamente
            start_timestamp = normalize_timestamp(start_timestamp)
            end_timestamp = normalize_timestamp(end_timestamp)

            define_sections = st.checkbox("Define experiment sections", value=False)
            section_data = []
            if define_sections:
                st.subheader("Define Sections")
                st.write("Define the different phases/sections of your experiment with their timestamp ranges:")
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
                    # Normaliza los timestamps de sección
                    section_data.append({
                        "label": section_label,
                        "start": normalize_timestamp(section_start),
                        "end": normalize_timestamp(section_end)
                    })

            if st.button("Process Data"):
                with st.spinner('Processing data... This may take several minutes for large files'):
                    progress_bar = st.progress(0)
                    try:
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
                        st.success('Processing complete!')
                        st.session_state.processed_data = result_df
                        st.subheader("Data Summary")
                        st.write(f"Total rows: {len(result_df)}")
                        st.write(f"Time range: {result_df['Time'].iloc[0]} to {result_df['Time'].iloc[-1]}")
                        if define_sections:
                            st.write(f"Sections: {result_df['Section'].nunique()}")
                        output_filename = f"user_{user_num}_processed.csv"
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

def process_eeg_data_in_chunks(file_path, user_num, start_timestamp, end_timestamp, section_data, include_sections=True, progress_callback=None):
    """
    Procesa datos EEG en chunks extremadamente pequeños para minimizar uso de memoria
    """
    # Solo almacenaremos datos resamplados
    resampled_data_buffer = []
    buffer_size_limit = 100  # Cantidad máxima de chunks resamplados en memoria
    
    # Variables para tracking de progreso
    total_size = os.path.getsize(file_path)
    processed_size = 0
    
    # Variable para almacenar el primer timestamp válido (para cálculos relativos)
    first_valid_ts = None
    
    # DataFrame final
    df_final = None
    
    try:
        if progress_callback:
            progress_callback(0.05)  # 5% - Iniciando
            
        # Primera pasada rápida solo para detectar el primer timestamp válido
        # con chunks extremadamente pequeños
        for small_chunk in pd.read_csv(file_path, chunksize=50, low_memory=False):
            small_chunk = small_chunk[small_chunk['Elements'].isna()]
            if not small_chunk.empty:
                small_chunk['TimeStamp'] = small_chunk['TimeStamp'].str[:-4]
                filtered_chunk = small_chunk[(small_chunk['TimeStamp'] >= start_timestamp) & 
                                        (small_chunk['TimeStamp'] <= end_timestamp)]
                if not filtered_chunk.empty:
                    first_valid_ts = pd.to_datetime(filtered_chunk['TimeStamp'].iloc[0])
                    break
            del small_chunk
            
        if progress_callback:
            progress_callback(0.1)  # 10% - Timestamp inicial encontrado
            
        if first_valid_ts is None:
            if progress_callback:
                progress_callback(1.0)
            return pd.DataFrame()
        
        # Procesamiento chunk por chunk extremadamente pequeños
        chunk_iter = pd.read_csv(file_path, chunksize=50, low_memory=False)
        
        for i, chunk in enumerate(chunk_iter):
            # Actualizar progreso
            processed_size += chunk.memory_usage(deep=True).sum()
            if progress_callback and i % 20 == 0:  # Actualizar cada 20 chunks para no sobrecargar la UI
                progress_callback(min(0.1 + processed_size / total_size * 0.6, 0.7))  # 10%-70%
            
            # FILTRADO BÁSICO
            # Filtrar filas con Elements no nulos y aplicar otros filtros
            chunk = chunk[chunk['Elements'].isna()]
            if chunk.shape[1] > 25:
                chunk = chunk.iloc[:, :25]
                
            # Filtro de filas con valores de ondas cerebrales en cero
            zero_mask = ((chunk['Delta_TP9'] == 0) & (chunk['Delta_AF7'] == 0) & 
                        (chunk['Delta_AF8'] == 0) & (chunk['Delta_TP10'] == 0) & 
                        (chunk['Theta_TP9'] == 0) & (chunk['Theta_AF7'] == 0) & 
                        (chunk['Theta_AF8'] == 0) & (chunk['Theta_TP10'] == 0) & 
                        (chunk['Alpha_TP9'] == 0) & (chunk['Alpha_AF7'] == 0) & 
                        (chunk['Alpha_AF8'] == 0) & (chunk['Alpha_TP10'] == 0) & 
                        (chunk['Beta_TP9'] == 0) & (chunk['Beta_AF7'] == 0) & 
                        (chunk['Beta_AF8'] == 0) & (chunk['Beta_TP10'] == 0) & 
                        (chunk['Gamma_TP9'] == 0) & (chunk['Gamma_AF7'] == 0) & 
                        (chunk['Gamma_AF8'] == 0) & (chunk['Gamma_TP10'] == 0))
            chunk = chunk[~zero_mask]
            
            # Saltar chunks vacíos
            if chunk.empty:
                continue
                
            # Formato y filtros adicionales
            if user_num < 10:
                chunk.insert(0, "User", '0' + str(user_num))
            else:
                chunk.insert(0, "User", str(user_num))
            
            # Eliminar milisegundos y filtrar por timestamp
            chunk['TimeStamp'] = chunk['TimeStamp'].str[:-4]
            chunk = chunk[(chunk['TimeStamp'] >= start_timestamp) & 
                          (chunk['TimeStamp'] <= end_timestamp)]
            
            if chunk.empty:
                continue
            
            # RESAMPLING DE ESTE CHUNK
            # Agregar columna Time
            chunk.insert(2, "Time", (pd.to_datetime(chunk['TimeStamp']) - first_valid_ts + 
                                   pd.to_datetime('0:00:01')).dt.strftime('%H:%M:%S.%f').str[:-3])
            
            # Convertir a datetime
            chunk['Time'] = pd.to_datetime(chunk['Time'])
            
            # Resampleo
            df_temp = chunk.copy()
            del chunk
            
            df_temp.set_index('Time', inplace=True)
            
            # Procesar columnas numéricas y no numéricas
            numeric_cols = df_temp.select_dtypes(include=['float64', 'int64'])
            if numeric_cols.empty:
                del df_temp, numeric_cols
                continue
                
            numeric_resampled = numeric_cols.resample('s').mean()
            del numeric_cols
            
            non_numeric_cols = df_temp.select_dtypes(exclude=['float64', 'int64'])
            if non_numeric_cols.empty:
                del df_temp, non_numeric_cols, numeric_resampled
                continue
                
            non_numeric_resampled = non_numeric_cols.resample('s').first()
            del non_numeric_cols
            del df_temp
            
            # Combinar columnas resampladas
            resampled = pd.merge(non_numeric_resampled, numeric_resampled, 
                                left_index=True, right_index=True)
            del non_numeric_resampled, numeric_resampled
            
            resampled.reset_index(inplace=True)
            
            # Almacenar en buffer
            resampled_data_buffer.append(resampled)
            del resampled
            
            # Si el buffer llega al límite, combinar y liberar memoria
            if len(resampled_data_buffer) >= buffer_size_limit:
                buffer_df = pd.concat(resampled_data_buffer, ignore_index=True)
                
                if df_final is None:
                    df_final = buffer_df
                else:
                    df_final = pd.concat([df_final, buffer_df], ignore_index=True)
                
                # Limpiar buffer
                resampled_data_buffer = []
                del buffer_df
        
        # Procesar cualquier dato restante en el buffer
        if resampled_data_buffer:
            buffer_df = pd.concat(resampled_data_buffer, ignore_index=True)
            
            if df_final is None:
                df_final = buffer_df
            else:
                df_final = pd.concat([df_final, buffer_df], ignore_index=True)
            
            del resampled_data_buffer, buffer_df
        
        # Verificar si tenemos datos procesados
        if df_final is None or df_final.empty:
            if progress_callback:
                progress_callback(1.0)
            return pd.DataFrame()
        
        if progress_callback:
            progress_callback(0.8)  # 80% - Procesamiento principal completado
        
        # FINALIZACIÓN
        # Eliminar duplicados
        df_final = df_final.drop_duplicates(subset=['TimeStamp'])
        
        # Convertir Time a formato de tiempo
        df_final['Time'] = df_final['Time'].dt.time
        
        # Organizar y ordenar
        df_final.sort_values('TimeStamp', inplace=True)
        df_final = df_final[['User', 'TimeStamp', 'Time'] + 
                          [col for col in df_final.columns if col not in ['User', 'TimeStamp', 'Time']]]
        
        # Agregar secciones si es necesario
        if include_sections and section_data:
            df_final['Section'] = None
            
            # Procesar secciones por lotes para minimizar operaciones en memoria
            for section in section_data:
                mask = (df_final['TimeStamp'] >= section['start']) & (df_final['TimeStamp'] <= section['end'])
                df_final.loc[mask, 'Section'] = section['label']
            
            cols = df_final.columns.tolist()
            cols.remove('Section')
            cols.insert(3, 'Section')
            df_final = df_final[cols]
        
        # Eliminar NaNs
        df_final = df_final.dropna()
        
        if progress_callback:
            progress_callback(1.0)
        
        return df_final
    
    except Exception as e:
        if progress_callback:
            progress_callback(1.0)
        if 'resampled_data_buffer' in locals():
            del resampled_data_buffer
        if 'df_final' in locals() and df_final is not None:
            del df_final
        raise e

if __name__ == "__main__":
    main()
