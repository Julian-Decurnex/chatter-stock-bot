from files_2_vector_store import Files_2_vs
import os
import concurrent.futures
import time
import logging
import json
from logging.handlers import RotatingFileHandler
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PATH_FILES = os.getenv("PATH_FILE")
FILE_LOG = os.getenv("FILE_LOG")

if not os.path.exists(FILE_LOG):
    os.makedirs(FILE_LOG, exist_ok=True)

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Nivel de log global

# Handler to general log
file_handler = RotatingFileHandler('./log/application_log.txt', maxBytes=1000000, backupCount=5)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s (GMT) - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Handler to error log
error_handler = RotatingFileHandler('./log/error_log.txt', maxBytes=1000000, backupCount=5)
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
error_handler.setFormatter(error_formatter)

# add both handler
logger.addHandler(file_handler)
logger.addHandler(error_handler)


def process_file_range(files_2_inst, files_range):
    files_2_inst.main(files_range)

def create_index(files_2_inst):
    # Create a serverless index
    files_2_inst.create_serverless_index(files_2_inst.INDEX_NAME, files_2_inst.get_dimension_to_session(files_2_inst.MODEL_NAME))

if __name__=="__main__":
    files_2_inst = Files_2_vs()
    
    create_index(files_2_inst)
    
    files = files_2_inst.get_files_paths(PATH_FILES)
    
    rang = 20
    # Dividir los archivos en sublistas de tamaño `rang`
    file_ranges = [files[i:i + rang] for i in range(0, len(files), rang)]
    
    start_time = time.time()
    
    # Usar ThreadPoolExecutor para ejecutar en paralelo
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file_range, files_2_inst, file_range) for file_range in file_ranges]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"[app.py] Error processing file range: {e}")
    
    end_time = time.time() - start_time
    time_total = f"{end_time/3600:.2f} hs." if end_time > 3600 else f"{end_time/60:.2f} min." if end_time > 60 else f"{end_time:.2f} seg."
    msg = f"""{'-'*30}\n[app.py] Total de archivos: {len(files)}\nTiempo total: {time_total} \n{'-'*30}"""
    logger.info(msg)
