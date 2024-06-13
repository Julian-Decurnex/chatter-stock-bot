from PyPDF2 import PdfReader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.pydantic_v1 import SecretStr

import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import time
import tiktoken
import re
import json
import logging
from logging.handlers import RotatingFileHandler

from requests.exceptions import ConnectionError, Timeout
from urllib3.exceptions import NewConnectionError

load_dotenv()
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

class Files_2_vs:
    def __init__(self, cache_file='cache.json'):
        self.cache_file = cache_file
        self.processed_files = self.load_cache()

        self.OPENAI_MODELS = ["text-embedding-3-small", "text-embedding-ada-002"]
        self.HUGGING_FACE_MODELS = ["hkunlp/instructor-large", "hkunlp/instructor-xl", "microsoft/Phi-3-vision-128k-instruct"]
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        self.MODEL_NAME = os.getenv("MODEL_NAME")
        self.INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.DEBUG_LOG = os.getenv("DEBUG_LOG").lower()
        self.DEBUG_LOG_FILE = os.getenv("DEBUG_LOG_FILE")    
        self.MAX_RETRIES = int(os.getenv("MAX_RETRIES"))
        self.RETRY_DELAY = os.getenv("RETRY_DELAY")
        self.FILE_LOG = os.getenv("FILE_LOG")
        self.HUGGING_FACE_KWARGS = {
            'model_kwargs': {'device': 'cpu'},
            'encode_kwargs': {'normalize_embeddings': True}
        }
        self.MODEL_NAME_ATTRIBUTE_MAP = {
            **{model: 'model' for model in self.OPENAI_MODELS},
            **{model: 'model_name' for model in self.HUGGING_FACE_MODELS}
        }
        self.DIMENSION_ATTRIBUTE_MAP = {
            **{model: '1536' for model in self.OPENAI_MODELS},
            **{model: '768' for model in self.HUGGING_FACE_MODELS}
        }

    # Function to get all PDF file paths from a specified directory for testing
    def get_files_paths(self, directory):
        file_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if (file.lower().endswith('.pdf') or file.lower().endswith('.txt')) and file.split("/")[-1] not in self.processed_files:
                    file_paths.append(os.path.join(root, file))
        if self.DEBUG_LOG == 'true':
            self.save_data(str(file_paths), f"{self.DEBUG_LOG_FILE}/log_file_paths.txt")
        return file_paths

    def clean_text(self, text):
        # Delete special character
        cleaned_text = re.sub(r'[^\w\s]', '', text)
        cleaned_text = re.sub(r'\n+', '\n', cleaned_text)
        return cleaned_text

    # Convert a single file to text
    def file_to_text(self, file):
        text = ""
        extention = file.split(".")[-1]
        if extention == "pdf":
            pdf_reader = PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif extention == "txt":
            with open(file, 'r') as file:
                text = file.read()
        
        return self.clean_text(text)

    # Function to create and split documents
    def process_text_chunk(self, text_chunk, data):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        document_splitter = text_splitter.create_documents([text_chunk])
        chunks = [
            Document(page_content=chunk.page_content, metadata={"index":i, **data}) 
            for i, chunk in enumerate(document_splitter)
            ]
        
        if self.DEBUG_LOG == 'true':
            self.save_data(f"{str(chunks)}\n\n\n\n", f"{self.DEBUG_LOG_FILE}/log_chunks.txt")

        return chunks

    # Process a batch of Files and split text into chunks
    def process_file_batch_doc(self, file_docs):
        all_chunks = []
        
        # get name file
        name_file = file_docs.split("/")[-1]
        # split name file to company and year
        name_file_split = name_file.split("_")
        company, p_year = "_".join(name_file_split[0:2]) , name_file_split[2].split(".")[0]
        year = re.search(r'\b\d{4}\b', p_year).group(0)
        try:
            # Detect files in cache
            if name_file not in self.processed_files:
                # Convert file to text
                text = self.file_to_text(file_docs)
                
                # Split text to recommended limit before chuck
                metadata = {"name": name_file, "company":company, "year":year}
                all_chunks.extend(self.process_text_chunk(text, metadata))

                # Save data to cache
                self.processed_files.append(name_file)
                self.save_cache()
                    
            if self.DEBUG_LOG == 'true':
                self.save_data(f"{str(all_chunks)}\n\n\n\n", f"{self.DEBUG_LOG_FILE}/log_all_chunks.txt")
        except Exception as e:
            logging.error(f"Error processing file {file_docs}: {e}")
            
        return all_chunks

    # Calculate the size of a list of files in MB
    def calculate_batch_size(self, file_paths):
        total_size = os.path.getsize(file_paths)
        return total_size / (1024 * 1024)  # Convert bytes to MB

    # Tokenize text and count tokens using tiktoken
    def count_tokens(self, texts, model_name):
        if model_name in self.HUGGING_FACE_MODELS:
           encoding = tiktoken.get_encoding("cl100k_base")
        if model_name in self.OPENAI_MODELS:
           encoding = tiktoken.encoding_for_model(model_name)
        total_tokens = sum(len(encoding.encode(text.page_content)) for text in texts)    
        
        if self.DEBUG_LOG == 'true':
            self.save_data(str([encoding.encode(text.page_content) for text in texts]), f"{self.DEBUG_LOG_FILE}/log_encoding.txt")
            self.save_data(str(total_tokens), f"{self.DEBUG_LOG_FILE}/log_total_tokens.txt")
        return total_tokens

    # Create a serverless index
    def create_serverless_index(self, index_name, dimension):
        logging.info("------ Creating Serverless Index ------")
        # Initialize Pinecone
        pc = Pinecone(api_key=self.PINECONE_API_KEY)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD"),
                    region=os.getenv("PINECONE_REGION")
                ) 
            )
            # wait for index to be initialized
            while not pc.describe_index(index_name).status['ready']:
                time.sleep(1)
            logging.info("------ Created Serverless Index ------")
        else:
            logging.info("------ Serverless Index already exist!!!------")

    # Save data in logs folder
    def save_data(self, text, filename):
        folders = filename.split("/")[:-1]
        path = "/".join(folders)

        if not os.path.exists("/".join(path)):
            os.makedirs(path, exist_ok=True)

        with open(filename, 'a') as file:
            file.write(text)
        logging.info(f"El contenido ha sido guardado en {filename}")

    # Embeddings Model
    def embeddings_scheme(self, model_name):
        embeddings_model = None

        if model_name in self.OPENAI_MODELS:
            embeddings_model = OpenAIEmbeddings(openai_api_key = SecretStr(self.OPENAI_API_KEY), model=model_name)
        
        if model_name in self.HUGGING_FACE_MODELS:
            embeddings_model =  HuggingFaceInstructEmbeddings(
                model_name=model_name,
                **self.HUGGING_FACE_KWARGS
            )
        return embeddings_model

    # get type of model by embedding model
    def get_model_name_embedding(self, model_name, embeddings_sch):
        attribute = self.MODEL_NAME_ATTRIBUTE_MAP.get(model_name)
        return getattr(embeddings_sch, attribute, None)

    # Get dimension by model
    def get_dimension_to_session(self, model_name):
        return int(self.DIMENSION_ATTRIBUTE_MAP.get(model_name))

    # Load data to cache
    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as file:
                return json.load(file)
        else:
            empty_list = []
            with open(self.cache_file, 'w') as json_file:
                json.dump(empty_list, json_file)
            return empty_list
    
    # Save data to cache
    def save_cache(self):
        with open(self.cache_file, 'w') as file:
            json.dump(self.processed_files, file)


    def add_document_to_pinecone(self, doc, file_name, embeddings_sch):
        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                PineconeVectorStore.from_documents([doc], index_name=self.INDEX_NAME, embedding=embeddings_sch)
                logging.info(f"Data added to PineconeVectorStore | file name: {file_name}")
                break
            except (ConnectionError, NewConnectionError, Timeout) as conn_err:
                logging.error(f"Connection error while adding document {file_name} (attempt {retries + 1}/{self.MAX_RETRIES}): {conn_err}")
                retries += 1
                time.sleep(self.RETRY_DELAY * 2**retries)  # Exponential backoff
            except Exception as e:
                logging.error(f"Failed to add document {file_name} (attempt {retries + 1}/{self.MAX_RETRIES}): {e}")
                retries += 1
                time.sleep(self.RETRY_DELAY * 2**retries)

    # Man function (execute this)
    def main(self, files_docs_list):
        file_added = []
        
        # Determine the embedding
        embeddings_sch = self.embeddings_scheme(self.MODEL_NAME)

        # Create a serverless index
        # self.create_serverless_index(self.INDEX_NAME, self.get_dimension_to_session(self.MODEL_NAME)) # size chuck
            
        batch_size = 1  # Adjust batch size as needed
        total_processing_time = 0
        batch_times = []
        batch_sizes = []
        total_tokens = 0
        
        for file in files_docs_list:
            # batch_docs = file_docs[i:i+batch_size]
            batch_size_mb = self.calculate_batch_size(file)
            batch_sizes.append(batch_size_mb)
            
            start_time = time.time()
            # logging.info(f"--------Processing batch {i//batch_size + 1}--------")
            
            # Time the chunkification process
            logging.info("--------Initializing chunkification--------")
            text_chunks = self.process_file_batch_doc(file)
                       
            chunkification_time = time.time() - start_time

            # Detect if text_chunks had values
            if text_chunks:
                # Count tokens for the batch
                tokens_count = self.count_tokens(text_chunks, self.get_model_name_embedding(self.MODEL_NAME, embeddings_sch))
                total_tokens += tokens_count

                # Time the embeddings creation process
                start_time = time.time()
                
                logging.info(f"--------Process VectorStore--------")
                for i, doc in enumerate(text_chunks):
                    file_name = doc.metadata['name']
                    if file_name not in file_added:
                        logging.info(f"File added: {file_name}")
                        file_added.append(file_name)
                    
                    self.add_document_to_pinecone(doc, file_name, embeddings_sch)
                    self.save_data(f"{file_name}_{i}\n", f"{self.FILE_LOG}/file_index_process_pinecone.txt")

                embeddings_creation_time = time.time() - start_time
                batch_time = chunkification_time + embeddings_creation_time
                batch_times.append(batch_time)
                total_processing_time += batch_time

                # logging.info(f"--------Batch {i//batch_size + 1} completed in {batch_time:.2f} seconds--------")
                logging.info(f"--------Batch size: {batch_size_mb:.2f} MB--------")

        average_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
        total_batch_size = sum(batch_sizes)
        average_batch_size = total_batch_size / len(batch_sizes) if batch_sizes else 0
        average_mb_per_sec = f"{total_batch_size / total_processing_time:.2f}" if (total_processing_time > 0) else ""
        msg =f"""----------------------- AVERAGE SUMMARY -----------------------
        Average processing time per batch: {average_batch_time:.2f} seconds
        Average batch size: {average_batch_size:.2f} MB
        Average tokens used per mb: {total_tokens/total_batch_size:.2f}
        Average MB per second: {average_mb_per_sec}
        Total processing time: {total_processing_time:.2f} seconds
        Total size processed: {total_batch_size:.2f} MB
        Total tokens used: {total_tokens}
        """
        logging.info(msg)