### Documentation for `files_2_vector_Store.py`

#### Overview

The `files_2_vector_Store.py` script is designed to process PDF and TXT files, convert them into text, chunkify the text into smaller segments, embed these segments using different embedding models (OpenAI and Hugging Face), and store them in a vector store (specifically Pinecone). It utilizes logging for tracking various stages of processing and error handling for robustness.

#### Dependencies

- `PyPDF2`: For extracting text from PDF files.
- `langchain`: Provides components for text processing and document handling.
- `langchain_community`: Specific embeddings from Hugging Face models.
- `langchain_pinecone`: Integration with Pinecone vector store.
- `langchain_openai`: Integration with OpenAI's embeddings.
- `dotenv`: For loading environment variables from a `.env` file.
- `pinecone`: Python client for Pinecone operations.
- `tiktoken`: Tokenization library for counting tokens in text.
- `re`, `json`, `logging`: Standard Python libraries for regular expressions, JSON handling, and logging functionalities.

#### Environment Variables

The script uses several environment variables loaded via `dotenv` for configuration:

- `OPENAI_API_KEY`, `MODEL_NAME`: Configuration for OpenAI embeddings.
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`: Configuration for Pinecone vector store.
- `DEBUG_LOG`, `DEBUG_LOG_FILE`: Configuration for debugging logs.
- `MAX_RETRIES`, `RETRY_DELAY`: Configuration for retry attempts and delays on connection errors.
- `FILE_LOG`: Directory path for storing log files.

#### Logging

The script configures two log handlers (`application_log.txt` for general logs and `error_log.txt` for errors) using `RotatingFileHandler`. It logs events and errors during the execution of various functions and processes.

#### Classes and Functions

1. **Files_2_vs Class**

   - **`__init__(self, cache_file='cache.json')`**: Initializes the class, loads cache, and sets up configurations including model names, environment variables, and logging settings.

   - **`get_files_paths(self, directory)`**: Retrieves all PDF and TXT file paths from a specified directory, excluding those already processed, and optionally logs the paths for debugging.

   - **`clean_text(self, text)`**: Cleans text by removing special characters and multiple newline characters.

   - **`file_to_text(self, file)`**: Converts a PDF or TXT file to plain text.

   - **`process_text_chunk(self, text_chunk, data)`**: Splits large text chunks into smaller segments (`Document` objects) using `RecursiveCharacterTextSplitter` and logs the chunks for debugging.

   - **`process_file_batch_doc(self, file_docs)`**: Processes a batch of files, converts them to text chunks, tokenizes them, and logs the processed chunks for debugging.

   - **`calculate_batch_size(self, file_paths)`**: Calculates the total size of a list of files in MB.

   - **`count_tokens(self, texts, model_name)`**: Tokenizes text using `tiktoken`, counts tokens, and logs token encodings and counts for debugging.

   - **`create_serverless_index(self, index_name, dimension)`**: Creates a serverless index in Pinecone with specified configurations, waits until the index is ready, and logs index creation status.

   - **`save_data(self, text, filename)`**: Saves data to specified log files, creating directories as needed, and logs the save operation.

   - **`embeddings_scheme(self, model_name)`**: Selects and returns the appropriate embeddings model (OpenAI or Hugging Face) based on the provided `model_name`.

   - **`get_model_name_embedding(self, model_name, embeddings_sch)`**: Retrieves the model name attribute from the embeddings schema.

   - **`get_dimension_to_session(self, model_name)`**: Retrieves the dimension attribute for the specified model name.

   - **`load_cache(self)`**: Loads processed file names from a JSON cache file.

   - **`save_cache(self)`**: Saves processed file names to a JSON cache file.

   - **`add_document_to_pinecone(self, doc, file_name, embeddings_sch)`**: Adds a document (`Document` object) to Pinecone vector store using the specified embeddings schema, handles retries on connection errors, and logs the process.

   - **`main(self, files_docs_list)`**: Main function orchestrating the entire process of file processing, chunkification, embedding, and storage in Pinecone, logging processing times and summaries.

2. **Logging Configuration**

   - Sets up two log handlers (`RotatingFileHandler`) for general and error logs, formatting log messages with timestamps and severity levels.

#### app.py Integration

The `app.py` script integrates with `files_2_vector_Store.py`, initializing the `Files_2_vs` class, creating a serverless index, processing files in batches using `ThreadPoolExecutor` for parallel execution, and logging overall processing times and summaries.

#### Summary

The script efficiently processes large volumes of PDF and TXT files, segments text chunks, applies embeddings, and stores them in a scalable vector store (Pinecone), ensuring robust error handling and detailed logging for monitoring and debugging purposes. Adjustments to batch sizes and parallel processing can be configured to optimize performance based on specific requirements.
