import os
import boto3
from io import BytesIO
import PyPDF2
from pathlib import Path
from dotenv import load_dotenv

MAX_FILES_TO_DOWNLOAD = 10

# Load environment variables
def load_environment_variables(env_path):
    load_dotenv(env_path)

def create_directory(path):
    os.makedirs(path, exist_ok=True)

def initialize_s3_client(aws_access_key, aws_secret_key):
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

def fetch_pdf_list(s3_client, bucket_name, bucket_prefix):
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=bucket_prefix)
    return [obj for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]

def download_and_extract_text(s3_client, bucket_name, obj_key, file_path):
    pdf_data = BytesIO()
    s3_client.download_fileobj(bucket_name, obj_key, pdf_data)
    pdf_data.seek(0)

    try:
        pdf_reader = PyPDF2.PdfReader(pdf_data)
        text = ''.join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        
        # Change the file extension to .txt
        txt_file_path = file_path.replace('.pdf', '.txt')

        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        return txt_file_path  # Return the path to the generated text file
    except PyPDF2.errors.PdfReadError:
        print(f"Failed to read the PDF file: {obj_key}. Skipping.")
        return None

def main():
    ENV_PATH = str(Path(__file__).parent.parent / "../.env")
    load_environment_variables(ENV_PATH)

    DATASET_DIR = str(Path(__file__).parent.parent / "dataset_download/dataset")
    create_directory(DATASET_DIR)

    AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
    AWS_BUCKET_PREFIX = os.getenv('AWS_BUCKET_PREFIX')
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')

    s3_client = initialize_s3_client(AWS_ACCESS_KEY, AWS_SECRET_KEY)
    pdf_list = fetch_pdf_list(s3_client, AWS_BUCKET_NAME, AWS_BUCKET_PREFIX)
    total_files = len(pdf_list)
    downloaded_files = []

    for pdf_obj in pdf_list:
        file_name = pdf_obj['Key'].split('/')[-1]
        file_path = os.path.join(DATASET_DIR, file_name)

        if os.path.exists(file_path):
            print(f"File {file_name} already exists. Skipping download.")
            continue

        txt_file_path = download_and_extract_text(s3_client, AWS_BUCKET_NAME, pdf_obj['Key'], file_path)
        if txt_file_path:
            downloaded_files.append(txt_file_path)
            remaining_files = total_files - len(downloaded_files)
            print(f"Remaining files to download: {remaining_files}")

        if len(downloaded_files) >= MAX_FILES_TO_DOWNLOAD:
            break

    print("Download process completed.")
    return downloaded_files

if __name__ == "__main__":
    main()
