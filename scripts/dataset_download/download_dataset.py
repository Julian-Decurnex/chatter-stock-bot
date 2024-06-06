
import os
import boto3
from io import BytesIO
import PyPDF2
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = str(Path(__file__).parent.parent / "../.env")
load_dotenv(ENV_PATH)

DATASET_ROOT_PATH = str(Path(__file__).parent.parent / "dataset_download/dataset")
os.makedirs(DATASET_ROOT_PATH, exist_ok=True)

AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
AWS_BUCKET_PREFIX = os.getenv('AWS_BUCKET_PREFIX')
AWS_ACCESS_KEY=os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY=os.getenv('AWS_SECRET_KEY')
max_files = 10

def fetch_texts_from_s3():
    s3 = boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    response = s3.list_objects_v2(Bucket=AWS_BUCKET_NAME, Prefix=AWS_BUCKET_PREFIX)

    texts = []
    file_count = 0

    total_files = len([obj for obj in response['Contents'] if obj['Key'].endswith('.pdf')])

    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].endswith('.pdf'):
                file_name = obj['Key'].split('/')[-1]
                file_path = os.path.join(DATASET_ROOT_PATH, file_name)

                if os.path.exists(file_path):
                    print(f"File {file_name} already exists. Skipping download.")
                    file_count += 1 
                    continue

                pdf_data = BytesIO()
                s3.download_fileobj(AWS_BUCKET_NAME, obj['Key'], pdf_data)
                pdf_data.seek(0)

                try:
                    pdf_reader = PyPDF2.PdfReader(pdf_data)
                    num_pages = len(pdf_reader.pages)
                    text = ''
                    for page_number in range(num_pages):
                        page = pdf_reader.pages[page_number]
                        try:
                            text += page.extract_text()
                        except TypeError:
                            print(f"Error reading font in file: {file_name}. Skipping this page.")
                            continue

                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(text)

                    print(f"Downloaded file: {file_name}")

                    texts.append(text)
                    file_count += 1

                    remaining_files = total_files - file_count
                    print(f"Remaining files to download: {remaining_files}")

                    if file_count >= max_files:
                        break
                except PyPDF2.errors.PdfReadError:
                    print(f"Error reading file: {file_name}. Skipping this file.")
                    continue

    print("All downloads have been completed.")

    return texts

if __name__ == "__main__":
    fetch_texts_from_s3()