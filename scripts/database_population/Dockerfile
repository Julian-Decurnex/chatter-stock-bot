# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo requirements.txt en el directorio de trabajo
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --upgrade pip
RUN pip install -qU \
openai==0.27.7 \
pinecone-client==3.1.0 \
pinecone-datasets==0.7.0 \
langchain==0.1.1 \
langchain-community==0.0.13 \
tiktoken>=0.7
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de la aplicación en el directorio de trabajo
COPY . .
     
# Definir el comando por defecto para ejecutar tu aplicación
CMD ["python", "app.py"]
# CMD ["python", "delete_vect_metadata.py"]
# CMD ["python", "query_dbvector.py"]
