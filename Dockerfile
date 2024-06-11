FROM python:3.11-slim

WORKDIR /app


COPY requirements.txt .

RUN pip install --upgrade pip && \
pip install -r requirements.txt

# Copiar el resto de la aplicación en el directorio de trabajo
COPY stock_info.csv .
COPY main.py .

EXPOSE 8000

# Definir el comando por defecto para ejecutar tu aplicación
CMD ["chainlit", "run", "main.py", "-w"]