FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

# Upgrade pip first
RUN pip install --upgrade pip


RUN pip install -r requirements.txt

RUN pip uninstall -y pinecone-plugin-inference || true

# RUN pip install pinecone_plugin_interface


CMD ["python3", "app.py"]
