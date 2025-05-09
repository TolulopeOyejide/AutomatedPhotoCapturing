FROM python:3.10
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -requirements.txt
COPY . .
CMD ["python", "demo.py"]