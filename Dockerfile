FROM tensorflow/tensorflow

COPY requirements.txt . 

RUN pip3 install -r requirements.txt 

COPY answer.py /app/answer.py

COPY ssl/server-ca.pem /app/ssl/server-ca.pem

COPY ssl/client-cert.pem /app/ssl/client-cert.pem

COPY ssl/client-key.pem /app/ssl/client-key.pem

CMD ["python3", "/app/answer.py"]
