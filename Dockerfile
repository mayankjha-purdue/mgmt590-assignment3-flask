FROM tensorflow/tensorflow

COPY requirements.txt . 

RUN pip3 install -r requirements.txt 

COPY answer.py /app/answer.py

COPY test_flaskr.py /app/test_flaskr.py

CMD ["python3", "/app/answer.py"]
