FROM python

VOLUME /app
ADD ./requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install tensorflow_hub

ENTRYPOINT ["python", "main.py", "--url"]
