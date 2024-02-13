
# syntax=docker/dockerfile:1
FROM python:3.11.8-bookworm
ARG MASTER_WORKER_ARG
ENV MASTER_WORKER ${MASTER_WORKER_ARG}
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install "gymnasium[accept-rom-license]" # downloads ALE roms, accepts license, installs
RUN pip install -r requirements.txt
RUN pip install "redis[hiredis]"
RUN pip install pika
#EXPOSE 5000
COPY . .
CMD python main.py ${MASTER_WORKER}
