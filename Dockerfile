
# syntax=docker/dockerfile:1
FROM python:3.11.8-bookworm
ARG MASTER_WORKER_ARG
ENV MASTER_WORKER ${MASTER_WORKER_ARG}
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install "gymnasium[accept-rom-license]" # downloads ALE roms, accepts license, installs
RUN pip install -r requirements.txt
RUN pip install pika
#EXPOSE 5000
COPY . .
# TODO not sure if this arg thing works after image is built and pulled?
CMD python main.py ${MASTER_WORKER}
