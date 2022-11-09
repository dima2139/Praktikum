# syntax=docker/dockerfile:1
FROM ubuntu:18.04
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

