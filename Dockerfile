

# The first instruction is what image we want to base our container on
# We Use an official Python runtime as a parent image
FROM python:3.10
RUN apt update && apt install ffmpeg libsm6 libxext6  -y

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1


# Allows docker to cache installed dependencies between builds
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --proxy=192.168.4.16:8081

# Mounts the application code to the image
COPY . code
WORKDIR /code
