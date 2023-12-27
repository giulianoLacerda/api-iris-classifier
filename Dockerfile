FROM python:3.9

ARG MODE_DEPLOY
ARG TAG
ARG PORT_SERVER
ENV WORKERS=1
ENV THREADS=2
ENV TIMEOUT=0
ENV PYTHONUNBUFFERED True

ADD / /app

WORKDIR /app
RUN ls -lf

# Python dependencies
RUN pip install -r /app/requirements.txt

RUN rm -rf /tmp/* /var/tmp/*
RUN rm -rf /app/.git/*

RUN python -m v1.modules.download_model

RUN python -m pytest -v

CMD exec gunicorn --bind :$PORT_SERVER --workers $WORKERS --threads $THREADS --timeout $TIMEOUT --preload main:app
