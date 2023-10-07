FROM python:3.8

RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg


COPY requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /app
COPY . /app/

CMD ["streamlit", "run", "home_page.py","--server.port", "8080" ]