FROM python:3.10.12

USER root
# コンテナ内のユーザを指定

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8

# コンテナの環境変数を指定
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm
ENV PYTHONPATH=/app

RUN apt-get install -y vim less
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# COPY requirements.txt .
# COPY . .

WORKDIR /app

# runするときにマウントするからやらなくてもいい？
# COPY . /app

RUN python -m pip install -r requirements.txt
# RUN python -m pip install jupyterlab

# CMD ["python", "hello.py"]
