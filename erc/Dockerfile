FROM python:3.6

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt  -i https://pypi.douban.com/simple/

EXPOSE 10001

CMD ["python", "erc/server.py"]
