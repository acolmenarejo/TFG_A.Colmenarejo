FROM python:3.8

WORKDIR /TFG
COPY . .

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

CMD  python3 interfaz.py