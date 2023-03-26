FROM python:3.10-slim as compiler

WORKDIR /app

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install git -y

COPY ./requirements.txt /app/requirements.txt

RUN pip3 install -r requirements.txt

FROM python:3.10-slim as runner

WORKDIR /app/

COPY --from=compiler /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY . /app/

CMD ["python3", "main.py"]