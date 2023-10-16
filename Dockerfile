FROM docker.io/python:3.11.5-slim

RUN pip install pipenv async-timeout

COPY Pipfile .
COPY Pipfile.lock .

RUN pipenv install --system --deploy --ignore-pipfile

COPY ./app /app
COPY main.py /

EXPOSE 7001

WORKDIR /

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7001", "--forwarded-allow-ips", "'*'"]
