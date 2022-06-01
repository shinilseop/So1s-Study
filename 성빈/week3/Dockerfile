FROM python:3.9.13-slim-bullseye

COPY ./ ./django
WORKDIR ./django

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD ["python", "./test/manage.py", "makemigrations"]
CMD ["python", "./test/manage.py", "migrate"]
CMD ["python", "./test/manage.py", "runserver", "0.0.0.0:8080"]