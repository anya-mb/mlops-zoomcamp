FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

COPY predict_cli.py /app
COPY Pipfile* /app

RUN pip install pipenv
RUN pipenv install --deploy


CMD ["pipenv", "run", "python", "predict_cli.py", "--year", "2023", "--month", "5"]