FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

COPY models/model.pkl /app/models/model.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet omw-1.4

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]