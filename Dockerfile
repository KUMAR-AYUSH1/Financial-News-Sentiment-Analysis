FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords

COPY api.py .
COPY app.py .
COPY model.pkl .
COPY LabelEncoder.pkl .
COPY tfidf.pkl .
COPY selector.pkl .
EXPOSE 8000
EXPOSE 8501

CMD sh -c "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"

