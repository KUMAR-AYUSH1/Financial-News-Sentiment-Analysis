from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

def preprocessing(text):
    ps = PorterStemmer()
    text = text.lower()
    text = re.sub('[0-9]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('[^a-zA-Z]',' ',text)
    text = [ps.stem(word) for word in text.split() if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text
le = joblib.load('LabelEncoder.pkl')
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')
selector = joblib.load('selector.pkl')

class text(BaseModel):
    text: str

app = FastAPI()

@app.get('/')
async def about():
    return {'message': 'Financial News sentiment analysis'}

@app.post('/predict')
async def predict(data: text):
    processed_text = preprocessing(data.text)
    vectorized_text = tfidf.transform([processed_text])
    selected_features = selector.transform(vectorized_text)
    
    prediction = model.predict(selected_features)
    prob = model.predict_proba(selected_features)
    
    confidence = float(prob.max())
    final_label = le.inverse_transform(prediction)[0]

    
    return {
        'prediction': final_label,
        'confidence': round(confidence*100, 2)
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)