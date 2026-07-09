import re
import pickle
import joblib
import pandas as pd
import uvicorn
import nltk
from fastapi import FastAPI
from pydantic import BaseModel
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize FastAPI app
app = FastAPI()

# Load models and transformers
le = joblib.load('LabelEncoder.pkl')
model = joblib.load('model.pkl')
tfidf = joblib.load('tfidf.pkl')
selector = joblib.load('selector.pkl')

with open('explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)

# Define request schema (Avoid using Python reserved keyword 'text' as class name)
class TextInput(BaseModel):
    text: str

def preprocessing(text: str) -> str:
    ps = PorterStemmer()
    text = text.lower()
    text = re.sub('[0-9]+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    
    stop_words = set(stopwords.words('english'))
    text = [ps.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(text)

@app.get('/')
async def about():
    return {'message': 'Financial News sentiment analysis'}

@app.post('/predict')
async def predict(data: TextInput):
    # 1. Text Preprocessing & Vectorization
    processed_text = preprocessing(data.text)
    vectorized_text = tfidf.transform([processed_text])
    selected_features = selector.transform(vectorized_text)
    text_dense = selected_features.toarray()
    
    # 2. Model Prediction
    prediction = model.predict(selected_features)[0]
    prob = model.predict_proba(selected_features)[0]
    
    # 3. SHAP Explanation
    shap_values = explainer(text_dense, max_evals=2185)
    
    # Fix 3D vs 2D SHAP array layout depending on your SHAP explainer type
    if len(shap_values.values.shape) == 3:
        # Layout: [sample_index, feature_index, class_index]
        scores = shap_values.values[0, :, prediction]
    else:
        # Layout: [sample_index, feature_index] (for binary/TreeExplainers sometimes)
        scores = shap_values.values[0, :]

    # 4. Feature Extraction
    present_word_indices = text_dense[0] > 0
    all_words = tfidf.get_feature_names_out()
    selected_words = all_words[selector.get_support()]
    
    words_in_text = selected_words[present_word_indices]
    scores_in_text = scores[present_word_indices]
    
    # 5. Format DataFrame
    df_words_present = pd.DataFrame({
        'Word': words_in_text,
        'SHAP_Score': scores_in_text
    })
    df_words_present['Abs_Impact'] = df_words_present['SHAP_Score'].abs()
    df_words_present = df_words_present.sort_values(by='Abs_Impact', ascending=False)
    
    # 6. Response Construction
    confidence = float(prob.max())
    final_label = le.inverse_transform([prediction])[0]
    
    # Convert DataFrame to a cleaner dictionary format for API responses instead of a raw string
    shap_importance = df_words_present[['Word', 'SHAP_Score']].to_dict(orient='records')
    
    return {
        'prediction': final_label,
        'confidence': round(confidence * 100, 2),
        'SHAP_word_importance': shap_importance
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
