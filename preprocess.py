import re
import emoji
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()

    # Eliminar menciones, enlaces y caracteres especiales
    text = re.sub(r"(@\S+|http\S+|www\S+|#\S+|\W|\d)", " ", text)
    
    # Eliminar emojis
    text = remove_emojis(text)
    
    # Tokenizar
    words = word_tokenize(text)
    
    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]
    
    # Lematizar
    nlp = spacy.load('es_core_news_sm')
    words = [nlp(word)[0].lemma_ for word in words]

    # Stemming
    stemmer = SnowballStemmer('spanish')
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)
