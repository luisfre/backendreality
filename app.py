from flask import Flask, request, jsonify
import pickle
import gensim
from nltk.corpus import stopwords
import nltk
import csv
import pandas as pd

from flask_cors import CORS



app = Flask(__name__)
CORS(app)

# Cargar el modelo y el vectorizador
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorize.pkl', 'rb'))

# Función de preprocesamiento
def load_spanish_stopwords():
    stopwords_path = 'nltk_data/corpora/stopwords/spanish'  # Ruta al archivo de stopwords
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as file:
            stop_words = file.read().splitlines()
    except FileNotFoundError:
        print(f"El archivo de stopwords en español no se encontró en la ruta: {stopwords_path}")
        stop_words = []
    return stop_words


def preprocess(text):
    result = []
    stop_words = load_spanish_stopwords()
    stop_words.extend(['según', 'tras', 'cabe', 'bajo', 'durante', 'mediante', 'so', 'toda', 'todas', 'cada', 'me', 
                       'después', 'despues', 'segun', 'solo', 'sido', 'estan', 'lunes', 'martes', 'miércoles', 
                       'jueves', 'viernes'])

    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
    return result

# Función para insertar fila en el CSV
def insertar_fila_csv(title, text, fuente, razon, fake_new_class):
    # Definir el archivo CSV
    archivo_csv = 'C:/Users/lucho/OneDrive/Documentos/Uce/tesisnube/tesis/Reality Hunter/assets/noticias_clasificadas.csv'
    
    # Limpiar el texto
    cleaned_news = preprocess(text)
    cleaned_news_joined = " ".join(cleaned_news)
    
    # Preparar la fila a insertar
    nueva_fila = [title, text.replace('\n', ' '), fuente, razon, fake_new_class]
    
    # Insertar la fila en el archivo CSV
    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as archivo:
        escritor_csv = csv.writer(archivo)
        escritor_csv.writerow(nueva_fila)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    new_news = data.get('news', '')

    # Preprocesar la noticia
    cleaned_news = preprocess(new_news)

    if not cleaned_news:
        return jsonify({"error": "No hay palabras válidas después del preprocesamiento."}), 400

    cleaned_news_joined = " ".join(cleaned_news)
    new_news_dtm = vectorizer.transform([cleaned_news_joined])
    probabilities = model.predict_proba(new_news_dtm)

    return jsonify({
        "real_probability": probabilities[0][0] * 100,
        "fake_probability": probabilities[0][1] * 100
    })

# Nueva ruta para insertar noticias directamente en el CSV
@app.route('/insert_news', methods=['POST'])
def insert_news():
    data = request.json
    title = data.get('title', '')
    text = data.get('text', '')
    fuente= data.get('fuente', '')
    razon= data.get('razon', '')
    fake_new_class = data.get('fake_new_class', 'unknown')

    if not title or not text:
        return jsonify({"error": "Faltan datos. Se requiere título y texto de la noticia."}), 400

    # Insertar la noticia en el CSV
    insertar_fila_csv(title, text, fuente, razon, fake_new_class)

    return jsonify({"message": "Noticia insertada correctamente en el archivo CSV."})

@app.route('/load_news', methods=['GET'])
def load_news():
    archivo_csv = 'C:/Users/lucho/OneDrive/Documentos/Uce/tesisnube/tesis/Reality Hunter/assets/noticias_clasificadas.csv'
    
    try:
        # Cargar el CSV usando pandas
        df = pd.read_csv(archivo_csv, encoding='utf-8')
        
        # Reemplazar NaN por un valor vacío o predeterminado
        df.fillna('', inplace=True)  # Reemplaza NaN con cadenas vacías

        # Filtrar las columnas que queremos devolver
        news_list = df[['title', 'text', 'fuente', 'razon', 'fake_new_class']].to_dict(orient='records')
        

        return jsonify(news_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500  




if __name__ == '__main__':
    app.run(debug=True)