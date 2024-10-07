from flask import Flask, request, jsonify
import joblib  # Asegúrate de tener joblib para cargar tu modelo y vectorizador
import gensim  # Importa gensim para el preprocesamiento
from flask_cors import CORS
import csv  # Importa csv para manipular archivos CSV
import os  # Para verificar rutas
import pandas as pd  # Importa pandas para cargar CSVs desde URL
import requests  # Para descargar el CSV desde una URL
from io import StringIO  # Para leer el contenido descargado de la URL
# Crear una instancia de la aplicación Flask
app = Flask(__name__)
CORS(app)

# Cargar el modelo y el vectorizador
model = joblib.load('model.pkl')  # Cambia esto por la ruta a tu modelo
vectorizer = joblib.load('vectorize.pkl')  # Cambia esto por la ruta a tu vectorizador

# Función para cargar stopwords en español
def load_spanish_stopwords():
    stopwords_path = 'spanish'  # Ruta al archivo de stopwords
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as file:
            stop_words = file.read().splitlines()
    except FileNotFoundError:
        print(f"El archivo de stopwords en español no se encontró en la ruta: {stopwords_path}")
        stop_words = []
    return stop_words

# Función de preprocesamiento
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

# Definir la ruta de la raíz
@app.route('/')
def hello_world():
    return '¡Hola, mundo!'

# Ruta para predecir si una noticia es real o falsa
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

# Ruta para insertar una noticia en el CSV
@app.route('/insert_news', methods=['POST'])
def insert_news():
    data = request.json
    title = data.get('title', '')
    text = data.get('text', '')
    fuente = data.get('fuente', '')
    razon = data.get('razon', '')
    fake_new_class = data.get('fake_new_class', 'unknown')

    if not title or not text:
        return jsonify({"error": "Faltan datos. Se requiere título y texto de la noticia."}), 400

    # Insertar la noticia en el CSV
    insertar_fila_csv(title, text, fuente, razon, fake_new_class)

    return jsonify({"message": "Noticia insertada correctamente en el archivo CSV."})

# Función para insertar una fila en el CSV
def insertar_fila_csv(title, text, fuente, razon, fake_new_class):
    archivo_csv = 'noticias_clasificadas.csv'  # Ruta local del archivo CSV

    # Limpiar el texto
    cleaned_news = preprocess(text)
    cleaned_news_joined = " ".join(cleaned_news)

    # Preparar la fila a insertar
    nueva_fila = [title, text.replace('\n', ' '), fuente, razon, fake_new_class]

    # Insertar la fila en el archivo CSV
    with open(archivo_csv, mode='a', newline='', encoding='utf-8') as archivo:
        escritor_csv = csv.writer(archivo)
        escritor_csv.writerow(nueva_fila)

# Ruta para cargar el contenido del CSV alojado en GitHub Pages
@app.route('/load_news', methods=['GET'])
def load_news():
    archivo_csv_url = 'https://luisfre.github.io/Realityhunter/assets/noticias_clasificadas.csv'
    
    try:
        # Descargar el CSV desde la URL
        response = requests.get(archivo_csv_url)
        response.raise_for_status()  # Verificar que la descarga fue exitosa
        content = StringIO(response.text)

        # Cargar el CSV en un DataFrame
        df = pd.read_csv(content, encoding='utf-8')

        # Reemplazar NaN por valores predeterminados
        df.fillna('', inplace=True)

        # Filtrar las columnas deseadas
        news_list = df[['title', 'text', 'fuente', 'razon', 'fake_new_class']].to_dict(orient='records')

        return jsonify(news_list), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Comprobar si este archivo se está ejecutando directamente
if __name__ == '__main__':
    # Ejecutar la aplicación
    app.run(host='0.0.0.0', port=5000)
