from flask import Flask, request, jsonify
import joblib  # Asegúrate de tener joblib para cargar tu modelo y vectorizador
import gensim  # Importa gensim para el preprocesamiento

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Cargar el modelo y el vectorizador
model = joblib.load('ruta_a_tu_modelo.pkl')  # Cambia esto por la ruta a tu modelo
vectorizer = joblib.load('ruta_a_tu_vectorizador.pkl')  # Cambia esto por la ruta a tu vectorizador

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

# Comprobar si este archivo se está ejecutando directamente
if __name__ == '__main__':
    # Ejecutar la aplicación
    app.run(host='0.0.0.0', port=5000)
