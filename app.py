from flask import Flask

# Crear una instancia de la aplicación Flask
app = Flask(__name__)

# Definir la ruta de la raíz
@app.route('/')
def hello_world():
    return '¡Hola, mundo!'

# Comprobar si este archivo se está ejecutando directamente
if __name__ == '__main__':
    # Ejecutar la aplicación
    app.run(host='0.0.0.0', port=5000)
