from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import logging
from flask_cors import CORS

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'predicciones')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

modelo_tf = load_model('models/modelo_final.keras')
modelo_yolo = YOLO('runs/cana_detect/weights/best.pt')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analizar', methods=['POST'])
def analizar():
    logging.info("[INFO] Recibida solicitud POST /analizar")

    if 'imagen' not in request.files:
        return jsonify({'error': 'No se envió ninguna imagen'}), 400

    imagen = request.files['imagen']
    if imagen.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400

    try:
        nombre_id = uuid.uuid4().hex
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{nombre_id}_temp.jpg')
        imagen.save(temp_path)
    except Exception as e:
        logging.error(f"[ERROR] Fallo al guardar imagen: {e}")
        return jsonify({'error': 'Fallo al guardar imagen'}), 500

    try:
        clase, confianza, probabilidad = image_verification(temp_path)
    except Exception as e:
        logging.error(f"[ERROR] Fallo en verificación de imagen: {e}")
        return jsonify({'error': 'Error al analizar la imagen'}), 500

    es_cana = clase == "cana" and confianza >= 0.85

    if not es_cana:
        os.remove(temp_path)
        return jsonify({'es_cana': False, 'probabilidad': round(probabilidad, 4)})
    
    limpiar_carpetas_predicciones()

    try:
        resultados = modelo_yolo(temp_path)[0]
    except Exception as e:
        logging.error(f"[ERROR] Error al ejecutar YOLO: {e}")
        return jsonify({'error': 'Error al analizar con YOLO'}), 500

    img = Image.open(temp_path).convert("RGB")

    for i, box in enumerate(resultados.boxes.data.tolist()):
        x1, y1, x2, y2, score, cls = box
        cls = int(cls)
        label = modelo_yolo.names[cls]
        box_coords = [int(x1), int(y1), int(x2), int(y2)]

        region = img.crop(box_coords)

        carpeta_clase = os.path.join(app.config['UPLOAD_FOLDER'], label)
        os.makedirs(carpeta_clase, exist_ok=True)

        ruta_guardado = os.path.join(carpeta_clase, f"{nombre_id}_{label}_{i}.jpg")
        region.save(ruta_guardado)

    def obtener_rutas_imagenes(carpeta):
        carpeta_absoluta = os.path.join(app.config['UPLOAD_FOLDER'], carpeta)
        if not os.path.exists(carpeta_absoluta):
            return []
        archivos = os.listdir(carpeta_absoluta)
        return [f"/static/predicciones/{carpeta}/{archivo}" for archivo in archivos if archivo.lower().endswith(('.jpg', '.png', '.jpeg'))]

    return jsonify({
        'es_cana': True,
        'probabilidad': f"{(1 - probabilidad) * 100:.2f}%",
        'cana_imgs': obtener_rutas_imagenes('cana'),
        'nudo_imgs': obtener_rutas_imagenes('nudo'),
        'entrenudo_imgs': obtener_rutas_imagenes('entrenudo'),
        'num_canas': len(obtener_rutas_imagenes('cana')),
        'num_nudos': len(obtener_rutas_imagenes('nudo')),
        'num_entrenudos': len(obtener_rutas_imagenes('entrenudo')),
    })

def limpiar_carpetas_predicciones():
    carpetas = ['cana', 'nudo', 'entrenudo']
    for carpeta in carpetas:
        ruta = os.path.join(app.config['UPLOAD_FOLDER'], carpeta)
        if os.path.exists(ruta):
            for archivo in os.listdir(ruta):
                archivo_path = os.path.join(ruta, archivo)
                if os.path.isfile(archivo_path):
                    os.remove(archivo_path)

def image_verification(imagen_path):
    clases = ["cana", "no_cana"]
    input_shape = modelo_tf.input_shape[1:3]

    img = image.load_img(imagen_path, target_size=input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prob = float(modelo_tf.predict(img_array)[0][0])
    clase = clases[0] if prob < 0.4 else clases[1]
    confianza = 1 - prob if prob < 0.5 else prob
    return clase, confianza, prob

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
