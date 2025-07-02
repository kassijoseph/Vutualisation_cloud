# app.py
from flask import Flask, render_template, request, redirect, flash, url_for
import os
import requests
import time
import cv2
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.secret_key = 'ma_clé_super_secrète'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 Mo

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL")
ROBOFLOW_VERSION = os.getenv("ROBOFLOW_VERSION")

HEADERS_ASSEMBLY = {
    "authorization": ASSEMBLYAI_API_KEY,
    "content-type": "application/json"
}

def envoyer_audio_assemblyai(audio_path):
    with open(audio_path, 'rb') as f:
        response = requests.post(
            'https://api.assemblyai.com/v2/upload',
            headers={"authorization": ASSEMBLYAI_API_KEY},
            files={'file': f}
        )
    return response.json()['upload_url']

def lancer_transcription(audio_url):
    response = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=HEADERS_ASSEMBLY,
        json={"audio_url": audio_url, "language_code": "fr"}
    )
    return response.json()['id']

def recuperer_transcription(transcript_id):
    endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
    while True:
        response = requests.get(endpoint, headers=HEADERS_ASSEMBLY)
        data = response.json()
        if data['status'] == 'completed':
            return data['text']
        elif data['status'] == 'failed':
            return "❌ Échec de transcription"
        time.sleep(2)

def analyser_image_opencv(image_path):
    try:
        original_name = secure_filename(os.path.basename(image_path))
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Impossible de lire l'image.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        result_filename = f"annotated_{original_name}"
        result_path = os.path.join(STATIC_FOLDER, result_filename)

        saved = cv2.imwrite(result_path, image)
        if not saved:
            raise IOError("Échec de sauvegarde de l'image annotée.")

        return f"✅ {len(faces)} visage(s) détecté(s).", result_filename

    except Exception as e:
        return f"❌ Erreur OpenCV : {e}", None

def analyser_image_roboflow(image_path):
    model_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}&format=image"
    with open(image_path, 'rb') as f:
        response = requests.post(
            model_url,
            files={"file": f},
            data={"name": os.path.basename(image_path)}
        )

    if 'image' in response.headers.get("content-type", ""):
        roboflow_filename = "roboflow_" + secure_filename(os.path.basename(image_path))
        roboflow_path = os.path.join("static", roboflow_filename)
        with open(roboflow_path, 'wb') as f_out:
            f_out.write(response.content)
        return "✅ Image annotée par Roboflow enregistrée.", roboflow_filename

    try:
        data = response.json()
        if "predictions" in data and not data["predictions"]:
            return "❌ Aucun objet détecté par Roboflow.", None
        return f"❌ Erreur Roboflow : {data}", None
    except Exception:
        return "❌ Erreur inconnue Roboflow.", None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("Aucun fichier sélectionné")
        return redirect('/')

    file = request.files['file']
    if file.filename == '':
        flash("Fichier vide")
        return redirect('/')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    extension = filename.lower().split('.')[-1]
    image_opencv = image_roboflow = None

    try:
        if extension in ['mp3', 'wav']:
            flash("🎿 Traitement audio en cours...")
            audio_url = envoyer_audio_assemblyai(filepath)
            transcript_id = lancer_transcription(audio_url)
            texte = recuperer_transcription(transcript_id)
            flash(f"🖍️ Transcription : {texte}")
            return render_template("index.html")

        elif extension in ['jpg', 'jpeg', 'png']:
            flash("🖼️ Analyse OpenCV en cours...")
            resultat_opencv, image_opencv = analyser_image_opencv(filepath)
            flash(resultat_opencv)

            flash("🤖 Analyse Roboflow en cours...")
            resultat_roboflow, image_roboflow = analyser_image_roboflow(filepath)
            flash(resultat_roboflow)

            return render_template("index.html",
                                   image_url_opencv=f"/static/{image_opencv}" if image_opencv else None,
                                   image_url_roboflow=f"/static/{image_roboflow}" if image_roboflow else None)

        else:
            flash("❌ Format non supporté (.mp3, .wav, .jpg, .png)")
            return render_template("index.html")

    except Exception as e:
        flash(f"❌ Erreur : {e}")
        return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
