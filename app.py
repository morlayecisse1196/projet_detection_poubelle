# api/app.py
import os
import uuid
import cv2
import time
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO

# Config
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best.pt")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Flask
app = Flask(__name__)
CORS(app, origins="*")  # à restreindre en prod

# Charge YOLO une seule fois
try:
    model = YOLO(MODEL_PATH)
    CLASS_NAMES = model.names  # dict {id: "classname"}
except Exception as e:
    raise RuntimeError(f"Impossible de charger le modèle: {e}")

def yolo_to_json(results):
    """
    Convertit les résultats Ultralytics en JSON friendly:
    [
      {
        "class": "poubelle pleine",
        "confidence": 0.92,
        "bbox": {"x1": 100, "y1": 80, "x2": 220, "y2": 200}
      }, ...
    ]
    """
    detections = []
    if not results or len(results) == 0:
        return detections

    res = results[0]  # une image traitée
    boxes = res.boxes
    if boxes is None:
        return detections

    for b in boxes:
        # xyxy, conf, cls
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf = float(b.conf[0])
        cls_id = int(b.cls[0])
        detections.append({
            "class": CLASS_names_safe(cls_id),
            "confidence": round(conf, 4),
            "bbox": {
                "x1": int(x1), "y1": int(y1),
                "x2": int(x2), "y2": int(y2)
            }
        })
    return detections

def CLASS_names_safe(cls_id):
    # model.names peut être dict ou list selon la version
    if isinstance(CLASS_NAMES, dict):
        return CLASS_NAMES.get(cls_id, str(cls_id))
    elif isinstance(CLASS_NAMES, list):
        return CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
    return str(cls_id)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True, "classes": list(CLASS_NAMES.values()) if isinstance(CLASS_NAMES, dict) else CLASS_NAMES})

@app.route("/predict/image", methods=["POST"])
def predict_image():
    """
    Form-data:
      - file: image (jpg/png)
      - conf (optionnel): seuil de confiance (0-1)
      - imgsz (optionnel): taille d’inférence (ex: 640)
    """
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier 'file' fourni"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nom de fichier vide"}), 400

    # Sauvegarde temporaire
    ext = os.path.splitext(file.filename)[1].lower()
    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    file.save(fpath)

    # Params
    conf = float(request.form.get("conf", 0.25))
    imgsz = int(request.form.get("imgsz", 640))

    # Inference
    try:
        t0 = time.time()
        results = model.predict(
            source=fpath,
            conf=conf,
            imgsz=imgsz,
            verbose=False
        )
        dt = time.time() - t0
        detections = yolo_to_json(results)

        return jsonify({
            "filename": file.filename,
            "inference_time_ms": int(dt * 1000),
            "detections": detections
        })
    except Exception as e:
        return jsonify({"error": f"Erreur inference: {e}"}), 500
    finally:
        # Nettoyage si besoin (désactive si tu veux garder les uploads)
        try:
            os.remove(fpath)
        except Exception:
            pass

@app.route("/predict/video", methods=["POST"])
def predict_video():
    """
    Form-data:
      - file: vidéo (mp4/avi/mov)
      - conf (optionnel): seuil
      - imgsz (optionnel): taille
      - frame_stride (optionnel): traite 1 frame sur N (ex: 5)
      - max_frames (optionnel): limite de frames
    Retourne un résumé des détections par frame (JSON).
    """
    if "file" not in request.files:
        return jsonify({"error": "Aucun fichier 'file' fourni"}), 400

    file = request.files["file"]
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".mp4", ".avi", ".mov", ".mkv"]:
        return jsonify({"error": "Format vidéo non supporté"}), 400

    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    file.save(fpath)

    conf = float(request.form.get("conf", 0.25))
    imgsz = int(request.form.get("imgsz", 640))
    frame_stride = int(request.form.get("frame_stride", 5))
    max_frames = int(request.form.get("max_frames", 200))

    capsumm = []
    try:
        cap = cv2.VideoCapture(fpath)
        if not cap.isOpened():
            return jsonify({"error": "Impossible d’ouvrir la vidéo"}), 400

        frame_idx = 0
        processed = 0
        t0 = time.time()

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_stride == 0:
                # Ultralytics accepte directement des arrays numpy BGR
                results = model.predict(
                    source=frame,
                    conf=conf,
                    imgsz=imgsz,
                    verbose=False
                )
                detections = yolo_to_json(results)
                capsumm.append({
                    "frame_index": frame_idx,
                    "detections": detections
                })
                processed += 1

            frame_idx += 1

        dt = time.time() - t0
        return jsonify({
            "frames_processed": processed,
            "inference_time_ms": int(dt * 1000),
            "summary": capsumm
        })
    except Exception as e:
        return jsonify({"error": f"Erreur vidéo: {e}"}), 500
    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            os.remove(fpath)
        except Exception:
            pass

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
