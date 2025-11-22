import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO

# Charger ton modèle YOLO (assure-toi que best.pt est dans ton repo)
model = YOLO("models/best.pt")

st.set_page_config(page_title="Détection YOLOv9", layout="wide")

st.title("🚀 Détection Poubelle YOLOv9")

tab1, tab2 = st.tabs(["📷 Détection Image", "🎥 Détection Vidéo"])

# Onglet Image
with tab1:
    st.header("Upload une image")
    img_file = st.file_uploader("Choisis une image", type=["jpg","jpeg","png"])
    if img_file is not None:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = model.predict(img)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Résultat", use_column_width=True)

# Onglet Vidéo
with tab2:
    st.header("Upload une vidéo")
    vid_file = st.file_uploader("Choisis une vidéo", type=["mp4","avi","mov","mkv"])
    if vid_file is not None:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(vid_file.read())

        cap = cv2.VideoCapture(tfile)
        frame_idx = 0
        summary = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:  # traite 1 frame sur 10
                results = model.predict(frame)
                detections = []
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        detections.append(f"{model.names[cls_id]} ({conf:.2f})")
                summary.append({"frame": frame_idx, "detections": detections})
            frame_idx += 1

        cap.release()

        st.subheader("Résumé des détections")
        for s in summary[:50]:
            st.write(f"Frame {s['frame']}: {', '.join(s['detections']) if s['detections'] else 'Aucune détection'}")
