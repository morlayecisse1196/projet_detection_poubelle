import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Détection Poubelle YOLOv9",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé pour un design moderne et professionnel
st.markdown("""
<style>
    /* Style général */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* En-tête principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Cartes d'onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 1rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.12);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Zone de téléchargement */
    .uploadedFile {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Boutons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102,126,234,0.4);
    }
    
    /* Cartes de résultats */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        border-left: 4px solid #00acc1;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Statistiques */
    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Charger le modèle YOLOv9
@st.cache_resource
def load_model():
    return YOLO("models/best.pt")

model = load_model()

# Fonction pour redimensionner l'image
def resize_image(image, max_width=800, max_height=600):
    """Redimensionne l'image tout en conservant le ratio d'aspect"""
    h, w = image.shape[:2]
    
    # Calculer le ratio de redimensionnement
    if w > max_width or h > max_height:
        ratio = min(max_width/w, max_height/h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

# Fonction pour dessiner les boxes
def draw_boxes(image, results):
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            
            # Dessiner rectangle et texte
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (102, 126, 234), 3)
            
            # Fond pour le texte
            text = f"{label} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(image, (int(x1), int(y1)-text_h-10), (int(x1)+text_w+10, int(y1)), (102, 126, 234), -1)
            cv2.putText(image, text, (int(x1)+5, int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            detections.append({"label": label, "confidence": conf})
    
    return image, detections

# En-tête principal
st.markdown("""
<div class="main-header">
    <h1>🗑️ Détection Intelligente de Poubelles</h1>
    <p>Système de détection automatique basé sur YOLOv9</p>
</div>
""", unsafe_allow_html=True)

# Onglets
tab1, tab2 = st.tabs(["📷 Détection sur Image", "🎥 Détection sur Vidéo"])

# ONGLET IMAGE
with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Charger une image")
        st.markdown('<div class="info-box">Formats supportés: JPG, JPEG, PNG</div>', unsafe_allow_html=True)
        
        img_file = st.file_uploader("", type=["jpg","jpeg","png"], key="img_uploader")
        
        if img_file is not None:
            # Lire et redimensionner l'image
            file_bytes = np.frombuffer(img_file.read(), np.uint8)
            img_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Afficher taille originale
            h_orig, w_orig = img_original.shape[:2]
            st.info(f"📐 Taille originale: {w_orig}x{h_orig} pixels")
            
            # Redimensionner pour l'affichage
            img = resize_image(img_original.copy(), max_width=600, max_height=500)
            h_new, w_new = img.shape[:2]
            
            if (h_orig != h_new or w_orig != w_new):
                st.success(f"✓ Image redimensionnée à: {w_new}x{h_new} pixels")
            
            # Afficher l'image originale
            st.markdown("**Image originale:**")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        if img_file is not None:
            st.markdown("### 🎯 Résultats de détection")
            
            with st.spinner("🔍 Analyse en cours..."):
                # Détecter sur l'image redimensionnée
                results = model.predict(img)
                img_out, detections = draw_boxes(img.copy(), results)
            
            # Afficher résultat
            st.markdown("**Image avec détections:**")
            st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Statistiques
            if detections:
                st.markdown("### 📊 Statistiques")
                col_stat1, col_stat2 = st.columns(2)
                
                with col_stat1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{len(detections)}</div>
                        <div class="stat-label">Objets détectés</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_stat2:
                    avg_conf = np.mean([d['confidence'] for d in detections])
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{avg_conf:.1%}</div>
                        <div class="stat-label">Confiance moyenne</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Détails des détections
                st.markdown("### 📋 Détails des détections")
                for i, det in enumerate(detections, 1):
                    st.markdown(f"**{i}.** {det['label']} - Confiance: {det['confidence']:.2%}")
            else:
                st.warning("⚠️ Aucun objet détecté dans l'image")

# ONGLET VIDÉO
with tab2:
    st.markdown("### 🎬 Charger une vidéo")
    st.markdown('<div class="info-box">Formats supportés: MP4, AVI, MOV, MKV</div>', unsafe_allow_html=True)
    
    vid_file = st.file_uploader("", type=["mp4","avi","mov","mkv"], key="vid_uploader")
    
    if vid_file is not None:
        tfile = "temp_video.mp4"
        with open(tfile, "wb") as f:
            f.write(vid_file.read())
        
        cap = cv2.VideoCapture(tfile)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        st.info(f"📹 Vidéo: {total_frames} frames @ {fps} FPS")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        frame_idx = 0
        summary = []
        
        with st.spinner("🔄 Traitement de la vidéo en cours..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % 10 == 0:  # Traite 1 frame sur 10
                    # Redimensionner frame pour accélérer la détection
                    frame_resized = resize_image(frame, max_width=640, max_height=480)
                    results = model.predict(frame_resized)
                    
                    detections = []
                    for r in results:
                        for box in r.boxes:
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            detections.append(f"{model.names[cls_id]} ({conf:.2f})")
                    summary.append({"frame": frame_idx, "detections": detections})
                    
                    # Mise à jour progression
                    progress = min(frame_idx / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Traitement: {frame_idx}/{total_frames} frames")
                
                frame_idx += 1
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Traitement terminé!")
        
        # Résumé des détections
        st.markdown("### 📊 Résumé des détections")
        
        # Statistiques globales
        total_detections = sum(len(s['detections']) for s in summary)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{len(summary)}</div>
                <div class="stat-label">Frames analysées</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{total_detections}</div>
                <div class="stat-label">Détections totales</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_per_frame = total_detections / len(summary) if summary else 0
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{avg_per_frame:.1f}</div>
                <div class="stat-label">Moyenne/frame</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Liste des détections
        st.markdown("### 📝 Détails par frame (50 premières)")
        for s in summary[:50]:
            if s['detections']:
                st.markdown(f"**Frame {s['frame']}:** {', '.join(s['detections'])}")
            else:
                st.markdown(f"**Frame {s['frame']}:** *Aucune détection*")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Propulsé par YOLOv9 | Détection en temps réel</p>
</div>
""", unsafe_allow_html=True)