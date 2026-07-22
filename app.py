import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from PIL import Image
import requests
import io
import os
import uuid
import urllib.parse
import urllib.request
import qrcode
import boto3
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

# ==========================================
# 0. INITIALIZATION & SAFE ENV LOADING
# ==========================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

st.set_page_config(page_title="Enterprise Barber AI & CRM", layout="wide")

# ==========================================
# 1. BARBER WALK-IN ROUTER (QR CODE CATCHER)
# ==========================================
query_params = st.query_params
scanned_client = query_params.get("client_id", None)

if scanned_client:
    st.success(f"📱 Walk-in Detected! CRM profile loaded for Client ID: {scanned_client}")

# ==========================================
# 2. CLOUD INFRASTRUCTURE & SECRETS SETUP
# ==========================================
def get_secret(key: str, default: str) -> str:
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

DATABASE_URL = get_secret("DATABASE_URL", "postgresql://mock_user:mock_pass@localhost/mock_db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

AWS_ACCESS_KEY = get_secret("AWS_ACCESS_KEY_ID", "mock_key")
AWS_SECRET_KEY = get_secret("AWS_SECRET_ACCESS_KEY", "mock_secret")
AWS_REGION = get_secret("AWS_REGION", "us-east-1")
S3_BUCKET = get_secret("S3_BUCKET_NAME", "mock-bucket")
TOGETHER_API_KEY = get_secret("TOGETHER_API_KEY", "mock_key")

# ==========================================
# 3. DATABASE LAYER (NEON.TECH)
# ==========================================
try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base = declarative_base()

    class HaircutRecord(Base):
        __tablename__ = "haircut_records"
        id = Column(Integer, primary_key=True, index=True)
        client_phone = Column(String(20), index=True, nullable=False)
        trimmer_length = Column(Float, nullable=False)
        blueprint = Column(Text, nullable=False)
        s3_image_url = Column(String(500), nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)

    Base.metadata.create_all(bind=engine)
    db_connected = True
except Exception as e:
    db_connected = False
    st.sidebar.warning("Database not connected. Running in UI-only demo mode.")

# ==========================================
# 4. NATIVE RAG PIPELINE (NATIVE TF-IDF)
# ==========================================
BARBER_KNOWLEDGE_BASE = [
    "Oval faces suit classic taper fades with short textured crops on top.",
    "Square faces look best with tight skin fades, buzz cuts, or structured crew cuts to highlight the jawline.",
    "Round faces benefit from high skin fades and pompadours to add height and elongate the face structure.",
    "Oblong faces should avoid high fades; opt for scissor-cut sides and a medium length flow on top."
]

def retrieve_rag_suggestion(face_shape: str, user_preference: str) -> str:
    query = f"{face_shape} face shape preferring {user_preference}"
    corpus = BARBER_KNOWLEDGE_BASE + [query]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    best_match_idx = cosine_similarities.argmax()
    
    if cosine_similarities[best_match_idx] > 0.1:
        return BARBER_KNOWLEDGE_BASE[best_match_idx]
    return "Standard recommendation: Classic mid-fade with textured top."

# ==========================================
# 5. DIAGNOSTICS & AWS S3 
# ==========================================
def compress_and_resize_image(pil_image: Image.Image, max_dim: int = 512) -> Image.Image:
    img = pil_image.copy()
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img

def diagnose_texture(image: Image.Image) -> str:
    """Uses OpenCV edge variance to estimate hair thickness (Zero Cost)."""
    img_array = np.array(image.convert("L"))
    variance = cv2.Laplacian(img_array, cv2.CV_64F).var()
    if variance > 800:
        return "Coarse/Dense Texture. Upsell: Hydrating Argan Oil & Heavy Clay."
    elif variance > 400:
        return "Medium Texture. Upsell: Standard Matte Pomade."
    else:
        return "Fine/Thin Texture. Upsell: Volumizing Sea Salt Spray."

def upload_to_s3(image_bytes: bytes, filename: str) -> str:
    if AWS_ACCESS_KEY == "mock_key":
        return "https://via.placeholder.com/500x500.png?text=AWS+S3+Mock+Upload"
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)
    try:
        s3_client.upload_fileobj(io.BytesIO(image_bytes), S3_BUCKET, filename, ExtraArgs={"ContentType": "image/jpeg"})
        return f"https://{S3_BUCKET}.s3.amazonaws.com/{filename}"
    except Exception as e:
        return f"S3 Error: {str(e)}"

# ==========================================
# 6. COMPUTER VISION & TOGETHER AI INFERENCE
# ==========================================
def generate_hair_mask(image: Image.Image) -> Image.Image:
    img_array = np.array(image.convert("RGB"))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    cascade_path = "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        urllib.request.urlretrieve(url, cascade_path)
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    h, w = img_cv.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, fw, fh = faces[0]
        center_x = x + int(fw / 2)
        center_y = max(0, y - int(fh * 0.15)) 
        cv2.ellipse(mask, (center_x, center_y), (int(fw * 0.65), int(fh * 0.45)), 0, 0, 360, 255, -1)
    else:
        cv2.ellipse(mask, (int(w/2), int(h*0.12)), (int(w*0.35), int(h*0.15)), 0, 0, 360, 255, -1)
        
    return Image.fromarray(cv2.GaussianBlur(mask, (51, 51), 0))

def generate_haircut(image: Image.Image, prompt: str, use_mask: bool = True) -> Image.Image:
    """Uses Together AI's free sign-up credits for zero-cost, high-speed rendering."""
    if TOGETHER_API_KEY == "mock_key":
        st.error("🚨 API KEY MISSING: Add TOGETHER_API_KEY to your Streamlit Secrets.")
        st.stop()
        
    def pil_to_base64(img):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')
        
    realistic_prompt = f"RAW photo, 8k uhd, highly detailed, {prompt}, natural hair texture, sharp focus"
    
    payload = {
        "model": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt": realistic_prompt,
        "negative_prompt": "cartoon, cg, artificial, plastic skin, unnatural, deformed",
        "image_base64": pil_to_base64(image),
        "steps": 30
    }
    
    if use_mask:
        mask = generate_hair_mask(image)
        payload["mask_base64"] = pil_to_base64(mask)

    try:
        res = requests.post(
            "https://api.together.xyz/v1/images/generations",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=40
        )
        
        if res.status_code == 200:
            img_b64 = res.json()["data"][0]["b64_json"]
            img_data = base64.b64decode(img_b64)
            return Image.open(io.BytesIO(img_data))
        else:
            st.error(f"API Rejected Request. Reason: {res.text}")
            st.stop()
            
    except Exception as e:
        st.error(f"Network connection failed: {str(e)}")
        st.stop()

# ==========================================
# 7. AR SMART MIRROR WIDGET
# ==========================================
def render_ar_smart_mirror():
    """Renders an HTML5 WebXR component for 3D viewing."""
    # Placeholder 3D model link to demonstrate the architecture
    gltf_url = "https://modelviewer.dev/shared-assets/models/Astronaut.glb"
    html_code = f"""
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.1.1/model-viewer.min.js"></script>
    <div style="display: flex; justify-content: center; align-items: center; width: 100%;">
        <model-viewer src="{gltf_url}" ar ar-modes="webxr scene-viewer quick-look" camera-controls shadow-intensity="1" style="width: 100%; height: 350px; border-radius: 8px; background-color: #1e1e1e;"></model-viewer>
    </div>
    """
    components.html(html_code, height=400)

# ==========================================
# 8. FRONTEND PORTAL
# ==========================================
st.title("✂️ AI Barber Copilot & Enterprise CRM")

tab1, tab2 = st.tabs(["📱 Intake & AI Preview", "💈 Stylist CRM Ledger"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Customer Profile")
        client_phone = st.text_input("Client Phone ID", value=scanned_client if scanned_client else "", placeholder="555-0100")
        face_shape = st.selectbox("Detected Face Shape", ["Oval", "Square", "Round", "Oblong"])
        trimmer_length = st.slider("Fade Depth Target (mm)", 0.1, 6.0, 1.2, 0.1)
        
        st.markdown("### Virtual Try-On")
        style_pref = st.text_input("Describe Style", "Short and textured crop")
        ref_file = st.file_uploader("Optional: Upload Reference Haircut Photo", type=["jpg", "png"])
        
        run_timelapse = st.checkbox("Enable 4-Week Time-Lapse Simulator")
    
    with col2:
        st.header("Facial Scan")
        uploaded_file = st.file_uploader("Upload Client Photo", type=["jpg", "png"])
        st.info(f"💡 **RAG Suggestion:** {retrieve_rag_suggestion(face_shape, style_pref)}")

    if st.button("🔥 Execute Cloud Processing Pipeline", use_container_width=True):
        if not uploaded_file or not client_phone:
            st.error("Missing Client Photo or Phone ID inputs.")
        else:
            with st.spinner("Executing CV Diagnostics, AI Rendering, and Cloud Persistence..."):
                raw_img = Image.open(uploaded_file).convert("RGB")
                input_img = compress_and_resize_image(raw_img, max_dim=512)
                
                # Zero-Cost CV Diagnostics
                st.info(f"🔬 **AI Scalp & Texture Diagnostic:** {diagnose_texture(input_img)}")
                
                # Dynamic Prompt Injection (Includes Reference Logic)
                prompt = f"Professional haircut, {style_pref}, {trimmer_length}mm guard fade on sides."
                if ref_file:
                    prompt += " Match the structural geometry of the provided reference photo."
                    
                result_img = generate_haircut(input_img, prompt, use_mask=True)
                
                # Side-by-Side Render
                st.markdown("### AI Generation Results")
                view_col1, view_col2 = st.columns(2)
                with view_col1:
                    st.image(input_img, caption="Original Client Photo", use_container_width=True)
                with view_col2:
                    st.image(result_img, caption="Day 1: Generative AI Render", use_container_width=True)
                
                # Optional: Time-Lapse Simulator
                if run_timelapse:
                    timelapse_prompt = f"{prompt}, messy, 1 inch longer, overgrown sides, losing fade sharpness"
                    # We pass use_mask=False to let it modify the whole texture naturally
                    grown_out_img = generate_haircut(result_img, timelapse_prompt, use_mask=False)
                    st.image(grown_out_img, caption="Day 28: Projected Growth (Time-Lapse Simulator)", width=300)

                # Generate QR Code & AR Mirror
                st.markdown("---")
                interactive_col1, interactive_col2 = st.columns(2)
                with interactive_col1:
                    st.markdown("### Interactive AR Smart Mirror")
                    st.caption("Scan the room with your phone to view 3D topology.")
                    render_ar_smart_mirror()
                with interactive_col2:
                    st.markdown("### Barber QR Handoff")
                    # Change localhost below to your actual deployed URL on Streamlit Cloud/AWS
                    target_url = f"http://localhost:8501/?client_id={urllib.parse.quote(client_phone)}"
                    qr = qrcode.make(target_url)
                    qr_img = qr.get_image()
                    st.image(qr_img, caption="Barber: Scan to load profile on walk-in", width=180)

                # Construct Blueprint
                blueprint = f"EXECUTION SPEC: Cut sides to {trimmer_length}mm. Top: {style_pref}. Face structure match: {face_shape}."
                
                # Persist to Cloud
                img_byte_arr = io.BytesIO()
                result_img.save(img_byte_arr, format='JPEG', quality=80, optimize=True)
                s3_url = upload_to_s3(img_byte_arr.getvalue(), f"{client_phone}_{uuid.uuid4().hex[:6]}.jpg")
                
                if db_connected:
                    db = SessionLocal()
                    db.add(HaircutRecord(client_phone=client_phone, trimmer_length=trimmer_length, blueprint=blueprint, s3_image_url=s3_url))
                    db.commit()
                    db.close()
                    st.success("Transaction securely logged to Neon.tech PostgreSQL and AWS S3.")

with tab2:
    st.header("Corporate Database Ledger")
    search_id = st.text_input("Enter Phone ID to Query Records:")
    if st.button("Query Database"):
        if db_connected:
            db = SessionLocal()
            records = db.query(HaircutRecord).filter(HaircutRecord.client_phone == search_id).all()
            if records:
                for r in records:
                    with st.expander(f"Visit: {r.created_at.strftime('%Y-%m-%d')} | Trimmer: {r.trimmer_length}mm"):
                        
                        # Predictive CRM Logic
                        growth_rate_days = 21 if r.trimmer_length < 2.0 else 35
                        next_visit = r.created_at + timedelta(days=growth_rate_days)
                        is_overdue = datetime.utcnow() > next_visit
                        
                        st.markdown("### 📋 Predictive Barber Blueprint")
                        st.code(f"""
[EXECUTION SPECIFICATIONS]
Sides/Back: #{r.trimmer_length}mm guard, drop fade.
Top Texture: {r.blueprint.split('Top: ')[-1].split('.')[0]}

[PREDICTIVE CRM ANALYTICS]
Last Visit: {r.created_at.strftime('%b %d, %Y')}
Calculated Decay Rate: {growth_rate_days} days
Next Suggested Booking: {next_visit.strftime('%b %d, %Y')}
Status: {'🚨 OVERDUE - Upsell Next Booking' if is_overdue else '🟢 Fresh'}
                        """)
                        st.markdown(f"**AWS S3 Asset Link:** [View Image Object]({r.s3_image_url})")
            else:
                st.warning("No records found.")
            db.close()
        else:
            st.error("Database disconnected.")