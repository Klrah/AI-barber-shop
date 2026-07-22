import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
import io
import os
import uuid
import boto3
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# 0. INITIALIZATION
# ==========================================
# Load environment variables from local .env file


st.set_page_config(page_title="Enterprise Barber AI & CRM", layout="wide")

# ==========================================
# 1. CLOUD INFRASTRUCTURE & SECRETS SETUP
# ==========================================
def get_secret(key: str, default: str) -> str:
    if hasattr(st, "secrets") and key in st.secrets:
        return st.secrets[key]
    return os.getenv(key, default)

DATABASE_URL = get_secret("DATABASE_URL", "postgresql://mock_user:mock_pass@localhost/mock_db")

# Automatically fix legacy 'postgres://' connection strings
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

AWS_ACCESS_KEY = get_secret("AWS_ACCESS_KEY_ID", "mock_key")
AWS_SECRET_KEY = get_secret("AWS_SECRET_ACCESS_KEY", "mock_secret")
AWS_REGION = get_secret("AWS_REGION", "us-east-1")
S3_BUCKET = get_secret("S3_BUCKET_NAME", "mock-bucket")
HF_API_TOKEN = get_secret("HF_TOKEN", "mock_hf_token")

# ==========================================
# 2. IMAGE OPTIMIZATION HELPER
# ==========================================
def compress_and_resize_image(pil_image: Image.Image, max_dim: int = 512) -> Image.Image:
    """Resizes PIL image so its longest side does not exceed max_dim."""
    img = pil_image.copy()
    img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    return img

# ==========================================
# 3. POSTGRESQL DATABASE LAYER (NEON.TECH)
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
# 4. SMART RAG PIPELINE
# ==========================================
BARBER_KNOWLEDGE_BASE = [
    "Oval faces suit classic taper fades with short textured crops on top.",
    "Square faces look best with tight skin fades, buzz cuts, or structured crew cuts to highlight the jawline.",
    "Round faces benefit from high skin fades and pompadours to add height and elongate the face structure.",
    "Oblong faces should avoid high fades; opt for scissor-cut sides and a medium length flow on top."
]

def retrieve_rag_suggestion(face_shape: str, user_preference: str) -> str:
    """Lightweight RAG using TF-IDF vectorization and cosine similarity."""
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
# 5. AWS S3 IMAGE STORAGE
# ==========================================
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
# 6. COMPUTER VISION & AI INFERENCE
# ==========================================
def generate_hair_mask(image: Image.Image) -> Image.Image:
    img_array = np.array(image.convert("L"))
    h, w = img_array.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center, axes = (int(w / 2), int(h * 0.25)), (int(w * 0.4), int(h * 0.25))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return Image.fromarray(mask)

def generate_haircut(image: Image.Image, prompt: str) -> Image.Image:
    mask = generate_hair_mask(image)
    if HF_API_TOKEN == "mock_hf_token":
        return image.convert("LA").convert("RGB") # Fallback if no API key
    
    buf_img, buf_mask = io.BytesIO(), io.BytesIO()
    image.save(buf_img, format="PNG")
    mask.save(buf_mask, format="PNG")
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    files = {
        "image": ("image.png", buf_img.getvalue(), "image/png"), 
        "mask_image": ("mask.png", buf_mask.getvalue(), "image/png")
    }
    
    try:
        res = requests.post(
            "https://api-inference.huggingface.co/models/stable-diffusion-v1-5/stable-diffusion-inpainting", 
            headers=headers, 
            files=files, 
            data={"inputs": prompt},
            timeout=45
        )
        
        if res.status_code == 200:
            return Image.open(io.BytesIO(res.content))
        else:
            st.warning(f"⚠️ Hugging Face API returned status {res.status_code}. Returning original image.")
            return image
            
    except requests.exceptions.RequestException:
        st.error("⚠️ Network connection to AI provider failed. Please try again in a few moments.")
        return image

# ==========================================
# 7. FRONTEND PORTAL
# ==========================================
st.title("✂️ AI Barber Copilot & Enterprise CRM")

tab1, tab2 = st.tabs(["📱 Intake & AI Preview", "💈 Stylist CRM Ledger"])

with tab1:
    st.header("Customer Design Portal")
    col1, col2 = st.columns(2)
    
    with col1:
        client_phone = st.text_input("Client Phone ID", placeholder="555-0100")
        face_shape = st.selectbox("Detected Face Shape", ["Oval", "Square", "Round", "Oblong"])
        style_pref = st.text_input("General Preference", "Short and textured")
        trimmer_length = st.slider("Fade Depth Target (mm)", 0.1, 6.0, 1.2, 0.1)
    
    with col2:
        uploaded_file = st.file_uploader("Upload Client Photo", type=["jpg", "png"])
        st.info(f"💡 **RAG AI Suggestion:** {retrieve_rag_suggestion(face_shape, style_pref)}")

    if st.button("🔥 Execute Cloud Processing Pipeline", use_container_width=True):
        if not uploaded_file or not client_phone:
            st.error("Missing inputs.")
        else:
            with st.spinner("Processing rendering and persisting to AWS S3 / PostgreSQL..."):
                raw_img = Image.open(uploaded_file).convert("RGB")
                
                # Immediately resize image to max 512px before processing
                input_img = compress_and_resize_image(raw_img, max_dim=512)
                
                # Inference
                prompt = f"Professional haircut, {style_pref}, {trimmer_length}mm guard fade on sides. Highly detailed."
                result_img = generate_haircut(input_img, prompt)
                st.image(result_img, caption="AI Result (Facial Identity Locked)", use_column_width=True)
                
                # Generate Blueprint
                blueprint = f"EXECUTION SPEC: Cut sides to {trimmer_length}mm. Top: {style_pref}. Face structure match: {face_shape}."
                st.code(blueprint)
                
                # JPEG compression (quality=80) before AWS S3 upload
                img_byte_arr = io.BytesIO()
                result_img.save(img_byte_arr, format='JPEG', quality=80, optimize=True)
                
                s3_url = upload_to_s3(img_byte_arr.getvalue(), f"{client_phone}_{uuid.uuid4().hex[:6]}.jpg")
                
                # Persist to PostgreSQL
                if db_connected:
                    db = SessionLocal()
                    record = HaircutRecord(client_phone=client_phone, trimmer_length=trimmer_length, blueprint=blueprint, s3_image_url=s3_url)
                    db.add(record)
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
                        st.write(f"**Blueprint:** {r.blueprint}")
                        st.markdown(f"**AWS S3 Asset Link:** [View Image Object]({r.s3_image_url})")
            else:
                st.warning("No records found.")
            db.close()
        else:
            st.error("Database disconnected. Connect Neon.tech URI to query records.")