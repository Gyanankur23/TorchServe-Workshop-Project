import streamlit as st
import requests
from PIL import Image
import io
import time
from typing import Dict, Optional

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="TorchServe Vision AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR PROFESSIONAL STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem 1rem;
    }
    
    /* Custom card styling */
    .stCard {
        border-radius: 12px;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Upload section styling */
    [data-testid="stFileUploader"] {
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed rgba(255, 255, 255, 0.3);
        background: rgba(255, 255, 255, 0.02);
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Metric card customization */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        border-radius: 10px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.02);
    }
    
    /* Custom divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }
    
    /* Image container */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
TORCHSERVE_URL = "http://localhost:8080/predictions/resnet"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FORMATS = ["jpg", "jpeg", "png", "webp"]
REQUEST_TIMEOUT = 30  # seconds

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_image(uploaded_file) -> tuple[bool, Optional[str]]:
    """Validate uploaded image file."""
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, "File size exceeds 10MB limit."
    
    try:
        image = Image.open(uploaded_file)
        image.verify()  # Verify it's a valid image
        uploaded_file.seek(0)  # Reset file pointer
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}"

def preprocess_image(image: Image.Image) -> bytes:
    """Convert PIL Image to bytes for TorchServe."""
    img_bytes = io.BytesIO()
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image.save(img_bytes, format='JPEG', quality=95)
    return img_bytes.getvalue()

def get_predictions(image_bytes: bytes) -> tuple[bool, Optional[Dict], Optional[str]]:
    """Send image to TorchServe and get predictions."""
    try:
        response = requests.post(
            TORCHSERVE_URL,
            data=image_bytes,
            timeout=REQUEST_TIMEOUT,
            headers={'Content-Type': 'application/octet-stream'}
        )
        
        if response.status_code == 200:
            predictions = response.json()
            return True, predictions, None
        else:
            error_msg = f"Server returned status {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail.get('message', response.text)}"
            except:
                error_msg += f": {response.text[:200]}"
            return False, None, error_msg
            
    except requests.exceptions.ConnectionError:
        return False, None, "Cannot connect to TorchServe. Please ensure Docker container is running on port 8080."
    except requests.exceptions.Timeout:
        return False, None, f"Request timed out after {REQUEST_TIMEOUT} seconds."
    except Exception as e:
        return False, None, f"Unexpected error: {str(e)}"

def format_confidence(value: float) -> str:
    """Format confidence score as percentage."""
    return f"{value * 100:.2f}%"

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.image("https://raw.githubusercontent.com/pytorch/serve/master/docs/images/logo.png", width=200)
    
    st.markdown("### 🔬 About This App")
    st.markdown("""
    This application demonstrates real-time image classification using:
    - **ResNet-18** deep learning model
    - **TorchServe** inference server
    - **Docker** containerization
    """)
    
    st.divider()
    
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. **Upload** an image (JPG, PNG, WEBP)
    2. **Click** the Analyze button
    3. **Review** AI predictions with confidence scores
    """)
    
    st.divider()
    
    st.markdown("### ⚙️ Configuration")
    with st.expander("Server Settings"):
        st.code(f"Endpoint: {TORCHSERVE_URL}")
        st.code(f"Timeout: {REQUEST_TIMEOUT}s")
        st.code(f"Max Size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    st.divider()
    
    # Connection status check
    st.markdown("### 🔌 Status")
    try:
        health_check = requests.get("http://localhost:8080/ping", timeout=2)
        if health_check.status_code == 200:
            st.success("✅ TorchServe Online")
        else:
            st.warning("⚠️ Server Responding (Non-200)")
    except:
        st.error("❌ TorchServe Offline")

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header section
st.markdown("""
    <div style='text-align: center; padding: 1rem 0 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🔬 AI Vision Classifier</h1>
        <p style='font-size: 1.2rem; color: rgba(255,255,255,0.7);'>
            Powered by ResNet-18 & TorchServe
        </p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'analysis_time' not in st.session_state:
    st.session_state.analysis_time = None

# ============================================================================
# TWO-COLUMN LAYOUT
# ============================================================================
col1, col2 = st.columns([1.2, 1.8], gap="large")

# --- LEFT COLUMN: IMAGE INPUT ---
with col1:
    st.markdown("### 📤 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=ALLOWED_FORMATS,
        help=f"Supported formats: {', '.join(ALLOWED_FORMATS).upper()} | Max size: {MAX_FILE_SIZE // (1024*1024)}MB"
    )
    
    if uploaded_file:
        # Validate image
        is_valid, error_msg = validate_image(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"📁 {uploaded_file.name}", use_container_width=True)
            
            # Image metadata
            with st.expander("📊 Image Information"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Format", image.format)
                    st.metric("Width", f"{image.width}px")
                with col_b:
                    st.metric("Mode", image.mode)
                    st.metric("Height", f"{image.height}px")
                
                st.caption(f"File size: {uploaded_file.size / 1024:.2f} KB")

# --- RIGHT COLUMN: PREDICTIONS ---
with col2:
    st.markdown("### 🎯 Analysis Results")
    
    if not uploaded_file:
        st.info("👈 Upload an image to begin analysis")
        st.markdown("""
            <div style='text-align: center; padding: 3rem 0;'>
                <p style='font-size: 4rem; margin: 0;'>🖼️</p>
                <p style='color: rgba(255,255,255,0.5); margin-top: 1rem;'>
                    Waiting for image upload...
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    elif not is_valid:
        st.warning("⚠️ Please upload a valid image file")
    
    else:
        # Analyze button
        analyze_btn = st.button(
            "🔍 Analyze Image",
            type="primary",
            use_container_width=True
        )
        
        if analyze_btn:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Preprocessing
            status_text.text("⏳ Preprocessing image...")
            progress_bar.progress(25)
            time.sleep(0.3)
            
            image_bytes = preprocess_image(image)
            
            # Step 2: Sending request
            status_text.text("📡 Sending to TorchServe...")
            progress_bar.progress(50)
            
            start_time = time.time()
            success, predictions, error_msg = get_predictions(image_bytes)
            analysis_time = time.time() - start_time
            
            # Step 3: Processing response
            status_text.text("🔄 Processing results...")
            progress_bar.progress(75)
            time.sleep(0.2)
            
            progress_bar.progress(100)
            time.sleep(0.3)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if success:
                st.session_state.predictions = predictions
                st.session_state.analysis_time = analysis_time
                st.success(f"✅ Analysis complete in {analysis_time:.2f}s")
            else:
                st.error(f"❌ {error_msg}")
                st.session_state.predictions = None
        
        # Display predictions if available
        if st.session_state.predictions:
            predictions = st.session_state.predictions
            
            st.divider()
            
            # Top prediction highlight
            best_label = list(predictions.keys())[0]
            best_score = list(predictions.values())[0]
            
            # Big metric card
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric(
                    label="🏆 Top Prediction",
                    value=best_label.replace('_', ' ').title(),
                    help="Most confident classification"
                )
            with col_metric2:
                st.metric(
                    label="📊 Confidence",
                    value=format_confidence(best_score),
                    delta="High" if best_score > 0.7 else "Medium" if best_score > 0.4 else "Low",
                    delta_color="normal" if best_score > 0.7 else "off"
                )
            
            st.divider()
            
            # Detailed predictions
            st.markdown("#### 📈 All Predictions")
            
            for idx, (label, confidence) in enumerate(predictions.items(), 1):
                score_pct = confidence * 100
                
                # Create custom layout for each prediction
                col_label, col_bar, col_value = st.columns([2, 5, 1])
                
                with col_label:
                    # Add medal emoji for top 3
                    emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else "▪️"
                    st.markdown(f"{emoji} **{label.replace('_', ' ').title()}**")
                
                with col_bar:
                    st.progress(min(int(score_pct), 100))
                
                with col_value:
                    st.markdown(f"**{score_pct:.1f}%**")
            
            # Additional info
            st.divider()
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.caption(f"⏱️ Analysis time: {st.session_state.analysis_time:.3f}s")
            with col_info2:
                st.caption(f"📦 Results: {len(predictions)} classes")
            
            # Raw JSON expander
            with st.expander("🔍 View Raw JSON Response"):
                st.json(predictions)
            
            # Download results button
            if st.download_button(
                label="💾 Download Results",
                data=str(predictions),
                file_name=f"predictions_{uploaded_file.name}.json",
                mime="application/json",
                use_container_width=True
            ):
                st.success("✅ Results downloaded!")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
    <div style='text-align: center; color: rgba(255,255,255,0.5); padding: 1rem 0;'>
        <p>Built with Streamlit | Powered by PyTorch & TorchServe</p>
    </div>
""", unsafe_allow_html=True)
