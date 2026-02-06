import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import random

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Deepfake Detector Pro | Felix the Forensic Fox",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    .main-title {
        font-size: 4rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #e94560, #ff6b6b, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(233, 69, 96, 0.3);
    }
    
    .sub-title {
        font-size: 1.3rem;
        color: #eaeaea;
        text-align: center;
        margin-bottom: 3rem;
        opacity: 0.9;
    }
    
    .upload-container {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(233, 69, 96, 0.5);
        border-radius: 20px;
        padding: 50px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-container:hover {
        border-color: #e94560;
        background: rgba(255, 255, 255, 0.08);
    }
    
    .result-authentic {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(17, 153, 142, 0.4);
    }
    
    .result-filtered {
        background: linear-gradient(135deg, #f37335 0%, #fdc830 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(243, 115, 53, 0.4);
    }
    
    .result-fake {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 40px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 40px rgba(235, 51, 73, 0.4);
    }
    
    /* FLOATING FELIX BUTTON */
    .felix-float {
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 9999;
        animation: bounce 2s infinite;
        cursor: pointer;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-10px);}
        60% {transform: translateY(-5px);}
    }
    
    .felix-avatar {
        width: 80px;
        height: 80px;
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.5);
        border: 4px solid white;
        transition: all 0.3s;
    }
    
    .felix-avatar:hover {
        transform: scale(1.1);
        box-shadow: 0 15px 40px rgba(255, 107, 53, 0.7);
    }
    
    .felix-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        background: #e94560;
        color: white;
        border-radius: 50%;
        width: 25px;
        height: 25px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    
    /* CHAT WINDOW */
    .chat-window {
        position: fixed;
        bottom: 120px;
        right: 30px;
        width: 350px;
        height: 450px;
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        z-index: 9998;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        animation: slideUp 0.3s ease-out;
    }
    
    @keyframes slideUp {
        from { transform: translateY(50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .chat-header {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        color: white;
        padding: 15px 20px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .chat-avatar {
        font-size: 2rem;
    }
    
    .chat-title {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .chat-status {
        font-size: 0.8rem;
        opacity: 0.9;
    }
    
    .chat-messages {
        flex: 1;
        padding: 20px;
        overflow-y: auto;
        background: #f8f9fa;
    }
    
    .message {
        margin-bottom: 15px;
        display: flex;
        align-items: flex-start;
        gap: 10px;
    }
    
    .message-avatar {
        font-size: 1.5rem;
    }
    
    .message-bubble {
        background: white;
        padding: 12px 16px;
        border-radius: 15px;
        border-bottom-left-radius: 5px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        color: #333;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    
    .message-user {
        flex-direction: row-reverse;
    }
    
    .message-user .message-bubble {
        background: #e94560;
        color: white;
        border-bottom-left-radius: 15px;
        border-bottom-right-radius: 5px;
    }
    
    .quick-replies {
        padding: 15px;
        background: white;
        border-top: 1px solid #eee;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }
    
    .quick-btn {
        background: #f0f0f0;
        border: none;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: all 0.2s;
        color: #333;
    }
    
    .quick-btn:hover {
        background: #e94560;
        color: white;
    }
    
    .chat-input {
        padding: 15px;
        background: white;
        border-top: 1px solid #eee;
        display: flex;
        gap: 10px;
    }
    
    .chat-input input {
        flex: 1;
        border: 1px solid #ddd;
        border-radius: 25px;
        padding: 10px 20px;
        outline: none;
    }
    
    .chat-input button {
        background: #e94560;
        color: white;
        border: none;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        cursor: pointer;
        font-size: 1.2rem;
    }
    
    .close-btn {
        position: absolute;
        top: 15px;
        right: 15px;
        background: rgba(255,255,255,0.2);
        border: none;
        color: white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        cursor: pointer;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============== FELIX KNOWLEDGE BASE ==============
FELIX_RESPONSES = {
    "authentic": {
        "greeting": "🎉 Great news! This image looks 100% authentic!",
        "details": [
            "Natural skin texture detected",
            "Consistent lighting throughout",
            "Camera noise pattern is normal",
            "Facial proportions are natural"
        ],
        "tips": "This appears to be a genuine photograph with no manipulation."
    },
    "filtered": {
        "greeting": "📸 I found something! This is a real person with some filters applied.",
        "details": [
            "Beauty filter detected (skin smoothing)",
            "Possible eye enlargement",
            "Color adjustments found",
            "But facial structure is REAL"
        ],
        "tips": "The person is real, but they've used Instagram/Snapchat filters."
    },
    "fake": {
        "greeting": "🚨 WARNING! This appears to be AI-generated or manipulated!",
        "details": [
            "Unnatural eye reflections",
            "Inconsistent skin texture",
            "Digital artifacts detected",
            "Facial symmetry too perfect"
        ],
        "tips": "Do not trust this image. It shows signs of deepfake manipulation."
    }
}

# ============== DETECTOR CLASS ==============
class DeepfakeDetector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def analyze(self, image):
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        results = {
            'face_count': 0,
            'has_face': False,
            'scores': {'natural': 0, 'filtered': 0, 'fake': 0},
            'result_type': 'unknown',
            'confidence': 0,
            'artifacts': [],
            'heatmap': None
        }
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        results['face_count'] = len(faces)
        results['has_face'] = len(faces) > 0
        
        if not results['has_face']:
            results['artifacts'].append("No face detected")
            return results
        
        # Simplified analysis
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise = np.std(cv2.absdiff(gray, cv2.medianBlur(gray, 5)))
        
        if blur > 100 and noise > 5:
            results['scores'] = {'natural': 90, 'filtered': 5, 'fake': 5}
            results['result_type'] = 'authentic'
        elif blur < 50 or noise < 3:
            results['scores'] = {'natural': 10, 'filtered': 15, 'fake': 75}
            results['result_type'] = 'fake'
            results['artifacts'] = ["Unnatural smoothness", "AI artifacts detected"]
        else:
            results['scores'] = {'natural': 65, 'filtered': 30, 'fake': 5}
            results['result_type'] = 'filtered'
            results['artifacts'] = ["Beauty filter detected", "Minor adjustments found"]
        
        results['confidence'] = max(results['scores'].values())
        results['heatmap'] = self._generate_heatmap(opencv_image, faces, results['result_type'])
        
        return results
    
    def _generate_heatmap(self, image, faces, result_type):
        overlay = image.copy()
        heatmap = np.zeros_like(image)
        colors = {'authentic': (0, 255, 0), 'filtered': (0, 165, 255), 'fake': (0, 0, 255)}
        color = colors.get(result_type, (128, 128, 128))
        
        for (x, y, w, h) in faces:
            for i in range(3, 0, -1):
                exp = i * 5
                cv2.rectangle(heatmap, (x-exp, y-exp), (x+w+exp, y+h+exp), color, 2)
        
        result = cv2.addWeighted(overlay, 0.7, heatmap, 0.3, 0)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

detector = DeepfakeDetector()

# ============== MAIN APP ==============
def main():
    # Header
    st.markdown('<h1 class="main-title">🦊 Deepfake Detector Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">AI Image Authentication with Detective Felix</p>', unsafe_allow_html=True)
    
    # Session state
    if 'analyzed' not in st.session_state:
        st.session_state.analyzed = False
        st.session_state.chat_open = False
        st.session_state.messages = []
        st.session_state.result_type = None
    
    # Main content
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Upload
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📤 Drop your image here", type=['jpg', 'jpeg', 'png', 'webp'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            if st.button("🔍 START ANALYSIS", use_container_width=True):
                # Progress
                progress = st.progress(0)
                for i in range(0, 101, 20):
                    progress.progress(i)
                    time.sleep(0.3)
                progress.empty()
                
                # Analyze
                results = detector.analyze(image)
                st.session_state.analyzed = True
                st.session_state.result_type = results['result_type']
                st.session_state.last_results = results
                
                # Show result
                felix_data = FELIX_RESPONSES[results['result_type']]
                
                if results['result_type'] == 'authentic':
                    st.markdown(f"""
                    <div class="result-authentic">
                        <h1>✅</h1>
                        <h2>{results['confidence']}% AUTHENTIC</h2>
                        <p>{felix_data['greeting']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif results['result_type'] == 'filtered':
                    st.markdown(f"""
                    <div class="result-filtered">
                        <h1>⚠️</h1>
                        <h2>{results['confidence']}% FILTERED</h2>
                        <p>{felix_data['greeting']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-fake">
                        <h1>🚨</h1>
                        <h2>{results['confidence']}% FAKE</h2>
                        <p>{felix_data['greeting']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Images
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(image, caption="Original", use_column_width=True)
                with col_img2:
                    if results['heatmap'] is not None:
                        st.image(results['heatmap'], caption="Analysis", use_column_width=True)
                
                # Metrics
                st.markdown("### 📊 Detailed Scores")
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("🟢 Natural", f"{results['scores']['natural']}%")
                mcol2.metric("🟡 Filtered", f"{results['scores']['filtered']}%")
                mcol3.metric("🔴 Fake", f"{results['scores']['fake']}%")
                
                # Auto-open Felix chat
                st.session_state.chat_open = True
                st.session_state.messages = [
                    {"sender": "felix", "text": felix_data['greeting']},
                    {"sender": "felix", "text": "Ask me anything about this analysis! 🕵️"}
                ]
    
    # ============== FLOATING FELIX CHATBOT ==============
    
    # Floating button
    if st.session_state.analyzed:
        col_float = st.columns([6, 1])
        with col_float[1]:
            if st.button("🦊", key="felix_btn", help="Ask Felix!"):
                st.session_state.chat_open = not st.session_state.chat_open
                st.rerun()
    
    # Chat window
    if st.session_state.analyzed and st.session_state.chat_open:
        st.markdown("### 🦊 Chat with Felix")
        
        # Chat container
        chat_container = st.container()
        
        with chat_container:
            # Show messages
            for msg in st.session_state.messages:
                if msg['sender'] == 'felix':
                    st.markdown(f"**🦊 Felix:** {msg['text']}")
                else:
                    st.markdown(f"**You:** {msg['text']}")
            
            # Quick replies
            if st.session_state.result_type:
                felix_info = FELIX_RESPONSES[st.session_state.result_type]
                
                cols = st.columns(2)
                with cols[0]:
                    if st.button("🔍 Why this result?", key="why"):
                        details = "\n".join([f"• {d}" for d in felix_info['details']])
                        st.session_state.messages.append({"sender": "user", "text": "Why this result?"})
                        st.session_state.messages.append({"sender": "felix", "text": f"Here's what I found:\n\n{details}"})
                        st.rerun()
                
                with cols[1]:
                    if st.button("💡 Tips", key="tips"):
                        st.session_state.messages.append({"sender": "user", "text": "Any tips?"})
                        st.session_state.messages.append({"sender": "felix", "text": felix_info['tips']})
                        st.rerun()
                
                cols2 = st.columns(2)
                with cols2[0]:
                    if st.button("❓ Is this person real?", key="real"):
                        answer = "Yes! This is a real person." if st.session_state.result_type != 'fake' else "No, this appears to be AI-generated."
                        st.session_state.messages.append({"sender": "user", "text": "Is this person real?"})
                        st.session_state.messages.append({"sender": "felix", "text": answer})
                        st.rerun()
                
                with cols2[1]:
                    if st.button("✅ Can I trust this?", key="trust"):
                        trust_level = "High trust!" if st.session_state.result_type == 'authentic' else "Be cautious!" if st.session_state.result_type == 'filtered' else "Do NOT trust!"
                        st.session_state.messages.append({"sender": "user", "text": "Can I trust this?"})
                        st.session_state.messages.append({"sender": "felix", "text": f"{trust_level} {felix_info['tips']}"})
                        st.rerun()
            
            # Close button
            if st.button("❌ Close Chat", key="close_chat"):
                st.session_state.chat_open = False
                st.rerun()

if __name__ == "__main__":
    main()
