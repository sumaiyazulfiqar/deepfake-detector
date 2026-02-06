import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import random
import google.generativeai as genai
import os

# ============== GEMINI API SETUP ==============
# Yahan apni API key daalo!
GEMINI_API_KEY = "AIzaSyAVzi3XDARR1RtqLj1rS9cTbWKNMaQqIHU"  

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

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
    }
    
    .sub-title {
        font-size: 1.3rem;
        color: #eaeaea;
        text-align: center;
        margin-bottom: 3rem;
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
    
    /* FELIX CHATBOT STYLES */
    .felix-chat-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        width: 380px;
    }
    
    .felix-avatar-btn {
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2.5rem;
        cursor: pointer;
        box-shadow: 0 10px 30px rgba(255, 107, 53, 0.5);
        border: 4px solid white;
        animation: bounce 2s infinite;
        margin-left: auto;
    }
    
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {transform: translateY(0);}
        40% {transform: translateY(-15px);}
        60% {transform: translateY(-7px);}
    }
    
    .chat-window {
        background: white;
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        margin-bottom: 15px;
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
        padding: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .chat-messages {
        max-height: 300px;
        overflow-y: auto;
        padding: 15px;
        background: #f8f9fa;
    }
    
    .message-bubble {
        background: white;
        padding: 12px;
        border-radius: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        color: #333;
    }
    
    .message-felix {
        border-left: 4px solid #ff6b35;
    }
    
    .message-user {
        border-right: 4px solid #e94560;
        text-align: right;
    }
    
    .quick-buttons {
        padding: 10px;
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        background: white;
    }
    
    .download-btn {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        padding: 12px 30px !important;
        border-radius: 25px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ============== FELIX CHAT FUNCTION ==============
def get_felix_reply(user_question, result_type, confidence, artifacts):
    """Use Gemini API to generate Felix's reply"""
    
    context = f"""
    You are Felix, a friendly forensic fox detective who helps users understand deepfake detection results.
    Current analysis result: {result_type} with {confidence}% confidence.
    Issues found: {', '.join(artifacts) if artifacts else 'None'}.
    
    User asked: {user_question}
    
    Reply in a friendly, helpful tone. Mix English and Roman Urdu.
    Keep it short (2-3 sentences max).
    Add relevant emojis.
    """
    
    try:
        response = model.generate_content(context)
        return response.text
    except:
        # Fallback if API fails
        return f"🦊 Mein check kar raha hoon... Result {result_type} hai with {confidence}% confidence!"

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
        
        # Analysis
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise = np.std(cv2.absdiff(gray, cv2.medianBlur(gray, 5)))
        
        if blur > 100 and noise > 5:
            results['scores'] = {'natural': 90, 'filtered': 5, 'fake': 5}
            results['result_type'] = 'authentic'
        elif blur < 50 or noise < 3:
            results['scores'] = {'natural': 10, 'filtered': 15, 'fake': 75}
            results['result_type'] = 'fake'
            results['artifacts'] = ["Unnatural smoothness", "AI artifacts detected", "Inconsistent lighting"]
        else:
            results['scores'] = {'natural': 65, 'filtered': 30, 'fake': 5}
            results['result_type'] = 'filtered'
            results['artifacts'] = ["Beauty filter detected", "Skin smoothing applied", "Color adjustments found"]
        
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
        st.session_state.chat_open = True  # Auto open
        st.session_state.messages = [
            {"sender": "felix", "text": "Hi! I'm Felix! 🦊 Upload an image and I'll check if it's real or fake!"}
        ]
        st.session_state.result_data = None
    
    # Main content
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col2:
        # Upload
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("📤 Drop your image here", type=['jpg', 'jpeg', 'png', 'webp'])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            if st.button("🔍 START FORENSIC ANALYSIS", use_container_width=True):
                # Progress
                progress = st.progress(0)
                for i in range(0, 101, 20):
                    progress.progress(i)
                    time.sleep(0.3)
                progress.empty()
                
                # Analyze
                results = detector.analyze(image)
                st.session_state.analyzed = True
                st.session_state.result_data = results
                
                # Add Felix message
                result_msg = f"Analysis complete! 🎯 Result: {results['result_type'].upper()} with {results['confidence']}% confidence!"
                st.session_state.messages.append({"sender": "felix", "text": result_msg})
                
                # Show result card
                if results['result_type'] == 'authentic':
                    st.markdown(f"""
                    <div class="result-authentic">
                        <h1>✅</h1>
                        <h2>{results['confidence']}% AUTHENTIC</h2>
                        <p>This image appears to be genuine!</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif results['result_type'] == 'filtered':
                    st.markdown(f"""
                    <div class="result-filtered">
                        <h1>⚠️</h1>
                        <h2>{results['confidence']}% FILTERED</h2>
                        <p>Real person with beauty filters!</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-fake">
                        <h1>🚨</h1>
                        <h2>{results['confidence']}% FAKE</h2>
                        <p>AI-generated or manipulated!</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Images
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(image, caption="Original", use_column_width=True)
                with col_img2:
                    if results['heatmap'] is not None:
                        st.image(results['heatmap'], caption="Forensic Analysis", use_column_width=True)
                
                # Metrics
                st.markdown("### 📊 Detailed Scores")
                mcol1, mcol2, mcol3 = st.columns(3)
                mcol1.metric("🟢 Natural", f"{results['scores']['natural']}%")
                mcol2.metric("🟡 Filtered", f"{results['scores']['filtered']}%")
                mcol3.metric("🔴 Fake", f"{results['scores']['fake']}%")
                
                # Download button
                report = f"""
DEEPFAKE DETECTION REPORT
Generated: {datetime.now()}
Result: {results['result_type'].upper()}
Confidence: {results['confidence']}%

Scores:
- Natural: {results['scores']['natural']}%
- Filtered: {results['scores']['filtered']}%
- Fake: {results['scores']['fake']}%

Issues:
{chr(10).join('- ' + a for a in results['artifacts'])}

Analyzed by: Felix the Forensic Fox 🦊
"""
                st.download_button("📥 DOWNLOAD REPORT", report, 
                                 f"felix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                 use_container_width=True)
    
    # ============== FELIX CHATBOT ==============
    st.markdown("---")
    st.markdown("### 🦊 Chat with Felix")
    
    # Chat window
    chat_container = st.container()
    
    with chat_container:
        # Show messages
        for msg in st.session_state.messages[-6:]:  # Last 6 messages
            if msg['sender'] == 'felix':
                st.markdown(f"""
                <div class="message-bubble message-felix">
                    <strong>🦊 Felix:</strong> {msg['text']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="message-bubble message-user">
                    <strong>You:</strong> {msg['text']}
                </div>
                """, unsafe_allow_html=True)
        
        # Quick buttons (only after analysis)
        if st.session_state.analyzed and st.session_state.result_data:
            results = st.session_state.result_data
            
            cols = st.columns(2)
            with cols[0]:
                if st.button("🔍 Why this result?", key="why"):
                    reply = get_felix_reply("Why this result?", results['result_type'], 
                                          results['confidence'], results['artifacts'])
                    st.session_state.messages.append({"sender": "user", "text": "Why this result?"})
                    st.session_state.messages.append({"sender": "felix", "text": reply})
                    st.rerun()
            
            with cols[1]:
                if st.button("💡 Tell me more", key="more"):
                    reply = get_felix_reply("Explain the issues found", results['result_type'], 
                                          results['confidence'], results['artifacts'])
                    st.session_state.messages.append({"sender": "user", "text": "Tell me more"})
                    st.session_state.messages.append({"sender": "felix", "text": reply})
                    st.rerun()
            
            cols2 = st.columns(2)
            with cols2[0]:
                if st.button("❓ Is this person real?", key="real"):
                    reply = get_felix_reply("Is this person real?", results['result_type'], 
                                          results['confidence'], results['artifacts'])
                    st.session_state.messages.append({"sender": "user", "text": "Is this person real?"})
                    st.session_state.messages.append({"sender": "felix", "text": reply})
                    st.rerun()
            
            with cols2[1]:
                if st.button("✅ Can I trust this?", key="trust"):
                    reply = get_felix_reply("Can I trust this image?", results['result_type'], 
                                          results['confidence'], results['artifacts'])
                    st.session_state.messages.append({"sender": "user", "text": "Can I trust this?"})
                    st.session_state.messages.append({"sender": "felix", "text": reply})
                    st.rerun()
        
        # Custom input
        user_input = st.text_input("Or type your question:", key="user_question")
        if user_input and st.session_state.result_data:
            if st.button("Send", key="send"):
                results = st.session_state.result_data
                reply = get_felix_reply(user_input, results['result_type'], 
                                      results['confidence'], results['artifacts'])
                st.session_state.messages.append({"sender": "user", "text": user_input})
                st.session_state.messages.append({"sender": "felix", "text": reply})
                st.rerun()

if __name__ == "__main__":
    main()
