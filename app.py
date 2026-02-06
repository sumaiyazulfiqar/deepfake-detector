import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import google.generativeai as genai

# ============== API KEY ==============
GEMINI_API_KEY = "AIzaSyAVzi3XDARR1RtqLj1rS9cTbWKNMaQqIHU"  # CHANGE THIS!
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Deepfake Detector Pro | Felix the Forensic Fox",
    page_icon="🦊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CSS ==============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Poppins', sans-serif; }
    
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
    }
    
    .sub-title {
        font-size: 1.3rem;
        color: #eaeaea;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .upload-box {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(233, 69, 96, 0.5);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
    }
    
    .result-real {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
    }
    
    .result-filtered {
        background: linear-gradient(135deg, #f37335 0%, #fdc830 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
    }
    
    .result-fake {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
    }
    
    .chat-box {
        background: white;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }
    
    .felix-msg {
        background: #fff3e0;
        border-left: 4px solid #ff6b35;
        padding: 10px;
        margin: 10px 0;
        border-radius: 0 10px 10px 0;
    }
    
    .user-msg {
        background: #e3f2fd;
        border-right: 4px solid #2196f3;
        padding: 10px;
        margin: 10px 0;
        border-radius: 10px 0 0 10px;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# ============== SMART FELIX REPLIES ==============
def felix_reply(question, result_type, confidence, artifacts):
    q = question.lower()
    
    if "why" in q:
        prompt = f"Explain in 2 sentences why this image is {result_type}. Issues: {artifacts}. Use emojis, mix English and Roman Urdu."
    
    elif "real" in q or "person" in q:
        is_real = "YES" if result_type != "fake" else "NO"
        prompt = f"Answer in 2 sentences: Is the person real? Result is {result_type}. Say {is_real} clearly. Use emojis, Roman Urdu mix."
    
    elif "trust" in q:
        trust = "trust karo" if result_type == "authentic" else "careful raho" if result_type == "filtered" else "trust mat karo"
        prompt = f"Give advice in 2 sentences: Should user trust this {result_type} image? Say '{trust}'. Use emojis, Roman Urdu mix."
    
    elif "tip" in q or "how" in q:
        prompt = f"Give 3 short tips to spot {result_type} images. Bullet points. Use emojis, Roman Urdu mix."
    
    elif "filter" in q or "edit" in q:
        prompt = f"Explain filters detected: {artifacts}. How to spot them. 2 sentences. Use emojis, Roman Urdu mix."
    
    else:
        prompt = f"User asked: {question}. Image is {result_type} ({confidence}%). Give helpful 2-sentence reply. Use emojis, Roman Urdu mix."
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except:
        return f"🦊 Felix busy hai! Result: {result_type}, {confidence}% confidence."

# ============== DETECTOR ==============
class Detector:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def analyze(self, image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        
        result = {
            'faces': len(faces),
            'type': 'unknown',
            'confidence': 50,
            'scores': {'real': 0, 'filtered': 0, 'fake': 0},
            'issues': [],
            'heatmap': None
        }
        
        if len(faces) == 0:
            result['issues'] = ["No face found!"]
            return result
        
        # Simple analysis
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur > 100:
            result['type'] = 'authentic'
            result['confidence'] = 90
            result['scores'] = {'real': 90, 'filtered': 5, 'fake': 5}
        elif blur < 50:
            result['type'] = 'fake'
            result['confidence'] = 85
            result['scores'] = {'real': 10, 'filtered': 5, 'fake': 85}
            result['issues'] = ["Too smooth", "AI artifacts"]
        else:
            result['type'] = 'filtered'
            result['confidence'] = 75
            result['scores'] = {'real': 60, 'filtered': 35, 'fake': 5}
            result['issues'] = ["Beauty filter", "Skin smoothed"]
        
        # Heatmap
        heatmap = img.copy()
        for (x, y, w, h) in faces:
            color = (0, 255, 0) if result['type'] == 'authentic' else (0, 165, 255) if result['type'] == 'filtered' else (0, 0, 255)
            cv2.rectangle(heatmap, (x-10, y-10), (x+w+10, y+h+10), color, 3)
        
        result['heatmap'] = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        return result

detector = Detector()

# ============== MAIN ==============
def main():
    st.markdown('<h1 class="main-title">🦊 Deepfake Detector Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Felix the Forensic Fox - AI Image Authentication</p>', unsafe_allow_html=True)
    
    # Session
    if 'done' not in st.session_state:
        st.session_state.done = False
        st.session_state.chat = [{"who": "felix", "text": "Hi! I'm Felix! 🦊 Upload image, I'll check real or fake!"}]
        st.session_state.data = None
    
    # Upload
    uploaded = st.file_uploader("📤 Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded and not st.session_state.done:
        img = Image.open(uploaded)
        
        if st.button("🔍 ANALYZE"):
            # Progress
            bar = st.progress(0)
            for i in range(0, 101, 25):
                bar.progress(i)
                time.sleep(0.3)
            bar.empty()
            
            # Analyze
            data = detector.analyze(img)
            st.session_state.data = data
            st.session_state.done = True
            
            # Add Felix message
            st.session_state.chat.append({
                "who": "felix", 
                "text": f"Done! 🎯 Result: {data['type'].upper()} ({data['confidence']}%)"
            })
            
            # Show result
            if data['type'] == 'authentic':
                st.markdown(f'<div class="result-real"><h2>✅ {data["confidence"]}% REAL</h2></div>', unsafe_allow_html=True)
            elif data['type'] == 'filtered':
                st.markdown(f'<div class="result-filtered"><h2>⚠️ {data["confidence"]}% FILTERED</h2></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-fake"><h2>🚨 {data["confidence"]}% FAKE</h2></div>', unsafe_allow_html=True)
            
            # Images
            c1, c2 = st.columns(2)
            with c1: st.image(img, caption="Original")
            with c2: st.image(data['heatmap'], caption="Analysis")
            
            # Scores
            st.markdown("### 📊 Scores")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Real", f"{data['scores']['real']}%")
            cc2.metric("Filtered", f"{data['scores']['filtered']}%")
            cc3.metric("Fake", f"{data['scores']['fake']}%")
            
            # Download
            report = f"""RESULT: {data['type']}
CONFIDENCE: {data['confidence']}%
ISSUES: {', '.join(data['issues'])}
By Felix 🦊"""
            st.download_button("📥 Download Report", report, "report.txt")
    
    # Chat Section
    if st.session_state.done:
        st.markdown("---")
        st.markdown("### 🦊 Chat with Felix")
        
        # Show messages
        for msg in st.session_state.chat:
            if msg['who'] == 'felix':
                st.markdown(f'<div class="felix-msg"><strong>🦊 Felix:</strong> {msg["text"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="user-msg"><strong>You:</strong> {msg["text"]}</div>', unsafe_allow_html=True)
        
        # Quick buttons
        data = st.session_state.data
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            if st.button("🔍 Why?"):
                q = "Why this result?"
                r = felix_reply(q, data['type'], data['confidence'], data['issues'])
                st.session_state.chat.append({"who": "user", "text": q})
                st.session_state.chat.append({"who": "felix", "text": r})
                st.rerun()
        
        with c2:
            if st.button("❓ Real?"):
                q = "Is this person real?"
                r = felix_reply(q, data['type'], data['confidence'], data['issues'])
                st.session_state.chat.append({"who": "user", "text": q})
                st.session_state.chat.append({"who": "felix", "text": r})
                st.rerun()
        
        with c3:
            if st.button("✅ Trust?"):
                q = "Can I trust this?"
                r = felix_reply(q, data['type'], data['confidence'], data['issues'])
                st.session_state.chat.append({"who": "user", "text": q})
                st.session_state.chat.append({"who": "felix", "text": r})
                st.rerun()
        
        with c4:
            if st.button("💡 Tips"):
                q = "Give me tips"
                r = felix_reply(q, data['type'], data['confidence'], data['issues'])
                st.session_state.chat.append({"who": "user", "text": q})
                st.session_state.chat.append({"who": "felix", "text": r})
                st.rerun()
        
        # Custom input
        custom = st.text_input("Or type your question:")
        if custom and st.button("Send"):
            r = felix_reply(custom, data['type'], data['confidence'], data['issues'])
            st.session_state.chat.append({"who": "user", "text": custom})
            st.session_state.chat.append({"who": "felix", "text": r})
            st.rerun()

if __name__ == "__main__":
    main()
