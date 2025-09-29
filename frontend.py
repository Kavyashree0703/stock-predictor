import streamlit as st
import requests
import openai
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="Stock Predictor", page_icon="📈", layout="wide")

# Load OpenAI key from env
openai.api_key = os.getenv("OPENAI_API_KEY")

# ---------- NAVBAR ----------
st.markdown(
    """
    <style>
        .navbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 25px;
            background-color: #0d6efd;
            color: white;
            border-radius: 8px;
        }
        .navbar h2 {
            margin: 0;
            color: white;
        }
        .nav-links a {
            margin-left: 20px;
            text-decoration: none;
            color: white;
            font-weight: bold;
        }
        .nav-links a:hover {
            text-decoration: underline;
        }
        .card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
    </style>

    <div class="navbar">
        <h2>📈 Stock Predictor</h2>
        <div class="nav-links">
            <a href="#prediction">Prediction</a>
            <a href="#chatbot">AI Chatbot</a>
            <a href="#about">About</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- HOME ----------
st.markdown("## Welcome to Stock Predictor 🚀")
st.write("Get real-time AI-powered insights on stock prices and financial trends.")

# ---------- STOCK PREDICTION ----------
st.markdown("### 🔮 Stock Prediction", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, MSFT)")
    if st.button("Predict Price"):
        try:
            response = requests.get(f"http://127.0.0.1:5000/predict?symbol={symbol}")
            if response.status_code == 200:
                data = response.json()
                if "prediction" in data:
                    st.success(f"Predicted price for {data['symbol']} is **${data['prediction']}**")
                else:
                    st.error(f"Error: {data['error']}")
            else:
                st.error("Backend server error.")
        except Exception as e:
            st.error(f"⚠️ Could not connect to Flask API: {e}")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- AI CHATBOT ----------
st.markdown("### 🤖 AI Chatbot", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("You:", key="chat_input")

    if st.button("Send"):
        if user_input:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant that explains finance and investing concepts in simple terms."},
                        {"role": "user", "content": user_input}
                    ]
                )
                bot_reply = response["choices"][0]["message"]["content"]

            except Exception as e:
                bot_reply = f"⚠️ Error contacting AI: {e}"

            st.session_state["chat_history"].append(("You", user_input))
            st.session_state["chat_history"].append(("Bot", bot_reply))

    for speaker, msg in st.session_state["chat_history"]:
        if speaker == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- ABOUT ----------
st.markdown("### ℹ️ About", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("""
        This project combines:
        - 💻 **Flask + TensorFlow** backend for stock price prediction  
        - 📊 **Streamlit** frontend for user interaction  
        - 🤖 **OpenAI GPT** chatbot for financial insights  
    """)
    st.markdown('</div>', unsafe_allow_html=True)
