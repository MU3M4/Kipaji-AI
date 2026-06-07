import streamlit as st
import websocket
import json
import threading
import time

st.set_page_config(page_title="Kipaji AI Live Telemetry", layout="wide")
st.title("📡 Kipaji AI: Live Bias Mitigation & Credit Telemetry")

if 'logs' not in st.session_state:
    st.session_state.logs = []

def on_message(ws, message):
    data = json.loads(message)
    st.session_state.logs.append(data)
    if len(st.session_state.logs) > 20:
        st.session_state.logs.pop(0)

def start_ws():
    # Point this to your live Render WebSocket URL
    ws_url = "wss://kipaji-ai.onrender.com/ws/telemetry" 
    ws = websocket.WebSocketApp(ws_url, on_message=on_message)
    ws.run_forever()

if 'ws_thread' not in st.session_state:
    st.session_state.ws_thread = threading.Thread(target=start_ws, daemon=True)
    st.session_state.ws_thread.start()
    time.sleep(2)

placeholder = st.empty()
while True:
    with placeholder.container():
        for log in reversed(st.session_state.logs):
            with st.expander(f"🏦 {log.get('merchant_id')} | Tier: {log.get('credit_tier')} | KSH {log.get('approved_local', 0)}"):
                st.json(log)
                if log.get('bias_proxy_removed'):
                    st.warning(f"⚠️ Bias Proxies Stripped: {', '.join(log['bias_proxy_removed'])}")
    time.sleep(2)