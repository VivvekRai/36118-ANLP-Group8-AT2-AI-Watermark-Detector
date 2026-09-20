import textwrap
import threading
import time

import streamlit as st
import streamlit.components.v1 as components

from src.watermark import load_model, generate_watermarked_text, detect_watermark, highlight_tokens
from src.watermark.robustness import remove_watermark_stub

st.set_page_config(page_title="AI Text Watermark Detector", page_icon="🟩", layout="wide")

# ---- Custom styling ----
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=Fira+Code:wght@400;500;600&display=swap');

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .stApp {
        background:
            linear-gradient(rgba(57, 255, 20, 0.05) 1px, transparent 1px) 0 0 / 42px 42px,
            linear-gradient(90deg, rgba(57, 255, 20, 0.05) 1px, transparent 1px) 0 0 / 42px 42px,
            radial-gradient(circle at 20% 10%, rgba(57, 255, 20, 0.07) 0%, transparent 40%),
            radial-gradient(circle at 80% 90%, rgba(255, 57, 57, 0.05) 0%, transparent 40%),
            radial-gradient(circle at top, #0d1410 0%, #050505 60%) !important;
        background-attachment: fixed !important;
        color: #d8ffd8;
        font-family: 'Fira Code', monospace;
    }

    .block-container {
        background: rgba(5, 10, 5, 0.72);
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
        border: 1px solid rgba(57, 255, 20, 0.15);
        border-radius: 16px;
        padding: clamp(1.2rem, 4vw, 3rem) !important;
        margin: 2rem auto;
        max-width: 1100px;
        width: 92%;
        box-shadow: 0 0 40px rgba(0, 0, 0, 0.6);
    }

    h1 {
        font-family: 'Chakra Petch', sans-serif !important;
        font-weight: 700 !important;
        font-size: clamp(1.6rem, 5vw, 2.4rem) !important;
        text-align: center !important;
        background: linear-gradient(90deg, #39ff14, #7fffb0, #39ff14);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: shimmer 4s linear infinite;
        letter-spacing: 1px;
    }
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    h2, h3, h4 {
        font-family: 'Chakra Petch', sans-serif !important;
        color: #39ff14 !important;
        text-shadow: 0 0 12px rgba(57, 255, 20, 0.5);
        letter-spacing: 1px;
    }

    .subtitle {
        color: #7fffb0;
        font-size: 13px;
        letter-spacing: 3px;
        text-transform: uppercase;
        opacity: 0.85;
        margin-top: -8px;
        font-family: 'Fira Code', monospace;
        text-align: center;
        display: block;
    }

    .field-hint {
        color: #6fce9a;
        font-size: 12px;
        margin-top: -8px;
        margin-bottom: 10px;
        font-family: 'Fira Code', monospace;
        opacity: 0.85;
    }

    .stTextArea textarea, .stTextInput input {
        background-color: #0a0f0a !important;
        color: #39ff14 !important;
        font-family: 'Fira Code', monospace !important;
        border: 1px solid #1f4d2e !important;
        border-radius: 8px !important;
        box-shadow: inset 0 0 12px rgba(57, 255, 20, 0.08);
    }

    label, .stMarkdown p, .stCaption, .stSlider label {
        color: #b8ffcf !important;
        font-family: 'Fira Code', monospace !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #0f5c26, #1fae4d);
        color: #eaffea;
        font-family: 'Chakra Petch', sans-serif;
        font-weight: 700;
        letter-spacing: 2px;
        border: 1px solid #39ff14;
        border-radius: 8px;
        box-shadow: 0 0 18px rgba(57, 255, 20, 0.35);
        transition: 0.2s ease-in-out;
    }
    div.stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 32px rgba(57, 255, 20, 0.8);
        transform: translateY(-2px) scale(1.01);
    }

    div.stButton > button[kind="secondary"] {
        background-color: #0a0f0a;
        color: #7fffb0;
        font-family: 'Chakra Petch', sans-serif;
        font-weight: 700;
        letter-spacing: 1.5px;
        border: 1px solid #1f4d2e;
        border-radius: 8px;
        transition: 0.2s ease-in-out;
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: #39ff14;
        box-shadow: 0 0 16px rgba(57, 255, 20, 0.3);
    }

    @keyframes reveal {
        0% { opacity: 0; transform: scale(0.96); }
        100% { opacity: 1; transform: scale(1); }
    }

    .verdict-banner {
        padding: 18px 20px;
        border-radius: 10px;
        font-family: 'Chakra Petch', sans-serif;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin: 16px 0 22px 0;
        letter-spacing: 2px;
        animation: reveal 0.4s ease-out;
    }
    .verdict-yes {
        background-color: #0a1f0f;
        color: #39ff14;
        border: 1px solid #39ff14;
        box-shadow: 0 0 30px rgba(57, 255, 20, 0.5);
    }
    .verdict-no {
        background-color: #1f0a0a;
        color: #ff3939;
        border: 1px solid #ff3939;
        box-shadow: 0 0 30px rgba(255, 57, 57, 0.5);
    }

    .token-box {
        background-color: #060a06;
        padding: 16px;
        border-radius: 10px;
        line-height: 2.3;
        border: 1px solid #1f4d2e;
        font-family: 'Fira Code', monospace;
        animation: reveal 0.5s ease-out;
    }
    .token-box span:hover {
        transform: scale(1.12);
        filter: brightness(1.3);
        cursor: default;
    }

    div[data-testid="stMetric"] {
        background-color: #0a0f0a;
        border: 1px solid #1f4d2e;
        border-radius: 10px;
        padding: 12px;
        box-shadow: 0 0 14px rgba(57, 255, 20, 0.12);
        transition: 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 0 24px rgba(57, 255, 20, 0.3);
    }
    div[data-testid="stMetricLabel"] { color: #7fffb0 !important; }
    div[data-testid="stMetricValue"] { color: #39ff14 !important; font-family: 'Fira Code', monospace !important; }

    hr { border-color: #1f4d2e !important; }

    .streamlit-expanderHeader {
        color: #39ff14 !important;
        font-family: 'Fira Code', monospace !important;
    }

    @media (max-width: 600px) {
        .block-container {
            width: 95%;
            border-radius: 10px;
        }
        div.stButton > button {
            font-size: 0.85rem;
            letter-spacing: 1px;
        }
        .verdict-banner {
            font-size: 18px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _loader_component(pct_display, label):
    """
    Fully self-contained loader (inline CSS + HTML) for use inside
    components.html, which renders in an isolated iframe and cannot see
    the main page's stylesheet.
    """
    dots = ""
    for i in range(12):
        angle = i * 30
        delay = i * (1.4 / 12)
        dots += (
            f'<div class="dot-holder" style="transform: rotate({angle}deg) translate(62px);">'
            f'<div class="loader-dot" style="animation-delay: {delay}s;"></div>'
            f"</div>"
        )
    html = f"""
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            background: transparent;
            font-family: 'Fira Code', monospace;
        }}
        .loader-wrap {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 0;
        }}
        .loader-ring {{
            position: relative;
            width: 150px;
            height: 150px;
        }}
        .dots-spin {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            animation: spin 1.4s linear infinite;
        }}
        @keyframes spin {{
            from {{ transform: rotate(0deg); }}
            to {{ transform: rotate(360deg); }}
        }}
        .dot-holder {{
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
        }}
        .loader-dot {{
            width: 13px;
            height: 13px;
            margin-left: -6.5px;
            margin-top: -6.5px;
            border-radius: 50%;
            background: #39ff14;
            box-shadow: 0 0 10px rgba(57, 255, 20, 0.9);
            animation: dotFade 1.4s infinite ease-in-out;
        }}
        @keyframes dotFade {{
            0%, 100% {{ opacity: 0.15; }}
            50% {{ opacity: 1; }}
        }}
        .loader-center {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            text-align: center;
            width: 100px;
        }}
        .loader-pct {{
            font-family: sans-serif;
            font-size: 26px;
            font-weight: 700;
            color: #39ff14;
            text-shadow: 0 0 10px rgba(57, 255, 20, 0.7);
        }}
        .loader-label {{
            font-size: 13px;
            color: #7fffb0;
            letter-spacing: 1px;
            margin-top: 14px;
            text-transform: uppercase;
            opacity: 0.85;
        }}
    </style>
    </head>
    <body>
        <div class="loader-wrap">
            <div class="loader-ring">
                <div class="dots-spin">
                    {dots}
                </div>
                <div class="loader-center">
                    {pct_display}
                </div>
            </div>
            <div class="loader-label">{label}</div>
        </div>
    </body>
    </html>
    """
    return textwrap.dedent(html).strip()


def show_loader(placeholder, label, pct=None):
    pct_display = f'<div class="loader-pct">{pct}%</div>' if pct is not None else ""
    with placeholder:
        components.html(_loader_component(pct_display, label), height=280)


def run_with_progress(placeholder, label, expected_seconds, func, *args, **kwargs):
    """
    Runs func(*args, **kwargs) on a background thread while showing a
    percentage-based loader in placeholder. Returns func's result.
    """
    result_holder = {}

    def _run():
        result_holder["value"] = func(*args, **kwargs)

    thread = threading.Thread(target=_run)
    thread.start()

    start = time.time()
    while thread.is_alive():
        elapsed = time.time() - start
        pct = min(95, int((elapsed / expected_seconds) * 100))
        show_loader(placeholder, label, pct)
        time.sleep(0.1)

    thread.join()
    show_loader(placeholder, label, 100)
    time.sleep(0.3)
    placeholder.empty()

    return result_holder["value"]


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model("gpt2")


if "model_loaded" not in st.session_state:
    progress_placeholder = st.empty()
    model = run_with_progress(progress_placeholder, "Loading GPT-2", 4, get_model)
    st.session_state.model_loaded = True

tokenizer, model = get_model()

# ---- Session state defaults (must exist before the widgets that use them as keys) ----
st.session_state.setdefault("check_text", "")
st.session_state.setdefault("removal_text", "")
st.session_state.setdefault("active_tab", "generate")

# ---- Header ----
st.markdown("# AI TEXT WATERMARK DETECTOR")
st.markdown('<p class="subtitle">GREEN/RED-LIST SCHEME (KIRCHENBAUER ET AL., 2023)</p>', unsafe_allow_html=True)

with st.expander("ℹ️  HOW THIS WORKS, AND WHAT IT CANNOT DO"):
    st.markdown(
        """
        **How detection works:** While generating text, the model is gently nudged to prefer a
        randomly chosen "green list" of words at each step. Later, we count how often the
        actual words match that green list. Far more matches than a 1-in-4 chance rate is
        strong statistical evidence of a watermark.

        **Important limitation:** This tool can only detect text watermarked with this specific
        scheme and secret key. It **cannot** detect text from ChatGPT, Claude, Gemini, or any
        other AI system. Those would use entirely different and private watermarking schemes,
        if they use one at all. This mirrors real production systems like Google's SynthID, which
        has the same fundamental limitation.
        """
    )

st.divider()


def render_result(label, text, result):
    st.markdown(f"#### {label}")
    st.text_area(" ", text, height=120, label_visibility="collapsed", key=f"display_{label}_{id(result)}")

    if result["is_watermarked"]:
        st.markdown('<div class="verdict-banner verdict-yes">🟢 WATERMARKED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-banner verdict-no">🔴 NOT WATERMARKED</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("GREEN FRACTION", f"{result['green_fraction']:.1%}")
    col2.metric("Z-SCORE", f"{result['z_score']:.2f}")
    col3.metric("CONFIDENCE", result["confidence"].upper())

    st.caption("🟩 matched the green list, 🟥 did not, ⬜ first token (no previous token to seed a list)")
    highlighted_html = highlight_tokens(tokenizer, text)
    st.markdown(f'<div class="token-box">{highlighted_html}</div>', unsafe_allow_html=True)


# ---- Tab switcher (built from regular buttons, not st.tabs) ----
col_t1, col_t2 = st.columns(2)
with col_t1:
    if st.button(
        "🧬 GENERATE & DETECT", use_container_width=True,
        type="primary" if st.session_state.active_tab == "generate" else "secondary",
        key="tab_btn_generate",
    ):
        st.session_state.active_tab = "generate"
        st.rerun()
with col_t2:
    if st.button(
        "🧪 REMOVAL LAB", use_container_width=True,
        type="primary" if st.session_state.active_tab == "removal" else "secondary",
        key="tab_btn_removal",
    ):
        st.session_state.active_tab = "removal"
        st.rerun()

st.divider()

# =========================================================
# TAB 1: GENERATE & DETECT
# =========================================================
if st.session_state.active_tab == "generate":
    st.markdown("### GENERATE")
    prompt_input = st.text_area(
        "PROMPT",
        height=120,
        placeholder="> The future of artificial intelligence is...",
        key="prompt_input",
    )
    delta = st.slider(
        "DELTA (WATERMARK STRENGTH)", 0.5, 5.0, 2.0, 0.5,
        help="Controls how strongly generation is pushed toward the green list.",
    )
    st.markdown(
        '<p class="field-hint">Low = subtle, natural-sounding, but harder to detect. '
        'High = easy to detect, but the text may sound slightly less natural.</p>',
        unsafe_allow_html=True,
    )
    generate_clicked = st.button("▶ GENERATE", use_container_width=True, type="primary", key="generate_btn")

    if generate_clicked:
        if not prompt_input.strip():
            st.warning("Please enter a prompt first.")
        else:
            gen_loader = st.empty()
            generated = run_with_progress(
                gen_loader, "Running GPT-2 (generating)", 6,
                generate_watermarked_text, tokenizer, model, prompt_input, delta=delta,
            )
            st.session_state.check_text = generated
            st.session_state.removal_text = generated

    st.divider()

    st.markdown("### DETECT")
    st.text_area(
        "TEXT TO CHECK (auto-filled after generating, editable, or paste your own)",
        height=150,
        key="check_text",
    )
    z_threshold = st.slider(
        "Z-SCORE THRESHOLD", 2.0, 6.0, 4.0, 0.5,
        help="How confident the detector needs to be before calling something watermarked.",
    )
    st.markdown(
        '<p class="field-hint">Low = catches weaker watermarks, but more risk of a false alarm. '
        'High = very confident when it does flag something, but may miss weak cases.</p>',
        unsafe_allow_html=True,
    )
    detect_clicked = st.button("▶ DETECT", use_container_width=True, type="primary", key="detect_btn")

    if detect_clicked:
        if not st.session_state.check_text.strip():
            st.warning("Please generate or paste some text first.")
        else:
            detect_loader = st.empty()
            result = run_with_progress(
                detect_loader, "Running GPT-2 (analyzing tokens)", 2,
                detect_watermark, tokenizer, st.session_state.check_text, z_threshold=z_threshold,
            )
            st.session_state.removal_text = st.session_state.check_text
            st.divider()
            render_result("Detection result", st.session_state.check_text, result)

# =========================================================
# TAB 2: REMOVAL LAB (contents hidden for now)
# =========================================================
else:
    st.info("🚧 Coming soon...")