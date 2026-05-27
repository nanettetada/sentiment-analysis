"""ReviewCoach — a conversational AI that analyses sentiment, explains why,
and helps you rewrite text. Powered by a fine-tuned DistilBERT.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

st.set_page_config(
    page_title="ReviewCoach — Sentiment AI",
    page_icon=":robot_face:",
    layout="wide",
)

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .hero {
        background: linear-gradient(135deg, #F39C12 0%, #7E4F00 100%);
        padding: 32px 28px;
        border-radius: 16px;
        color: white;
        margin: -10px 0 22px 0;
        box-shadow: 0 12px 32px rgba(243, 156, 18, 0.25);
    }
    .hero h1 { margin: 0; font-size: 36px; font-weight: 800; letter-spacing: -0.5px; }
    .hero p  { margin: 8px 0 0 0; font-size: 16px; opacity: 0.95; }
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 700; letter-spacing: 0.6px;
        color: white; margin: 0 4px;
    }
    .badge.pos { background: #27AE60; }
    .badge.neg { background: #E74C3C; }
    .insight {
        background: linear-gradient(180deg, #FFFFFF 0%, #FFF8E7 100%);
        border-left: 4px solid #F39C12;
        padding: 14px 18px;
        border-radius: 10px;
        margin: 8px 0;
    }
    .insight .head { font-size: 11px; color: #F39C12; font-weight: 800; letter-spacing: 1.2px; }
    .insight .body { font-size: 15px; color: #2C3E50; margin-top: 4px; }
    </style>

    <div class="hero">
      <h1>:robot_face: ReviewCoach</h1>
      <p>A conversational AI that scores sentiment, explains <i>why</i>, and helps you rewrite. Powered by a fine-tuned DistilBERT.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading DistilBERT (first launch downloads ~250 MB)...")
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl


tokenizer, model = load_model()


# --------------------------------------------------------------------------- #
def score(text: str) -> dict[str, Any]:
    """Return sentiment label, confidence, and per-token attribution."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    pos_prob = float(probs[1])
    label = "POSITIVE" if pos_prob >= 0.5 else "NEGATIVE"
    confidence = max(pos_prob, 1 - pos_prob)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attributions = []
    for i in range(1, len(tokens) - 1):  # skip [CLS] and [SEP]
        masked_ids = inputs["input_ids"].clone()
        masked_ids[0, i] = tokenizer.mask_token_id
        with torch.no_grad():
            masked_out = model(
                input_ids=masked_ids, attention_mask=inputs["attention_mask"]
            )
        masked_pos = float(torch.softmax(masked_out.logits, dim=-1)[0, 1])
        attributions.append({
            "token": tokens[i].replace("##", ""),
            "delta": pos_prob - masked_pos,
        })

    return {
        "label": label, "confidence": confidence, "pos_prob": pos_prob,
        "attributions": attributions,
    }


# --------------------------------------------------------------------------- #
def parse_intent(message: str, has_prior_text: bool) -> str:
    m = message.lower()
    if re.search(r"\b(rewrite|reword|rephrase|make it|change it|fix it|improve)\b", m):
        if "positive" in m or "nicer" in m or "happier" in m:
            return "rewrite_positive"
        if "negative" in m or "harsh" in m or "critical" in m:
            return "rewrite_negative"
        if "formal" in m or "professional" in m or "polite" in m:
            return "rewrite_formal"
        return "rewrite_positive"
    if re.search(r"\b(why|explain|what made|reason|which word)\b", m) and has_prior_text:
        return "explain"
    if re.search(r"\b(hi|hello|hey|sup|hola)\b", m) and len(m) < 30:
        return "greet"
    if re.search(r"\b(help|what can you do|how does this work|features)\b", m):
        return "help"
    return "score"


NEGATIVE_TO_POSITIVE = {
    "terrible": "challenging", "awful": "underwhelming", "horrible": "disappointing",
    "worst": "less than ideal", "hate": "find difficult to enjoy",
    "broken": "needing attention", "useless": "limited", "trash": "subpar",
    "garbage": "lacking", "disgusting": "unappealing", "stupid": "puzzling",
    "ridiculous": "unexpected", "annoying": "noticeable", "rude": "abrupt",
    "incompetent": "still learning", "slow": "taking time", "boring": "subtle",
    "bad": "off", "waste": "investment that didn't land",
}
POSITIVE_TO_NEGATIVE = {
    "love": "tolerate", "amazing": "passable", "wonderful": "fine",
    "excellent": "adequate", "perfect": "okay", "fantastic": "alright",
    "great": "fine", "brilliant": "passable", "outstanding": "noticeable",
    "best": "least bad", "incredible": "noteworthy",
}
INFORMAL_TO_FORMAL = {
    "stuff": "items", "gonna": "going to", "wanna": "want to",
    "kinda": "somewhat", "sorta": "in a sense", "yeah": "yes",
    "yep": "yes", "nope": "no", "ok": "acceptable",
    "guys": "team", "thing": "item", "really": "considerably",
}


def rewrite(text: str, mapping: dict[str, str]) -> str:
    def _swap(match):
        word = match.group(0)
        repl = mapping.get(word.lower(), word)
        if word[0].isupper() and repl:
            repl = repl[0].upper() + repl[1:]
        return repl
    pattern = r"\b(" + "|".join(re.escape(k) for k in mapping) + r")\b"
    return re.sub(pattern, _swap, text, flags=re.IGNORECASE)


def respond(message: str, history: list) -> dict[str, Any]:
    last_user_text = None
    for turn in reversed(history):
        if turn["role"] == "user" and turn.get("scored_text"):
            last_user_text = turn["scored_text"]
            break

    intent = parse_intent(message, has_prior_text=bool(last_user_text))

    if intent == "greet":
        return {
            "text": (
                "Hi! I'm ReviewCoach. Paste any review or message and I'll tell you "
                "how positive or negative it sounds, and *why*. Try things like:\n\n"
                "- *\"I absolutely love this gadget!\"*\n"
                "- *\"Why is this negative?\"*\n"
                "- *\"Rewrite that more positively\"*"
            ),
            "scored_text": None,
        }

    if intent == "help":
        return {
            "text": (
                "**What I can do:**\n"
                "1. **Score sentiment** of any text you paste (just send the text).\n"
                "2. **Explain** the score — token-by-token contribution.\n"
                "3. **Rewrite** the last text: more positive, more critical, or more formal.\n\n"
                "Powered by `distilbert-base-uncased-finetuned-sst-2-english` — "
                "≈92% accuracy on the SST-2 benchmark."
            ),
            "scored_text": None,
        }

    if intent == "rewrite_positive" and last_user_text:
        out = rewrite(last_user_text, NEGATIVE_TO_POSITIVE)
        new_score = score(out)
        return {
            "text": (
                f"Here's a more positive version:\n\n"
                f"> {out}\n\n"
                f"New sentiment: **{new_score['label']}** at "
                f"{new_score['confidence']*100:.0f}% confidence."
            ),
            "scored_text": out, "score": new_score,
        }

    if intent == "rewrite_negative" and last_user_text:
        out = rewrite(last_user_text, POSITIVE_TO_NEGATIVE)
        new_score = score(out)
        return {
            "text": (
                f"Here's a more critical version:\n\n"
                f"> {out}\n\n"
                f"New sentiment: **{new_score['label']}** at "
                f"{new_score['confidence']*100:.0f}% confidence."
            ),
            "scored_text": out, "score": new_score,
        }

    if intent == "rewrite_formal" and last_user_text:
        out = rewrite(last_user_text, INFORMAL_TO_FORMAL)
        new_score = score(out)
        return {
            "text": (
                f"Here's a more formal version:\n\n"
                f"> {out}\n\n"
                f"New sentiment: **{new_score['label']}** at "
                f"{new_score['confidence']*100:.0f}% confidence."
            ),
            "scored_text": out, "score": new_score,
        }

    if intent == "explain" and last_user_text:
        last_score = None
        for turn in reversed(history):
            if turn.get("score"):
                last_score = turn["score"]
                break
        if last_score:
            attrs = sorted(last_score["attributions"], key=lambda x: x["delta"])
            top_neg = [a for a in attrs[:3] if a["delta"] < 0]
            top_pos = [a for a in reversed(attrs[-3:]) if a["delta"] > 0]
            parts = []
            if top_pos:
                parts.append(
                    "Pushing it **positive**: "
                    + ", ".join(f"`{a['token']}`" for a in top_pos)
                )
            if top_neg:
                parts.append(
                    "Pushing it **negative**: "
                    + ", ".join(f"`{a['token']}`" for a in top_neg)
                )
            return {"text": "\n\n".join(parts) or "No strong signal in either direction.",
                    "scored_text": None}

    # Default: treat the user's message as text to score
    s = score(message)
    return {"text": "", "scored_text": message, "score": s}


# --------------------------------------------------------------------------- #
def render_score(s: dict, text: str) -> None:
    colour = "#27AE60" if s["label"] == "POSITIVE" else "#E74C3C"
    a, b = st.columns([1, 1.3])
    with a:
        st.markdown(
            f'<span class="badge {"pos" if s["label"] == "POSITIVE" else "neg"}">'
            f'{s["label"]}</span>'
            f' &nbsp; <b>{s["confidence"]*100:.0f}%</b> confidence',
            unsafe_allow_html=True,
        )
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=s["pos_prob"] * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colour},
                "steps": [
                    {"range": [0, 50], "color": "#FADBD8"},
                    {"range": [50, 100], "color": "#D5F5E3"},
                ],
            },
        ))
        fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with b:
        attrs = sorted(s["attributions"], key=lambda x: x["delta"])
        plot_df = pd.DataFrame(attrs[:6] + attrs[-6:])
        plot_df = plot_df.sort_values("delta")
        fig = px.bar(
            plot_df, x="delta", y="token", orientation="h",
            color="delta", color_continuous_scale="RdYlGn", color_continuous_midpoint=0,
            labels={"delta": "Contribution to positive score", "token": ""},
        )
        fig.update_layout(coloraxis_showscale=False, height=300,
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# --------------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.subheader("How this works")
    st.markdown(
        "- Paste a review or message in the chat.\n"
        "- The model returns sentiment, confidence, and per-token contribution.\n"
        "- Then ask follow-ups:\n"
        "  - *\"Why is this negative?\"*\n"
        "  - *\"Rewrite it more positively\"*\n"
        "  - *\"Make it more formal\"*"
    )
    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.caption("Model: `distilbert-base-uncased-finetuned-sst-2-english`")
    st.caption("≈ 250 MB local model — no API, no data leaves the Space.")

for turn in st.session_state.history:
    with st.chat_message(turn["role"], avatar=("🧑" if turn["role"] == "user" else "🤖")):
        if turn.get("text"):
            st.markdown(turn["text"])
        if turn.get("score"):
            render_score(turn["score"], turn.get("scored_text", ""))

if not st.session_state.history:
    st.markdown(
        '<div class="insight"><div class="head">START HERE</div>'
        '<div class="body">Try pasting <i>"This product is absolutely fantastic, '
        'the best purchase of the year!"</i> — or paste your own review. '
        'Then ask me to <i>rewrite</i> or <i>explain</i>.</div></div>',
        unsafe_allow_html=True,
    )

prompt = st.chat_input("Paste a review, ask me to rewrite, or ask why...")
if prompt:
    st.session_state.history.append({"role": "user", "text": prompt, "scored_text": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            result = respond(prompt, st.session_state.history)
        if result.get("text"):
            st.markdown(result["text"])
        if result.get("score"):
            render_score(result["score"], result.get("scored_text", prompt))

    st.session_state.history.append({
        "role": "assistant",
        "text": result.get("text", ""),
        "scored_text": result.get("scored_text"),
        "score": result.get("score"),
    })
