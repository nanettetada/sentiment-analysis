"""Streamlit dashboard for the review sentiment project.

A sentiment tool built on a fine-tuned DistilBERT: coach a single review
(score it, explain why, rewrite it) or score a whole file of reviews.

Run with:
    streamlit run dashboard.py
"""
from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# --- Aspect keyword dictionary ---------------------------------------------
# Simple lexicon used to tag which business aspect a review touches on. We
# keep this hand-curated rather than ML so the demo runs without extra deps.
ASPECT_KEYWORDS: dict[str, list[str]] = {
    "Product":  ["product", "quality", "build", "material", "feature", "design",
                 "size", "colour", "color", "flimsy", "sturdy", "broken"],
    "Service":  ["service", "support", "staff", "rep", "agent", "call centre",
                 "call center", "helpful", "rude", "polite", "responsive"],
    "Price":    ["price", "cost", "expensive", "cheap", "value", "money",
                 "refund", "overpriced", "affordable", "worth"],
    "Delivery": ["delivery", "shipping", "shipped", "arrived", "package",
                 "packaging", "courier", "late", "fast", "slow", "wait"],
    "Support":  ["support", "help", "warranty", "return", "refund", "complaint",
                 "ticket", "response", "fix", "repair"],
}

# ---- Fintech palette -------------------------------------------------------
BRAND = "#4C6FFF"      # electric blue
BRAND2 = "#6D8BFF"     # lighter blue, for the hero gradient
INK = "#16161D"
MUTED = "#5B6172"
BODY = "#5B6172"
GREY = "#9AA0AE"
SOFT = "#F5F6FA"
LINE = "#EEF0F4"
FONT = "Manrope"
POS = "#16B364"      # green for positive
NEG = "#FF5A5F"      # coral for negative
WARN = "#FB8C00"
PLOT_TEMPLATE = "plotly_white"

st.set_page_config(
    page_title="Review sentiment",
    page_icon="💬",
    layout="wide",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"], .stMarkdown, button, input, textarea {{
        font-family: '{FONT}', system-ui, sans-serif;
    }}
    #MainMenu, header, footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }}

    .badge {{ display:inline-block; padding:5px 14px; border-radius:999px;
        font-size:12px; font-weight:700; letter-spacing:.6px; color:#fff; margin:0 4px; }}
    .badge.pos {{ background:{POS}; }}
    .badge.neg {{ background:{NEG}; }}

    .hero {{ background: linear-gradient(135deg, {BRAND} 0%, {BRAND2} 100%);
        border-radius: 24px; padding: 26px 30px 22px 30px; color:#fff;
        box-shadow: 0 18px 40px rgba(76,111,255,.26); }}
    .hero .brand {{ font-size:14px; font-weight:700; opacity:.92; display:flex;
        align-items:center; gap:8px; }}
    .hero .dot {{ width:9px; height:9px; border-radius:50%; background:#fff; display:inline-block; }}
    .hero .value {{ font-size:32px; font-weight:800; line-height:1.12; margin-top:14px; letter-spacing:-.5px; }}
    .hero .sub {{ font-size:15px; opacity:.95; margin-top:8px; max-width:660px; }}
    .chips {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }}
    .chip {{ background: rgba(255,255,255,.18); border-radius:12px; padding:9px 14px; font-size:13px; }}
    .chip b {{ font-size:16px; font-weight:800; display:block; }}

    .callout {{ border-radius:16px; padding:15px 18px; margin:6px 0 20px 0;
        font-size:15px; line-height:1.6; color:#3a3f4d; }}

    [data-testid="stMetric"] {{ background:#fff; border:1px solid #F0F1F5; border-radius:16px;
        padding:14px 18px; box-shadow:0 1px 3px rgba(20,22,30,.05), 0 8px 22px rgba(20,22,30,.04); }}
    [data-testid="stMetricValue"] {{ font-weight:800; color:{INK}; }}
    [data-testid="stMetricLabel"] p {{ font-weight:600; color:{BODY}; }}
    </style>

    <div class="hero">
      <div class="brand"><span class="dot"></span> Customer reviews &middot; sentiment</div>
      <div class="value">Reading the tone of what customers write</div>
      <div class="sub">A fine-tuned DistilBERT that reads sentiment the way a person would —
        coach a single review and see <i>why</i> it landed, or score a whole file and find
        where the complaints cluster.</div>
      <div class="chips">
        <span class="chip">model <b>DistilBERT</b></span>
        <span class="chip">SST-2 accuracy <b>~92%</b></span>
        <span class="chip">runs locally <b>no API</b></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


def note(text: str, tone: str = "neutral") -> None:
    bg = {"brand": "#EEF1FF", "good": "#E9FBF3", "warn": "#FFF6E9", "neutral": SOFT}[tone]
    bar = {"brand": BRAND, "good": POS, "warn": WARN, "neutral": GREY}[tone]
    st.markdown(
        f'<div class="callout" style="background:{bg};border-left:4px solid {bar};">{text}</div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=340, legend=False):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        margin=dict(l=8, r=8, t=30, b=8),
        font=dict(family=FONT, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=legend,
    )
    if fig.layout.title.text:
        fig.update_layout(title_font=dict(family=FONT, size=15, color=INK))
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    return fig


# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading DistilBERT (first launch downloads ~250 MB)...")
def load_model():
    # torch + transformers are imported here, not at module top: they take tens
    # of seconds to load and would otherwise freeze the page on every launch.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()
    return tok, mdl


tokenizer, model = load_model()


# --------------------------------------------------------------------------- #
def score(text: str) -> dict[str, Any]:
    """Return sentiment label, confidence, and per-token attribution."""
    import torch  # cheap after load_model() has run; kept off the import path
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
    colour = POS if s["label"] == "POSITIVE" else NEG
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
            number={"suffix": "%", "font": {"color": colour}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": colour},
                "bgcolor": "white", "borderwidth": 0,
                "steps": [
                    {"range": [0, 50], "color": "#FFECEC"},
                    {"range": [50, 100], "color": "#E9FBF3"},
                ],
            },
        ))
        st.plotly_chart(style_fig(fig, 200), use_container_width=True,
                        config={"displayModeBar": False})

    with b:
        attrs = sorted(s["attributions"], key=lambda x: x["delta"])
        plot_df = pd.DataFrame(attrs[:6] + attrs[-6:])
        plot_df = plot_df.sort_values("delta")
        fig = px.bar(
            plot_df, x="delta", y="token", orientation="h",
            color="delta", color_continuous_scale=[NEG, "#e8e8e8", POS],
            color_continuous_midpoint=0,
            labels={"delta": "Contribution to positive score", "token": ""},
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_fig(fig, 300), use_container_width=True,
                        config={"displayModeBar": False})


# --------------------------------------------------------------------------- #
# Batch + analytics helpers
# --------------------------------------------------------------------------- #
def score_batch(texts: list[str], batch_size: int = 16) -> list[float]:
    """Return positive-sentiment probability for each text."""
    import torch
    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = [str(t)[:1000] for t in texts[i : i + batch_size]]
            enc = tokenizer(chunk, return_tensors="pt", truncation=True,
                            padding=True, max_length=256)
            out = model(**enc)
            p = torch.softmax(out.logits, dim=-1)[:, 1].cpu().numpy().tolist()
            probs.extend(p)
    return probs


def label_aspects(text: str) -> list[str]:
    t = (text or "").lower()
    return [a for a, kws in ASPECT_KEYWORDS.items() if any(k in t for k in kws)]


def parse_uploaded_reviews(uploaded) -> pd.DataFrame:
    """Accept CSV or TXT. CSV looks for a 'text' or 'review' column; TXT = one
    review per line. Returns a DataFrame with at least a 'text' column."""
    name = uploaded.name.lower()
    raw = uploaded.read()
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw))
        text_col = next((c for c in df.columns
                         if c.lower() in {"text", "review", "comment", "body",
                                          "content", "message"}),
                        df.columns[0])
        df = df.rename(columns={text_col: "text"})
    else:
        text = raw.decode("utf-8", errors="ignore")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        df = pd.DataFrame({"text": lines})
    # If a date-like column already exists, normalise it; otherwise fabricate
    # one from the row index so the time-series view still works.
    date_col = next((c for c in df.columns
                     if c.lower() in {"date", "order_date", "review_date",
                                      "timestamp", "created_at"}), None)
    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if "date" not in df.columns or df["date"].isna().all():
        base = datetime.today() - timedelta(days=max(len(df) - 1, 0))
        df["date"] = [base + timedelta(days=i) for i in range(len(df))]
    return df[["text", "date"]].dropna(subset=["text"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    mode = st.radio(
        "Mode",
        ["Chat coach", "Batch analytics"],
        help="Chat for single-review coaching, Batch for uploading a file of reviews.",
    )
    st.markdown("---")
    st.subheader("How this works")
    st.markdown(
        "- Paste a review or message in the chat.\n"
        "- The model returns sentiment, confidence, and per-token contribution.\n"
        "- Then ask follow-ups:\n"
        "  - *\"Why is this negative?\"*\n"
        "  - *\"Rewrite it more positively\"*\n"
        "  - *\"Make it more formal\"*\n"
        "- Switch to **Batch analytics** to score a CSV/TXT of reviews."
    )
    st.markdown("---")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.markdown("---")
    st.caption("Model: `distilbert-base-uncased-finetuned-sst-2-english`")
    st.caption("≈ 250 MB local model — no API, no data leaves the Space.")

if mode == "Chat coach":
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            if turn.get("text"):
                st.markdown(turn["text"])
            if turn.get("score"):
                render_score(turn["score"], turn.get("scored_text", ""))

    if not st.session_state.history:
        note(
            'To get started, try pasting something like <i>"This product is absolutely '
            'fantastic, the best purchase of the year!"</i> — or any review of your own. '
            'Once it\'s scored, you can ask me to <i>rewrite</i> it or <i>explain</i> why '
            'it landed where it did.'
        )

    prompt = st.chat_input("Paste a review, ask me to rewrite, or ask why...")
    if prompt:
        st.session_state.history.append({"role": "user", "text": prompt, "scored_text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
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

else:
    # --------------------------------------------------------------------- #
    # Batch analytics mode
    # --------------------------------------------------------------------- #
    st.markdown("#### Batch sentiment analytics")
    st.caption(
        "Drag-drop a CSV (with a `text` or `review` column) or a TXT file "
        "(one review per line). The model scores every row, tags aspects, "
        "and produces a 30-day sentiment trend plus a recommended action."
    )

    upl = st.file_uploader(
        "Upload reviews (CSV or TXT)", type=["csv", "txt"],
        accept_multiple_files=False,
    )

    use_sample = st.checkbox(
        "Use the bundled sample (data/reviews.csv, 500 rows)", value=upl is None,
    )

    df_in: pd.DataFrame | None = None
    if upl is not None:
        df_in = parse_uploaded_reviews(upl)
    elif use_sample:
        try:
            sample = pd.read_csv("data/reviews.csv").head(500)
            text_col = next((c for c in sample.columns
                             if c.lower() in {"text", "review"}), sample.columns[0])
            sample = sample.rename(columns={text_col: "text"})
            base = datetime.today() - timedelta(days=len(sample) - 1)
            sample["date"] = [base + timedelta(days=i) for i in range(len(sample))]
            df_in = sample[["text", "date"]].reset_index(drop=True)
        except FileNotFoundError:
            st.warning("Sample file data/reviews.csv not found. Upload your own file.")

    if df_in is not None and len(df_in) > 0:
        n = len(df_in)
        if n > 2000:
            st.warning(f"File has {n:,} rows — scoring the first 2,000 to keep the demo snappy.")
            df_in = df_in.head(2000)

        with st.spinner(f"Scoring {len(df_in):,} reviews with DistilBERT..."):
            df_in = df_in.copy()
            df_in["pos_prob"] = score_batch(df_in["text"].astype(str).tolist())
            df_in["sentiment"] = np.where(df_in["pos_prob"] >= 0.5, "POSITIVE", "NEGATIVE")
            df_in["aspects"] = df_in["text"].astype(str).apply(label_aspects)

        # -- Top KPI strip -------------------------------------------------
        pos_share = float((df_in["sentiment"] == "POSITIVE").mean())
        neg_share = 1 - pos_share
        avg_score = float(df_in["pos_prob"].mean())
        a, b, c, d = st.columns(4)
        a.metric("Reviews scored", f"{len(df_in):,}")
        b.metric("Positive share", f"{pos_share*100:.1f}%")
        c.metric("Negative share", f"{neg_share*100:.1f}%")
        d.metric("Mean confidence (+)", f"{avg_score*100:.1f}%")

        # -- Aggregate breakdown ------------------------------------------
        st.markdown("##### Sentiment distribution")
        breakdown = (df_in["sentiment"].value_counts(normalize=True) * 100).reset_index()
        breakdown.columns = ["sentiment", "share"]
        fig = px.bar(
            breakdown, x="sentiment", y="share", color="sentiment",
            color_discrete_map={"POSITIVE": POS, "NEGATIVE": NEG},
            text=breakdown["share"].map(lambda x: f"{x:.1f}%"),
        )
        fig.update_layout(yaxis_title="Share of reviews (%)", xaxis_title="",
                          showlegend=False)
        st.plotly_chart(style_fig(fig, 320), use_container_width=True)
        dominant = "positive" if pos_share >= 0.5 else "negative"
        tone_tail = ("Customers are mostly happy here, so the useful move is to work out "
                     "what's going right and amplify it."
                     if pos_share >= 0.5 else
                     "Most reviews lean negative, so it's worth drilling into the aspects "
                     "below to see where to intervene first.")
        note(
            f"The batch reads <b>{dominant}</b> overall — <b>{pos_share*100:.1f}%</b> "
            f"positive against <b>{neg_share*100:.1f}%</b> negative. {tone_tail}"
        )

        # -- Aspect drill-down --------------------------------------------
        st.markdown("##### Aspect drill-down")
        st.caption("Sentiment per business aspect. A review can touch multiple aspects.")

        aspect_rows = []
        for _, row in df_in.iterrows():
            for a in row["aspects"]:
                aspect_rows.append({"aspect": a, "pos_prob": row["pos_prob"],
                                    "sentiment": row["sentiment"]})
        aspect_df = pd.DataFrame(aspect_rows)

        if len(aspect_df) == 0:
            st.info("No reviews mentioned any of the tracked aspects.")
            top_neg_aspect = None
        else:
            grp = aspect_df.groupby("aspect").agg(
                mentions=("aspect", "size"),
                pos_share=("sentiment", lambda s: (s == "POSITIVE").mean()),
                mean_score=("pos_prob", "mean"),
            ).reset_index().sort_values("mean_score")
            grp["neg_share"] = 1 - grp["pos_share"]

            fig = px.bar(
                grp, x="mean_score", y="aspect", orientation="h",
                color="mean_score",
                color_continuous_scale=[NEG, WARN, POS],
                range_color=[0, 1],
                text=grp["mean_score"].map(lambda x: f"{x*100:.0f}%"),
                hover_data={"mentions": True, "pos_share": ":.0%",
                            "neg_share": ":.0%", "mean_score": False},
                labels={"mean_score": "Mean positive score", "aspect": ""},
            )
            fig.update_layout(coloraxis_showscale=False, xaxis_tickformat=".0%")
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
            top_neg_aspect = grp.iloc[0]["aspect"]
            top_neg_share = float(grp.iloc[0]["neg_share"])
            note(
                f"Of the tracked aspects, <b>{top_neg_aspect}</b> has the lowest sentiment "
                f"— <b>{top_neg_share*100:.0f}%</b> of the mentions are negative. That's "
                f"usually where a fix pays back fastest, since it's the thing customers "
                f"are already telling you about."
            )

        # -- Time-series trend --------------------------------------------
        st.markdown("##### 30-day sentiment trend")
        st.caption("Rolling 30-day mean positive score. If your file has no date column, we use the row order as a synthetic timeline so the demo still works.")
        ts = (df_in.set_index("date")["pos_prob"]
                    .sort_index()
                    .rolling("30D", min_periods=1)
                    .mean()
                    .reset_index())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts["date"], y=ts["pos_prob"], mode="lines",
            line=dict(color=POS, width=3), fill="tozeroy",
            fillcolor="rgba(75,158,122,0.16)", name="30-day rolling sentiment",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color=MUTED,
                      annotation_text="Neutral", annotation_position="right")
        fig.update_layout(
            yaxis=dict(range=[0, 1], tickformat=".0%",
                       title="Positive-sentiment share"),
            xaxis=dict(title=""),
        )
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

        if len(ts) > 1:
            start, end = float(ts["pos_prob"].iloc[0]), float(ts["pos_prob"].iloc[-1])
            delta = end - start
            direction = "improving" if delta > 0.02 else ("worsening" if delta < -0.02 else "flat")
            trend_tail = (
                "It's heading the right way, so the sensible thing is to keep doing whatever's working."
                if direction == "improving" else
                "It's slipping — worth tracing which aspect started dragging, and roughly when."
                if direction == "worsening" else
                "It's fairly flat, but a small, well-aimed intervention can still tip it."
            )
            note(
                f"Rolling sentiment has moved from <b>{start*100:.0f}%</b> to "
                f"<b>{end*100:.0f}%</b> across the window — broadly <b>{direction}</b>. "
                f"{trend_tail}"
            )

        # -- Business action panel ----------------------------------------
        st.markdown("##### Recommended business actions")
        action_lib = {
            "Product":  "Retrain QA on the failure modes named in complaints; consider a targeted product refresh or recall communication.",
            "Service":  "Run a call-centre coaching cycle on the top 10 negative tickets; tighten the SLA for first response.",
            "Price":    "Audit the price ladder against competitors; consider a value-bundle or loyalty discount before more customers churn.",
            "Delivery": "Review the courier mix in the worst-performing lanes; introduce SMS proactive ETA updates.",
            "Support":  "Re-time the support escalation matrix; auto-route refund / warranty tickets to a senior agent.",
        }
        if top_neg_aspect:
            body = action_lib.get(top_neg_aspect, "investigate it further before acting.")
            note(
                f"The aspect dragging hardest is <b>{top_neg_aspect}</b>, so that's where "
                f"I'd start. In practice that means: {body[0].lower() + body[1:]}"
            )
        else:
            st.info("No dominant negative aspect detected. Continue monitoring.")

        # -- Per-row table -------------------------------------------------
        st.markdown("##### Scored reviews (top 100)")
        show = df_in.head(100).copy()
        show["aspects"] = show["aspects"].apply(lambda a: ", ".join(a) if a else "—")
        show["pos_prob"] = show["pos_prob"].map(lambda x: f"{x*100:.1f}%")
        st.dataframe(
            show[["date", "text", "sentiment", "pos_prob", "aspects"]],
            use_container_width=True, hide_index=True, height=420,
        )

        csv_bytes = df_in.assign(
            aspects=df_in["aspects"].apply(lambda a: "|".join(a))
        ).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scored CSV", csv_bytes, file_name="scored_reviews.csv",
            mime="text/csv",
        )
    else:
        st.info("Upload a file or tick the sample-data checkbox to begin.")
