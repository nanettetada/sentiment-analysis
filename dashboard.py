"""ReviewCoach — a conversational AI that analyses sentiment, explains why,
and helps you rewrite text. Powered by a fine-tuned DistilBERT.

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
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
# Batch + analytics helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_batch(texts: list[str], batch_size: int = 16) -> list[float]:
    """Return positive-sentiment probability for each text."""
    probs: list[float] = []
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

else:
    # --------------------------------------------------------------------- #
    # Batch analytics mode
    # --------------------------------------------------------------------- #
    st.subheader(":bar_chart: Batch sentiment analytics")
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
            color_discrete_map={"POSITIVE": "#27AE60", "NEGATIVE": "#E74C3C"},
            text=breakdown["share"].map(lambda x: f"{x:.1f}%"),
        )
        fig.update_layout(yaxis_title="Share of reviews (%)", xaxis_title="",
                          height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        dominant = "positive" if pos_share >= 0.5 else "negative"
        st.markdown(
            '<div class="insight"><div class="head">OVERALL TONE</div>'
            f'<div class="body">The batch reads <b>{dominant}</b> overall — '
            f'<b>{pos_share*100:.1f}%</b> positive vs <b>{neg_share*100:.1f}%</b> negative. '
            f'{"Customers are mostly happy; focus on amplifying the wins." if pos_share >= 0.5 else "Most reviews lean negative — drill into aspects below to see where to intervene first."}'
            '</div></div>',
            unsafe_allow_html=True,
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
                color_continuous_scale=["#E74C3C", "#F39C12", "#27AE60"],
                range_color=[0, 1],
                text=grp["mean_score"].map(lambda x: f"{x*100:.0f}%"),
                hover_data={"mentions": True, "pos_share": ":.0%",
                            "neg_share": ":.0%", "mean_score": False},
                labels={"mean_score": "Mean positive score", "aspect": ""},
            )
            fig.update_layout(coloraxis_showscale=False, height=320,
                              margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            top_neg_aspect = grp.iloc[0]["aspect"]
            top_neg_share = float(grp.iloc[0]["neg_share"])
            st.markdown(
                '<div class="insight"><div class="head">TOP COMPLAINT AREA</div>'
                f'<div class="body">Out of the tracked aspects, '
                f'<b>{top_neg_aspect}</b> has the lowest sentiment '
                f'(<b>{top_neg_share*100:.0f}%</b> of mentions are negative). '
                'This is where retention spend pays back fastest.'
                '</div></div>',
                unsafe_allow_html=True,
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
            line=dict(color="#F39C12", width=3), fill="tozeroy",
            fillcolor="rgba(243,156,18,0.18)", name="30-day rolling sentiment",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="#95A5A6",
                      annotation_text="Neutral", annotation_position="right")
        fig.update_layout(
            yaxis=dict(range=[0, 1], tickformat=".0%",
                       title="Positive-sentiment share"),
            xaxis=dict(title=""),
            height=340, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        if len(ts) > 1:
            start, end = float(ts["pos_prob"].iloc[0]), float(ts["pos_prob"].iloc[-1])
            delta = end - start
            direction = "improving" if delta > 0.02 else ("worsening" if delta < -0.02 else "flat")
            st.markdown(
                '<div class="insight"><div class="head">TREND</div>'
                f'<div class="body">Rolling sentiment has moved from '
                f'<b>{start*100:.0f}%</b> to <b>{end*100:.0f}%</b> '
                f'over the window — <b>{direction}</b>. '
                f'{"Keep doing what is working." if direction == "improving" else ("Investigate which aspect started slipping and when." if direction == "worsening" else "Stable book — small interventions can still tilt this.")}'
                '</div></div>',
                unsafe_allow_html=True,
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
            head = f"PRIORITY ACTION: {top_neg_aspect.upper()}"
            body = action_lib.get(top_neg_aspect, "Investigate further.")
            st.markdown(
                f'<div class="insight"><div class="head">{head}</div>'
                f'<div class="body">Top complaint area: <b>{top_neg_aspect}</b> &mdash; {body}</div></div>',
                unsafe_allow_html=True,
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
