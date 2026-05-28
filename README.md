<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:F39C12,100:7E4F00&height=200&section=header&text=ReviewCoach&fontSize=58&fontColor=ffffff&fontAlignY=38&animation=fadeIn&desc=Conversational%20AI%20for%20sentiment%20%2B%20rewriting&descSize=18&descAlignY=68" />

<a href="https://github.com/nanettetada">
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=22&duration=3500&pause=800&color=F39C12&center=true&vCenter=true&width=720&lines=Chat+to+a+sentiment+model;DistilBERT+running+locally+in-app;Score+%2B+explain+%2B+rewrite%2C+turn+by+turn" />
</a>

<p>
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
<img src="https://img.shields.io/badge/DistilBERT-FF6F61?style=for-the-badge" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
</p>

<a href="https://huggingface.co/spaces/NanetteTada/review-sentiment-engine"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Open%20Live%20Demo-FFD21E?style=for-the-badge" /></a>

</div>

---

<p align="center">
  <img src="docs/preview.png" alt="Dashboard preview" width="900">
</p>


## Why I built this

A sentiment classifier on its own is a science-fair project. **ReviewCoach** is the same model wrapped in a conversational interface that actually does something useful — it scores any text you paste, explains the per-token contribution, and helps you rewrite it more positively, more critically, or more formally.

The model is a fine-tuned **DistilBERT** running locally inside the app (no API keys, no data leaves the environment). The conversation layer is a small intent classifier I wrote on top.

## What it can do

| Ask it... | And it... |
|---|---|
| *"This product is amazing!"* | Scores it, shows a confidence gauge, and lists the top words pushing it positive. |
| *"Why did you say that?"* | Highlights the per-token contribution from the previous score — both positive and negative. |
| *"Rewrite that more positively"* | Swaps the negative-leaning tokens for softer alternatives and re-scores the result. |
| *"Make it more formal"* | Replaces casual words (*gonna*, *stuff*, *guys*) with formal equivalents. |
| *"Rewrite it more harshly"* | Flips positive-leaning tokens to critical equivalents. |

## How it works

```
   User message
         │
         ▼
   intent classifier   ──►  GREET / HELP / SCORE / EXPLAIN / REWRITE_*
         │
         ▼
   DistilBERT (sst-2)   ──►  sentiment label + confidence + per-token attribution
         │
         ▼
   response builder    ──►  conversational reply + interactive chart
         │
         ▼
   Streamlit chat UI   ──►  turn-by-turn conversation with score gauges
```

The per-token attribution is done by **masking each token in turn and observing how the positive-class probability shifts** — a lightweight version of LIME / occlusion sensitivity.

## Run it yourself

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

First launch downloads DistilBERT (~250 MB). After that, every score is instant.

There's also a classical-ML notebook (`sentiment_analysis.ipynb`) comparing TF-IDF + Logistic Regression / SVM / Naive Bayes side-by-side with DistilBERT — the classical pipeline matches DistilBERT on accuracy at roughly **100× the serving cost**, which was the punchline of the original project.

## Try these conversations

1. **Live scoring**
   - You: *"Honestly this gadget is the best thing I've bought all year, build quality is incredible and delivery was lightning fast."*
   - Bot: 🟢 POSITIVE, 99% — and shows you the top contributing tokens.

2. **Token attribution**
   - You: *"Terrible product, broke within a week, never buying from this brand again."*
   - You: *"Why is this negative?"*
   - Bot: *Pushing it negative: `terrible`, `broke`, `never`.*

3. **Rewriting**
   - You: *"Their support is useless and slow. Total waste of money."*
   - You: *"Rewrite that to be more positive"*
   - Bot: *"Their support is limited and taking time. Total investment that didn't land."* — then shows the new score.

## What I'd build next

- Swap the rule-based rewrites for a real seq2seq model (T5-small or Flan-T5) so the suggestions are grammatically tighter.
- Add aspect-based sentiment: separate the *price* complaints from the *quality* complaints in a long review.
- Multi-language support — Shona / English code-switching is the realistic Zim case.
- Add an "intent" badge inline so the user can see what the bot interpreted from their last message.

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:F39C12,100:7E4F00&height=100&section=footer" />

Built by <b>Tadaishe Maumbe</b> · <a href="https://github.com/nanettetada">@nanettetada</a> · <a href="mailto:maumbetadaishe@gmail.com">email</a>

</div>
