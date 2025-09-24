# emotion_analyzer_app_final.py
import io, os, base64, json
from flask import Flask, request, render_template_string, send_file
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from rake_nltk import Rake
from langdetect import detect, DetectorFactory
from better_profanity import profanity

DetectorFactory.seed = 0
profanity.load_censor_words()

app = Flask(__name__)
analyzer_vader = SentimentIntensityAnalyzer()
rake = Rake()

# ---------- تحليل النص ----------
def analyze_text(text):
    text = str(text).strip()
    if not text:
        return {}
    # اللغة
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    # TextBlob
    blob = TextBlob(text)
    tb_polarity = round(blob.sentiment.polarity, 4)
    tb_subjectivity = round(blob.sentiment.subjectivity, 4)
    # Vader
    vs = analyzer_vader.polarity_scores(text)
    vader_compound = round(vs.get("compound", 0), 4)
    # مشاعر NRC
    nrc = NRCLex(text)
    raw_emotions = nrc.raw_emotion_scores
    total = sum(raw_emotions.values()) if raw_emotions else 0
    emotions_norm = {k: round(v / total, 4) if total>0 else 0.0 for k,v in raw_emotions.items()}
    # كلمات مفتاحية
    rake.extract_keywords_from_text(text)
    keywords = rake.get_ranked_phrases()[:8]
    # ألفاظ نابية
    contains_profanity = profanity.contains_profanity(text)
    censored = profanity.censor(text) if contains_profanity else text
    # التقييم النهائي
    if vader_compound >= 0.5 or tb_polarity >= 0.5:
        final_label = "Strong Positive"
    elif vader_compound > 0.2 or tb_polarity > 0.2:
        final_label = "Positive"
    elif vader_compound < -0.5 or tb_polarity <= -0.5:
        final_label = "Strong Negative"
    elif vader_compound < -0.2 or tb_polarity < -0.2:
        final_label = "Negative"
    else:
        final_label = "Neutral/Mixed"
    return {
        "text": text,
        "language": lang,
        "textblob_polarity": tb_polarity,
        "textblob_subjectivity": tb_subjectivity,
        "vader_compound": vader_compound,
        "nrc_emotions": emotions_norm,
        "keywords": keywords,
        "contains_profanity": contains_profanity,
        "censored_text": censored,
        "final_label": final_label
    }

# ---------- واجهة HTML ----------
INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Emotion Analyzer — Final</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {background: linear-gradient(135deg,#0f172a,#0f4c81); color:#fff; min-height:100vh;}
  .card {background: rgba(255,255,255,0.07); border:none; border-radius:16px; box-shadow:0 8px 25px rgba(0,0,0,0.5);}
  textarea {background:rgba(255,255,255,0.05); color:#fff; border-radius:10px;}
  .card p, .card h4, .card b {color: #ffffff !important;}
</style>
</head>
<body>
<div class="container py-5">
  <h1 class="mb-4 text-center">✨ Emotion Analyzer</h1>
  <div class="row g-4">
    <!-- إدخال النص -->
    <div class="col-lg-6">
      <div class="card p-4">
        <h4>أدخل نص للتحليل</h4>
        <form action="/analyze" method="post">
          <textarea name="text" rows="6" class="form-control mb-3" placeholder="اكتب النص هنا ..."></textarea>
          <button class="btn btn-primary w-100">تحليل</button>
        </form>
      </div>
    </div>
    <!-- النتائج + الرسم -->
    <div class="col-lg-6">
      <div class="card p-4">
        <h4>النتيجة</h4>
        {% if res %}
          <p><b>Final Label:</b> {{res.final_label}}</p>
          <p><b>Language:</b> {{res.language}}</p>
          <p><b>VADER:</b> {{res.vader_compound}}</p>
          <p><b>TextBlob:</b> {{res.textblob_polarity}}</p>
          <p><b>Subjectivity:</b> {{res.textblob_subjectivity}}</p>
          <p><b>Profanity:</b> {% if res.contains_profanity %}<span class="text-danger">Yes</span>{% else %}<span class="text-success">No</span>{% endif %}</p>
          <p><b>Keywords:</b>
            {% for k in res.keywords %}
              <span class="badge bg-light text-dark">{{k}}</span>
            {% endfor %}
          </p>
          <canvas id="emotionChart" height="250"></canvas>
        {% else %}
          <p class="text-muted">لم يتم إدخال نص بعد.</p>
        {% endif %}
      </div>
    </div>
  </div>
</div>
{% if res %}
<script>
  const emotions = {{ emotions_json|safe }};
  const ctx = document.getElementById('emotionChart').getContext('2d');
  new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: Object.keys(emotions),
      datasets: [{
        data: Object.values(emotions),
        backgroundColor: [
          '#ff6384','#36a2eb','#ffcd56','#4bc0c0',
          '#9966ff','#ff9f40','#c9cbcf','#00ff99'
        ],
        borderWidth: 1
      }]
    },
    options: {responsive: true, plugins:{legend:{labels:{color:'#fff'}}}}
  });
</script>
{% endif %}
</body>
</html>
"""

# ---------- المسارات ----------
@app.route('/', methods=['GET'])
def index():
    return render_template_string(INDEX_HTML, res=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text','')
    res = analyze_text(text)
    all_emotions = {k: res.get('nrc_emotions',{}).get(k,0.0) for k in
                    ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]}
    return render_template_string(INDEX_HTML, res=res, emotions_json=json.dumps(all_emotions))

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
