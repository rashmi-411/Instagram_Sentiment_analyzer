import instaloader
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from collections import Counter
import pandas as pd

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def get_comments_from_post(url, username, password):
    try:
        L = instaloader.Instaloader()
        shortcode = url.split("/")[-2]
        L.login(username, password)
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        comments = [comment.text for comment in post.get_comments()]
        return comments, None
    except Exception as e:
        #print(f"Error fetching comments: {e}")
        return [],str(e)

'''def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    text = text.lower()
    return " ".join([w for w in text.split() if w not in stop_words])'''

def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    text = text.lower()
    return " ".join([w for w in text.split() if w not in stop_words])
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        sentiment = 'Positive'
    elif score <= -0.05:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'
    return sentiment, score

def get_word_frequency(comments, top_n=10):
    all_words = []
    for comment in comments:
        words = clean_text(comment).split()
        all_words.extend(words)
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)