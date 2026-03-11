import instaloader
import re
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
from collections import Counter

# make sure nltk stopwords are downloaded
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


def get_comments_from_post(url, username, password):
    L = instaloader.Instaloader()
    
    # Try to load an existing session to avoid getting flagged
    session_file = os.path.join(os.getcwd(), f"session-{username}")
    try:
        if os.path.exists(session_file):
            L.load_session_from_file(username, filename=session_file)
            print("Loaded session from file.")
        else:
            L.login(username, password)
            L.save_session_to_file(filename=session_file)
            print("New login successful; session saved.")
    except Exception as e:
        return [], f"Login failed: {str(e)}"

    try:
        # Extract shortcode accurately
        shortcode_match = re.search(r'/p/([^/]+)/', url)
        shortcode = shortcode_match.group(1) if shortcode_match else url.split("/")[-2]
        
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        
        # Scrape comments (limiting to avoid rate limits) 
        comments = []
        for i, comment in enumerate(post.get_comments()):
            comments.append(comment.text)
            if i >= 49: break # Start with a small number to test
            
        return comments, None
    except Exception as e:
        return [], f"Instagram challenged the request: {str(e)}"


def clean_text(text: str) -> str:
    """Basic text cleaning: remove URLs, mentions, emojis, punctuation, and stopwords."""
    if not isinstance(text, str):
        return ""
    # lower-case
    txt = text.lower()
    # strip URLs
    txt = re.sub(r"https?://\S+", "", txt)
    # remove @mentions and hashtags
    txt = re.sub(r"[@#]\w+", "", txt)
    # remove non-alphanumeric characters
    txt = re.sub(r"[^a-z0-9\s]", "", txt)
    # tokenize and drop stopwords
    tokens = txt.split()
    stops = set(stopwords.words("english"))
    tokens = [t for t in tokens if t not in stops]
    return " ".join(tokens)


def analyze_sentiment(text: str):
    """Return a tuple (label, score) using Vader sentiment intensity."""
    if not isinstance(text, str) or text.strip() == "":
        return "Neutral", 0.0
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores.get("compound", 0.0)
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return label, compound


def get_word_frequency(texts):
    """Given an iterable of strings return Counter of word frequencies after cleaning."""
    if texts is None:
        return Counter()
    all_words = []
    for t in texts:
        cleaned = clean_text(t)
        all_words.extend(cleaned.split())
    return Counter(all_words)
