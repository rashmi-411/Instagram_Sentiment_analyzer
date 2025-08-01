import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from utils import get_comments_from_post, clean_text, analyze_sentiment, get_word_frequency

st.set_page_config(page_title="Instagram Comment Sentiment", layout="wide")
st.title("📊 Instagram Post Comment Sentiment Analyzer")

# Input fields
post_url = st.text_input("🔗 Enter Instagram Post URL:")
username = st.text_input("👤 Instagram Username:")
password = st.text_input("🔑 Instagram Password:", type="password")

if st.button("Analyze"):
    if not (post_url and username and password):
        st.warning("⚠️ Please fill in all fields.")
    else:
        with st.spinner("Fetching comments and analyzing..."):
            try:
                # Fetch comments
                comments = get_comments_from_post(post_url, username, password)
                if not comments:
                    st.error("No comments found or failed to fetch comments.")
                else:
                    # Clean and analyze
                    cleaned = [clean_text(c) for c in comments]
                    results = [analyze_sentiment(c) for c in cleaned]

                    df = pd.DataFrame({
                        "Original Comment": comments,
                        "Cleaned": cleaned,
                        "Sentiment": [r[0] for r in results],
                        "Score": [r[1] for r in results]
                    })

                    st.success(f"✅ Fetched {len(comments)} comments!")

                    # Sentiment Distribution Plot
                    st.subheader("📈 Sentiment Distribution")
                    fig1, ax1 = plt.subplots()
                    sns.countplot(data=df, x="Sentiment", ax=ax1, palette="Set2")
                    ax1.set_ylabel("Count")
                    st.pyplot(fig1)

                    # Frequent Words Plot
                    st.subheader("📝 Most Frequent Words")
                    word_counts = get_word_frequency(comments)
                    if word_counts:
                        word_df = pd.DataFrame(word_counts, columns=["Word", "Frequency"])
                        fig2, ax2 = plt.subplots()
                        sns.barplot(data=word_df, x="Frequency", y="Word", ax=ax2, palette="Blues_d")
                        st.pyplot(fig2)
                    else:
                        st.info("No frequent words to display.")

                    # Word Cloud
                    st.subheader("☁️ Word Cloud")
                    if any(df["Cleaned"]):
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(df["Cleaned"]))
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
                        ax_wc.imshow(wordcloud, interpolation='bilinear')
                        ax_wc.axis("off")
                        st.pyplot(fig_wc)
                    else:
                        st.info("Not enough text for word cloud.")

                    # Display DataFrame
                    with st.expander("🔍 See Comment Analysis Data"):
                        st.dataframe(df)

            except Exception as e:
                st.error(f"❌ Error: {e}")



