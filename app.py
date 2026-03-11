import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px # Better for professional interactivity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from utils import get_comments_from_post, clean_text, analyze_sentiment, get_word_frequency
import plotly.express as px

ART_PALETTE = ["#8e2de2", "#4a00e0", "#ff0066", "#00d2ff", "#3a7bd5"]

def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return np.random.choice(ART_PALETTE)
# Page Config
st.set_page_config(page_title="Instagram Sentiment Pro", layout="wide", initial_sidebar_state="collapsed")

# Inject Custom CSS for Glassmorphism and modern UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        background: linear-gradient(to right, #8e2de2, #4a00e0);
        color: white;
        border: none;
        padding: 12px;
        font-weight: bold;
    }
    .reportview-container .main .block-container {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.7);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    </style>
    """, unsafe_allow_html=True)

# Header Section
st.title("💜 Instagram Sentiment Analyzer Pro")
st.markdown("---")

# Organized Input Layout
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    post_url = st.text_input("🔗 Post URL", placeholder="https://www.instagram.com/p/...")
with col2:
    username = st.text_input("👤 Username", placeholder="Enter username")
with col3:
    password = st.text_input("🔑 Password", type="password", placeholder="Enter password")

with st.spinner("Fetching comments..."):
    comments, error = get_comments_from_post(post_url, username, password)
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Tip: Try logging into Instagram on your browser first, or wait a few minutes before retrying.")
    elif not comments:
        st.warning("⚠️ No comments found or post is private.")
    else:
                # Data Processing
                df = pd.DataFrame({"comment": comments})
                df['cleaned'] = df['comment'].apply(clean_text)
                df[['sentiment', 'score']] = df['cleaned'].apply(lambda x: pd.Series(analyze_sentiment(x)))

                # Visual Dashboard Section
                st.success(f"Analysis complete for {len(comments)} comments!")
                
                # Metric Cards
                m1, m2, m3 = st.columns(3)
                m1.metric("Positive", f"{len(df[df['sentiment']=='Positive'])}")
                m2.metric("Neutral", f"{len(df[df['sentiment']=='Neutral'])}")
                m3.metric("Negative", f"{len(df[df['sentiment']=='Negative'])}")

                # Advanced Visualization Row
                v1, v2 = st.columns(2)
                
                with v1:
                    st.subheader("📊 Sentiment Breakdown")
                    fig = px.pie(df, names='sentiment', color='sentiment',
                                 color_discrete_map={'Positive':'#00CC96', 'Neutral':'#636EFA', 'Negative':'#EF553B'},
                                 hole=.4)
                    st.plotly_chart(fig, use_container_width=True)

                with v2:
                    st.subheader("☁️ Keyword Cloud")
                    text = " ".join(df['cleaned'])
                    wc = WordCloud(width=800,
                                   height=450,
                                   background_color='white',
                                   colormap='magma',
                                   max_words = 100,
                                   color_func= color_func,
                                   font_path=None
                                   ).generate(text)
                    fig_wc, ax_wc = plt.subplots()
                    ax_wc.imshow(wc, interpolation='bilinear')
                    ax_wc.axis('off')
                    st.pyplot(fig_wc)

                # Adding a 'Recent Comments' table with sentiment tags
                st.subheader("💬 Recent Comment Insights")
                st.dataframe(
                    df[['comment', 'sentiment', 'score']].sort_values(by='score', ascending=False),
                    column_config={
                        "sentiment": st.column_config.TextColumn("Sentiment Tag"),
                        "score": st.column_config.ProgressColumn("Polarity Score", min_value=-1, max_value=1)
                },
                use_container_width=True,
                hide_index=True
            )   
                # Detailed Data Table
                with st.expander("📂 View Raw Sentiment Data"):
                    st.dataframe(df[['comment', 'sentiment', 'score']], use_container_width=True)