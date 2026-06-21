import streamlit as st
import pandas as pd
from model import predict_review

# page config
st.set_page_config(page_title="FakeReview Detector", page_icon="🔍", layout="centered")

# title section
st.title("🔍 FakeReview Detector")
st.markdown("Check if a product review is **genuine** or **AI/fake generated** — trained on 40,000+ real reviews.")
st.markdown("---")

# single review checker
st.subheader("📝 Check a Single Review")
user_review = st.text_area("Paste a review here", height=150, placeholder="e.g. This product is amazing! Works perfectly and arrived on time...")

if st.button("Analyze Review"):
    if user_review.strip() == "":
        st.warning("Please enter a review first.")
    else:
        prediction, confidence = predict_review(user_review)
        
        if prediction == "CG":
            st.error(f"🚨 Fake Review Detected — {confidence}% confidence")
        else:
            st.success(f"✅ Genuine Review — {confidence}% confidence")

st.markdown("---")

# batch csv upload
st.subheader("📂 Check Multiple Reviews via CSV")
st.markdown("Upload a CSV file with a column named `text_` containing reviews.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    if 'text_' not in df.columns:
        st.error("CSV must have a column named 'text_'")
    else:
        st.info("Analyzing reviews... please wait")
        df['prediction'] = df['text_'].apply(lambda x: predict_review(x)[0])
        df['confidence'] = df['text_'].apply(lambda x: predict_review(x)[1])
        df['result'] = df['prediction'].apply(lambda x: "🚨 Fake" if x == "CG" else "✅ Genuine")
        
        st.dataframe(df[['text_', 'result', 'confidence']])
        
        # summary
        total = len(df)
        fake_count = (df['prediction'] == 'CG').sum()
        genuine_count = total - fake_count
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", total)
        col2.metric("Fake Reviews", fake_count)
        col3.metric("Genuine Reviews", genuine_count)

st.markdown("---")
st.caption("Built by Akshit Kapoor | B.Tech AI & Data Science | Delhi NCR")