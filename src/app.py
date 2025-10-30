# app.py
import streamlit as st
from transformers import pipeline

st.title("🔍 感情分析アプリ")
st.write("テキストを入力して感情を分析します。")


@st.cache_resource
def load_model():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")


analyzer = load_model()

# 入力欄
text = st.text_area("テキストを入力", "今日はとても良い気分です！")

if st.button("分析する"):
    with st.spinner("分析中..."):
        result = analyzer(text)[0]
        st.success(f"ラベル: {result['label']}, スコア: {result['score']:.3f}")
