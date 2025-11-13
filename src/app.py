# app.py
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-128k-instruct", trust_remote_code=False)
    return tokenizer, model

tokenizer, model = tokenizer_model()

# 入力欄
text = st.text_area("テキストを入力", "こんにちは\nこの前頼まれたやつだけど、間に合わないから遅れるわ\nごめんね")
chat = [
    {"role": "system", "content":"入力された文章を、礼儀正しいビジネスメール風の本文に最低限書き換えてください。"},
    {"role": "user", "content": text},
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

model.to("cuda")

if st.button("分析する"):
    with st.spinner("分析中..."):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(
                token_ids.to(model.device),
                do_sample=True,
                temperature=0.6,
                max_new_tokens=512,
            )

        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        st.success(output)