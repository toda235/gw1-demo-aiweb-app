# app.py
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/japanese-stablelm-instruct-gamma-7b")
    model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/japanese-stablelm-instruct-gamma-7b",
    torch_dtype="auto",
    )
    return tokenizer, model

def build_prompt(user_query, inputs="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [": \n" + user_query, ": \n"]
    if inputs:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + inputs)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

tokenizer, model = tokenizer_model()



# 入力欄
text = st.text_area("テキストを入力", "ごめん、明日の会議だけど、資料がまだできてないからリスケでお願い。")
user_inputs = {
    "user_query": "与えられた文章を、ビジネスメール風の丁寧で簡潔な文章に変換してください。",
    "inputs": text
}
prompt = build_prompt(**user_inputs)

if torch.cuda.is_available():
    model = model.to("cuda")

if st.button("分析する"):
    with st.spinner("分析中..."):
        with torch.no_grad():
            input_ids = tokenizer.encode(
                prompt, 
                add_special_tokens=True, 
                return_tensors="pt"
            )

            tokens = model.generate(
                input_ids.to(device=model.device),
                max_new_tokens=256,
                temperature=1,
                top_p=0.95,
                do_sample=True,
            )

            out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        st.success(out)
        print(out)