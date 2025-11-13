# app.py
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@st.cache_resource
def tokenizer_model():
    model_name = "4bit/ELYZA-japanese-Llama-2-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    return tokenizer, model

tokenizer, model = tokenizer_model()

# 入力欄
text = st.text_area("テキストを入力", "こんにちは\nこの前頼まれた件だけど、間に合わないから遅れるわ\nごめんね")
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "入力された文章を礼儀正しいビジネスメール風の本文に書き換えてください。出力："

if torch.cuda.is_available():
    model = model.to("cuda")

prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)

if st.button("分析する"):
    with st.spinner("分析中..."):
        with torch.no_grad():
            token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

            output_ids = model.generate(
                token_ids.to(model.device),
                max_new_tokens=256,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        st.success(output)
