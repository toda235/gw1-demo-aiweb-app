# app.py
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from streamlit_option_menu import option_menu # インポート
from streamlit_toggle import st_toggle_switch
from huggingface_hub import InferenceClient

@st.cache_resource
def tokenizer_model():
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/japanese-stablelm-instruct-gamma-7b")
    model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/japanese-stablelm-instruct-gamma-7b",
    torch_dtype="auto",
    )
    pipe = pipeline("text-classification", model="namopanda/deberta-v3-base-japanese-politeness", device=-1)
    return tokenizer, model, pipe

st.set_page_config(page_title="メール文変換・評価", page_icon="✉")
st.markdown("""
    <style>
    .stButton>button { 
        background-color: #ff6b6b; 
        color: white; 
        border-radius: 10px; 
        transition: transform 0.2s;
        font-weight: 900;
    }
            
    .stButton>button:hover { 
        transform: scale(1.05); 
        color: black;
    }
    </style>
    """, unsafe_allow_html=True)

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

def build_prompt_funny(user_query, inputs="", sep="\n\n### "):
    sys_msg = "以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす冗談交じりの陽気な応答を書きなさい。"
    p = sys_msg
    roles = ["指示", "応答"]
    msgs = [": \n" + user_query, ": \n"]
    if inputs:
        roles.insert(1, "入力")
        msgs.insert(1, ": \n" + inputs)
    for role, msg in zip(roles, msgs):
        p += sep + role + msg
    return p

def readonly_textarea(title, text, min_height="1em"):
    if title != "":
        st.write(f"### {title}")
    st.markdown(f"""
    <div style="
        width: 100%;
        background-color:#f0f0f0;
        padding:15px;
        border-radius:10px;
        font-family: sans-serif;
        font-size:16px;
        color:black;
        white-space: pre-wrap;
        word-wrap: break-word;
        margin-bottom:20px;
        min-height: {min_height};
    ">{text}</div>
    """, unsafe_allow_html=True)


#準備
tokenizer, model, pipe = tokenizer_model()
if torch.cuda.is_available():
    model = model.to("cuda")
client = InferenceClient(api_key="")
labels = {"LABEL_0":3, "LABEL_1":2, "LABEL_2":1, "LABEL_3":0}


selected_mode = option_menu(
    menu_title=None,  
    options=["メール文章変換", "メール評価"],  # 選択肢
    icons=None,  
    menu_icon=None, 
    default_index=0,  # 最初に選択されているインデックス
    orientation="horizontal", 
    styles={
        "nav-item": {
            "margin": "0px 5px 0px 5px", # (上 右 下 左)
            "--hover-color":"#fff"
        }
    }
)


# --- 選択肢に応じた処理 ---
if selected_mode == "メール文章変換":
    selected_method = option_menu(
        menu_title=None,  
        options=["普通モード", "高級モード", "冗談モード"],  # 選択肢
        icons=None,
        menu_icon=None, 
        default_index=0,  # 最初に選択されているインデックス
        orientation="horizontal", 
        styles={
            "nav-item": {
                "margin": "0px 5px 0px 5px", # (上 右 下 左)
                "--hover-color":"#fff"
            }
        }
    )

    status = st_toggle_switch(
        label="件名生成　",
        key="switch_1",
        default_value=False,
        label_after=False, 
        inactive_color="#D3D3D3",  
        active_color= "#fd0000",  
        track_color="#ff6b6b"  
    )

    if selected_method == "普通モード":
        st.success("【入力】  \n・500文字以内で入力してください  \n"\
                    "・メール風に変換したい文章の本文のみを入力してください  \n" \
                    "【注意点】  \n" \
                    "・必ずしも礼儀正しい文章が出力されるとは限りません  \n" \
                    "・変換した文章によってトラブルが発生しても責任を取りません")
    elif selected_method == "高級モード":
        st.success("【入力】  \n・500文字以内で入力してください  \n"\
            "・メール風に変換したい文章の本文のみを入力してください  \n" \
            "【注意点】  \n" \
            "・必ずしも礼儀正しい文章が出力されるとは限りません  \n" \
            "・変換した文章によってトラブルが発生しても責任を取りません  \n" \
            "・変換回数に制限があります  \n" \
            "・APIで外部LLMに入力を転送しています")
    elif selected_method == "冗談モード":
              st.success("【入力】  \n・500文字以内で入力してください  \n"\
            "【注意点】  \n" \
            "・変換した文章によってトラブルが発生しても責任を取りません" \
            )

    st.divider()

    # 入力欄
    text = st.text_area("テキストを入力", "ごめん、明日の会議だけど、資料がまだできてないからリスケでお願い。")

    if st.button("変換する"):
        with st.spinner("変換中..."):
            st.divider()
            if selected_method == "普通モード":
                with torch.no_grad():
                    if status:
                        user_inputs_title = {
                            "user_query": "入力される文章の内容に合う適切な件名を出力してください。",
                            "inputs": text
                        }
                        prompt_title = build_prompt(**user_inputs_title)
                        input_ids_title = tokenizer.encode(
                            prompt_title, 
                            add_special_tokens=True, 
                            return_tensors="pt"
                        )

                        tokens_title = model.generate(
                            input_ids_title.to(device=model.device),
                            max_new_tokens=256,
                            temperature=0.5,
                            top_p=0.95,
                            do_sample=True,
                        )

                        out_title = tokenizer.decode(tokens_title[0][input_ids_title.shape[1]:], skip_special_tokens=True).strip()
                        readonly_textarea("件名", out_title, min_height="3em")

                    user_inputs = {
                        "user_query": "与えられた文章を、ビジネスメール風の丁寧で簡潔な文章に変換してください。",
                        "inputs": text
                    }
                    prompt = build_prompt(**user_inputs)
                    input_ids = tokenizer.encode(
                        prompt, 
                        add_special_tokens=True, 
                        return_tensors="pt"
                    )

                    tokens = model.generate(
                        input_ids.to(device=model.device),
                        max_new_tokens=512,
                        temperature=0.5,
                        top_p=0.95,
                        do_sample=True,
                    )

                    out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                readonly_textarea("変換文", out)
            
            elif selected_method == "高級モード":
                if status:
                    messages_title = [
                        {
                            "role": "user",
                            "content": f"""
                    [SYSTEM]
                    入力された文章に対する適切な件名を出力してください。
                    
                    [USER]
                    {text}
                    """
                        }
                    ]

                    completion_title = client.chat.completions.create(
                        model="openai/gpt-oss-120b", 
                        messages=messages_title, 
                        max_tokens=256
                    )
                    readonly_textarea("変換文", completion_title.choices[0].message.content)

                messages = [
                    {
                        "role": "user",
                        "content": f"""
                [SYSTEM]
                入力された文章を、簡潔なビジネスメール風のに変換してください。件名や署名は不要です。本文のみを出力してください。
                
                [USER]
                {text}
                """
                    }
                ]

                completion = client.chat.completions.create(
                    model="openai/gpt-oss-120b", 
                    messages=messages, 
                    max_tokens=512
                )
                readonly_textarea("変換文", completion.choices[0].message.content)

            elif selected_method == "冗談モード":
                with torch.no_grad():
                    if status:
                        user_inputs_title = {
                            "user_query": "入力される文章の内容をフレンドリーで簡潔な件名に変換してください。",
                            "inputs": text
                        }
                        prompt_title = build_prompt_funny(**user_inputs_title)
                        input_ids_title = tokenizer.encode(
                            prompt_title, 
                            add_special_tokens=True, 
                            return_tensors="pt"
                        )

                        tokens_title = model.generate(
                            input_ids_title.to(device=model.device),
                            max_new_tokens=256,
                            temperature=0.7,
                            top_p=0.95,
                            do_sample=True,
                        )

                        out_title = tokenizer.decode(tokens_title[0][input_ids_title.shape[1]:], skip_special_tokens=True).strip()
                        readonly_textarea("件名", out_title)

                    user_inputs = {
                        "user_query": "入力される文章をフレンドリーで簡潔な文章に変換してください。",
                        "inputs": text
                    }
                    prompt = build_prompt_funny(**user_inputs)
                    input_ids = tokenizer.encode(
                        prompt, 
                        add_special_tokens=True, 
                        return_tensors="pt"
                    )

                    tokens = model.generate(
                        input_ids.to(device=model.device),
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.95,
                        do_sample=True,
                    )

                    out = tokenizer.decode(tokens[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
                    readonly_textarea("変換文", out)

elif selected_mode == "メール評価":
    selected_method = option_menu(
        menu_title=None,  
        options=["丁寧レベル評価", "メール添削評価"],  # 選択肢
        icons=None, 
        menu_icon=None,  
        default_index=0,  # 最初に選択されているインデックス
        orientation="horizontal", 
        styles={
            "nav-item": {
                "margin": "0px 5px 0px 5px", # (上 右 下 左)
                "--hover-color":"#fff"
            }
        }
    )

    st.divider()

    # 入力欄
    text = st.text_area("テキストを入力", "ごめん、明日の会議だけど、資料がまだできてないからリスケでお願い。")
    if st.button("評価する"):
        with st.spinner("評価中..."):
            st.divider()
            if selected_method == "丁寧レベル評価":
                output = pipe(text)
                level = labels[output[0]["label"]]
                score = int(output[0]["score"]*100)
                st.markdown(
                    f"""
                    <div style="
                        background-color: #FFD700;
                        color: black;
                        font-size: 50px;
                        font-weight: bold;
                        text-align: center;
                        padding: 40px;
                        border-radius: 20px;
                        ">
                        丁寧レベル {level} <span style="font-size:30px;">(信頼度 {score}%)</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif selected_method == "メール添削評価":
                messages = [
                    {
                        "role": "user",
                        "content": f"""
                [SYSTEM]
                入力された文章がビジネスメールとして適切か評価してください。ポイントに分けて添削してください。署名は考慮しないでください。
                
                [USER]
                {text}
                """
                    }
                ]

                completion = client.chat.completions.create(
                    model="openai/gpt-oss-120b", 
                    messages=messages, 
                    max_tokens=1024
                )

                st.write(completion.choices[0].message.content)