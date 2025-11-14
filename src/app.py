# app.py
import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForTokenClassification
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

tokenizer, model, pipe = tokenizer_model()
if torch.cuda.is_available():
    model = model.to("cuda")

# client = InferenceClient(api_key="")

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
        label="件名も作成　",
        key="switch_1",
        default_value=False,
        label_after=False, 
        inactive_color="#D3D3D3",  
        active_color="#11567f",  
        track_color="#29B5E8"  
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
    user_inputs = {
        "user_query": "与えられた文章を、ビジネスメール風の丁寧で簡潔な文章に変換してください。",
        "inputs": text
    }
    prompt = build_prompt(**user_inputs)



    if st.button("変換する"):
        with st.spinner("変換中..."):
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
                pass
                # messages = [
                #     {
                #     "role": "user",
                #     "content": "hello"
                #     }
                # ]

                # completion = client.chat.completions.create(
                #     model="meta-llama/Llama-3.2-3B-Instruct", 
                #     messages=messages, 
                #     max_tokens=500
                # )

                # print(completion.choices[0].message.content)


    

    # client = InferenceClient(api_key="")
    # messages = [
    #     {
    #     "role": "user",
    #     "content": "hello"
    #     }
    # ]

    # completion = client.chat.completions.create(
    #     model="meta-llama/Llama-3.2-3B-Instruct", 
    #     messages=messages, 
    #     max_tokens=500
    # )

    # print(completion.choices[0].message.content)
