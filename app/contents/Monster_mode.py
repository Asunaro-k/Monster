import streamlit as st
from audio_recorder_streamlit import audio_recorder
from tempfile import NamedTemporaryFile
import ffmpeg
import io
import edge_tts
from edge_tts import VoicesManager
import random
import asyncio
from langchain.prompts import PromptTemplate
import re
import os
from diffusers import StableDiffusionPipeline, PNDMScheduler
from safetensors.torch import load_file
from PIL import Image
import time
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:1" if torch.cuda.is_available() else "cpu"
class Monster:
    def __init__(self, name):
        self.name = name
        self.level = 1
        self.exp = 0
        self.hp = 100
        self.strength = 10
        self.image = None

    def feed(self):
        self.exp += 20
        if self.exp >= 100:
            self.level_up()
    
    def level_up(self):
        self.level += 1
        self.exp = 0
        self.hp += 20
        self.strength += 5
        return True
    
    def to_dict(self):
        return {
            "name": self.name,
            "level": self.level,
            "exp": self.exp,
            "hp": self.hp,
            "strength": self.strength
        }

@st.cache_resource     
def download_and_save_models(save_dir="./models"):
    os.makedirs(save_dir, exist_ok=True)
    
    stdmodel_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    stdmodel = StableDiffusionPipeline.from_pretrained(
        stdmodel_id,
        torch_dtype=torch.float16
    )
    StableDef_model_path = os.path.join(save_dir, "STD_model")
    stdmodel.save_pretrained(StableDef_model_path)


@st.cache_resource(show_spinner="stdモデルを読み込み中...", max_entries=1)
def load_saved_stdmodels():
    """
    保存したモデルを読み込む関数
    """
    StableDef_model_path = "./models/anylora_diffusers_model"
    
    stdmodel = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=StableDef_model_path,
        torch_dtype=torch.float16
    ).to(device)
    
    # LoRAのロード
    lora_path = "./models/pokemon_v3_offset.safetensors"
    lora_weights = load_file(lora_path)

    # LoRAをモデルに適用
    def apply_lora(pipe, lora_weights, alpha=1.0):
        for name, param in pipe.unet.named_parameters():
            if name in lora_weights:
                param.data += alpha * lora_weights[name].data

    apply_lora(stdmodel, lora_weights)

    # スケジューラの設定
    stdmodel.scheduler = PNDMScheduler.from_config(stdmodel.scheduler.config)
    
    # return monsterstdmodel,stdmodel,stt_model
    return stdmodel

@st.cache_resource
def download_and_save_models_if_needed():
    # モデルが保存されるパス
    std_path = "./models/anylora_diffusers_model"

    # もしモデルが存在しなければダウンロードと保存を実行
    if not os.path.exists(std_path):
        print("モデルが存在しません。ダウンロードを開始します。")
        download_and_save_models()
    else:
        print("モデルはすでに存在しています。ダウンロードは不要です。")
        print("torch Num GPUs Available: ", torch.cuda.device_count())
           
    
# 初回のみ実行
#monsterStableDef_model_path, StableDef_model_path, stt_model_path = download_and_save_models_if_needed()
download_and_save_models_if_needed()
  
# 保存したモデルの読み込み
stdmodel = load_saved_stdmodels()


async def generate_monster_image(prompt,negative_prompt):
    try:
        # プロンプトに基づいて画像を生成
        image = stdmodel(
            prompt=prompt,  # 正しいキーワード引数
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=7.5
        ).images[0]
        return image
    except Exception as e:
        st.error(f"画像生成エラー: {str(e)}")
        return None
    
def render():
    st.title("育成モード / Nurturing Mode")
    # セッション状態の初期化
    if 'monster' not in st.session_state:
        st.session_state.monster = None
        st.session_state.element = None
        st.session_state.monster_type = None
    
    # モンスター作成フォーム
    if st.session_state.monster is None:
        with st.form("create_monster"):
            monster_name = st.text_input("モンスターの名前を入力してください")
            monster_type = st.selectbox(
                "モンスターのタイプを選択",
                ["dragon", "slime", "fairy", "rock monster"]
            )
            
            # 属性の追加
            element = st.selectbox(
                "モンスターの属性を選択",
                ["fire", "water", "grass", "electric"]
            )
            
            submit = st.form_submit_button("モンスターを作成")
            
            if submit and monster_name:
                st.session_state.monster = Monster(monster_name)
                st.session_state.element = element
                st.session_state.monster_type = monster_type
                with st.spinner("モンスターを生成中..."):
                    # プロンプトの作成
                    #prompt = f"a cute {element} type {monster_type} pokemon, simple design, white background, masterpiece"
                    prompt= f"a cute {element} type {monster_type} fantasy one monster, resembling a Pokémon, small and chubby, with bright and colorful fur, large sparkling eyes, a happy expression, highly detailed, anime-inspired style, soft shading, vibrant colors"
                    negative_prompt="realistic, horror, scary, creepy, low quality, blurry, dull colors"
                    image = asyncio.run(generate_monster_image(prompt,negative_prompt))
                    if image:
                        st.session_state.monster.image = image
    
    # モンスター情報の表示と操作
    #with st.sidebar:
    if st.session_state.monster:
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.monster.image:
                st.image(st.session_state.monster.image, caption=st.session_state.monster.name)
        
        with col2:
            st.write(f"名前: {st.session_state.monster.name}")
            st.progress(st.session_state.monster.exp / 100)
            st.write(f"レベル: {st.session_state.monster.level}")
            st.write(f"経験値: {st.session_state.monster.exp}/100")
            st.write(f"HP: {st.session_state.monster.hp}")
            st.write(f"力: {st.session_state.monster.strength}")
        
        # アクションボタン
        col3, col4 = st.columns(2)
        with col3:
            if st.button("餌をあげる"):
                st.session_state.monster.feed()
                if st.session_state.monster.exp == 0:  # レベルアップした場合
                    st.balloons()
                    with st.spinner("モンスターが進化中..."):
                        # セッション状態から属性とタイプを取得
                        element = st.session_state.element
                        monster_type = st.session_state.monster_type
                        # 進化時はより強そうなプロンプトを使用
                        prompt = f"a powerful evolved {element} type {monster_type} fantasy one monster, resembling a Pokémon, small and chubby, with bright and colorful fur, large sparkling eyes, a happy expression, highly detailed, anime-inspired style, soft shading, vibrant colors, detailed features"
                        negative_prompt="realistic, creepy, low quality, blurry"
                        new_image = asyncio.run(generate_monster_image(prompt,negative_prompt))
                        if new_image:
                            st.session_state.monster.image = new_image
                st.rerun()
        
        with col4:
            if st.button("リセット"):
                st.session_state.monster = None
                st.session_state.element = None
                st.session_state.monster_type = None
                st.rerun()    

