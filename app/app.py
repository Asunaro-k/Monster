# app.py
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import random
import time

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
def load_pipeline():
    # pokemon-blipモデルをロード
    pipeline = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2",
        torch_dtype=torch.float16
    )
    if torch.cuda.is_available():
        pipeline = pipeline.to("cuda")
    return pipeline

def generate_monster_image(pipeline, prompt):
    try:
        # プロンプトに基づいて画像を生成
        image = pipeline(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5
        ).images[0]
        return image
    except Exception as e:
        st.error(f"画像生成エラー: {str(e)}")
        return None

def main():
    st.title("モンスター育成ゲーム")
    # モデルのロード
    pipeline = load_pipeline()
    
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
                    prompt = f"a cute {element} type {monster_type} pokemon, simple design, white background"
                    image = generate_monster_image(pipeline, prompt)
                    if image:
                        st.session_state.monster.image = image
    
    # モンスター情報の表示と操作
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
                        prompt = f"a powerful evolved {element} type {monster_type} pokemon, detailed features, white background"
                        new_image = generate_monster_image(pipeline, prompt)
                        if new_image:
                            st.session_state.monster.image = new_image
                st.rerun()
        
        with col4:
            if st.button("リセット"):
                st.session_state.monster = None
                st.session_state.element = None
                st.session_state.monster_type = None
                st.rerun()

if __name__ == "__main__":
    main()
