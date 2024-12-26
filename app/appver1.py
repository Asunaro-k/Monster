import groq
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
from langchain_groq import ChatGroq
import re
import os
from diffusers import StableDiffusionPipeline
from PIL import Image
import time


import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:1" if torch.cuda.is_available() else "cpu"

# 音声生成のためのプロンプト
TTS_PROMPT = """
あなたは与えられた文章に対して、以下の判断を行うアシスタントです：
1. この言語が日本語なのか英語なのか

文章: {prompt}

以下の形式で応答してください：
language_jp: [true/false] - 英語が含まれている場合はfalse
"""

level_prompt = """
ユーザーの英語レベル: {user_level}
このレベルに適した会話練習問題を行いたいです。
トピックを1つ考え、何か問いかけを私にしてください。レベルに合わせて日本語訳などを追加してください。
例：トピック: "環境保護と経済成長のバランス"
質問: "環境保護と経済成長は相反する概念として見られることが多いですが、実際にはどのように両立できるでしょうか?" (How can environmental protection and economic growth be reconciled, given that they are often seen as conflicting concepts?)
この質問は、環境保護と経済成長の関係についての考えを深める機会となり、ユーザーが批判的思考力と表現力を試すことができます。
英語の答えを考えてみてください。
"""

def convert_audio_format(audio_bytes, target_format="mp3"):
    import ffmpeg
    from tempfile import NamedTemporaryFile
    
    with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
        temp_file.write(audio_bytes)
        temp_file.flush()

        with NamedTemporaryFile(delete=True, suffix=f".{target_format}") as converted_file:
            try:
                # overwrite_output() を追加
                ffmpeg.input(temp_file.name).output(converted_file.name).overwrite_output().run()
                # 変換されたファイルの内容を返す
                return converted_file.read()
            except ffmpeg._run.Error as e:
                #print("FFmpeg error:", e.stderr)  # 標準エラー出力を表示
                raise


        
def transcribe_audio_to_text(audio_bytes):
    groq_client = groq.Groq()
    try:
        with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            with open(temp_file.name, "rb") as audio_file:
                response = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo", file=audio_file, response_format="text"
                )
                return response
    except groq.BadRequestError as e:
        st.error(f"Error transcribing audio: {e}")
        return None
    

# Text-to-Speech function
async def text_to_speech(text,language="ja"):
    try:
        voices = await VoicesManager.create()  # 非同期関数として呼び出す
        voice = voices.find(Gender="Female", Language=language)
        
        # ユニークな一時ファイルをセッションごとに作成
        if 'tts_audio_path' not in st.session_state:
            temp_file = NamedTemporaryFile(delete=False, suffix=".mp3")
            st.session_state.tts_audio_path = temp_file.name

        # ファイルに音声を保存
        communicate = edge_tts.Communicate(text, random.choice(voice)["Name"])
        await communicate.save(st.session_state.tts_audio_path)  # 非同期で保存

        return st.session_state.tts_audio_path  # ユニークな一時ファイルパスを返す
    except Exception as e:
        st.error(f"Error converting text to speech: {e}")
        return None
    
# Generate text using Groq
def generate_text(prompt):
    groq_client = groq.Groq()
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-70b-versatile"
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return None
    
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
        pipeline = pipeline.to(device)
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

def Monster_page():
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
    with st.sidebar:
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


def main():
    st.title("Voice to Text Transcription")

    # 初期化
    if 'user_level' not in st.session_state:
        st.session_state.user_level = "Usual"
    if 'skip_first_attempt' not in st.session_state:
        st.session_state.skip_first_attempt = True
    if 'flag' not in st.session_state:
        st.session_state.flag = False
    if 'content' not in st.session_state:
        st.session_state.content = ""
    if 'audioflag' not in st.session_state:
        st.session_state.audioflag = False
    if 'prompt' not in st.session_state:
        st.session_state.prompt = None
    if 'reset_audio_input' not in st.session_state:
        st.session_state.reset_audio_input = False

    # サイドバー: 難易度選択 & 新しい問題生成ボタン
    levels = st.sidebar.radio(
        "ナビゲーション",
        ["通常モード (usual)", "初心者 (Beginner)", "中級者 (Intermediate)", "上級者 (Advanced)","Monster"],
        index=0
    )
    
    with st.sidebar:
        if st.button("新しい問題を生成"):
            # 問題再生成時に状態をリセット
            st.session_state.flag = False
            st.session_state.prompt = None
            st.session_state.reset_audio_input = True
            
    # レベル別タスク生成
        if levels == "通常モード (usual)":
            user_level = "Usual"
        elif levels == "初心者 (Beginner)":
            user_level = "Beginner"
        elif levels == "中級者 (Intermediate)":
            user_level = "Intermediate"
        elif levels == "上級者 (Advanced)":
            user_level = "Advanced" 
        elif levels == "Monster":
            user_level = "Monster" 
        else:
            user_level = "Usual"
    
    if user_level=="Monster":
        Monster_page()
    else:
        # 難易度変更時の状態リセット
        if user_level != st.session_state.user_level:
            st.session_state.user_level = user_level
            st.session_state.flag = False
            st.session_state.prompt = None
            st.session_state.reset_audio_input = True
        
        groq_client = ChatGroq(model_name="llama-3.1-70b-versatile")
        tts_prompt = PromptTemplate(template=TTS_PROMPT, input_variables=["prompt"])
        tts_chain = tts_prompt | groq_client
        level_prompts = PromptTemplate(template=level_prompt, input_variables=["user_level"])
        level_chain = level_prompts | groq_client

        # 難易度別タスク生成
        if st.session_state.user_level == "Usual":
            st.write("話しかけてみよう！")
        else:
            if not st.session_state.flag:
                question_prompts = asyncio.run(level_chain.ainvoke(st.session_state.user_level))
                st.session_state.content = question_prompts.content if hasattr(question_prompts, 'content') else str(question_prompts)
            if st.session_state.content:
                st.write(st.session_state.content)
            st.session_state.flag = True

        # 音声録音ウィジェット
        audio_bytes = audio_recorder(
            text="",
            recording_color="#e8b62c",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=5
        )
        # st.markdown(st.session_state.reset_audio_input)

        # マイク権限のため初回をスキップ
        if st.session_state.skip_first_attempt:
            if audio_bytes:
                st.warning("Skipping first attempt to allow microphone permissions.")
                st.session_state.skip_first_attempt = False
            return  # 初回は何もしない

        # 音声入力のリセットフラグがTrueの場合、音声入力をクリア
        if st.session_state.reset_audio_input:
            audio_bytes = None
            st.session_state.reset_audio_input = False


        # 音声データ処理
        if audio_bytes:
            audio_bytes = convert_audio_format(audio_bytes, target_format="mp3")
            transcript = transcribe_audio_to_text(audio_bytes)
            st.session_state.prompt = transcript
            if st.session_state.prompt:
                st.write("Transcribed Text:", st.session_state.prompt)
                # テキスト生成
                if st.session_state.user_level == "Usual":
                    generated_text = generate_text(st.session_state.prompt)
                else:
                    generated_text = generate_text(f"{st.session_state.content}の質問をもとに返答しています。以下の発話の英語としての正確性を評価し、改善点を提示してください。:\n{st.session_state.prompt}")
                
                if generated_text:
                    st.write("Generated Text:")
                    st.write(generated_text)

                    # 言語判定と音声生成
                    tts_result = asyncio.run(tts_chain.ainvoke(generated_text))
                    tts_content = tts_result.content if hasattr(tts_result, 'content') else str(tts_result)
                    language_flag = "language_jp: true" in tts_content
                    language = "ja" if language_flag else "en"

                    # 音声生成
                    audio_path = asyncio.run(text_to_speech(generated_text, language))
                    if audio_path:
                        st.audio(audio_path, format='audio/mpeg')
                        try:
                            os.remove(audio_path)
                            print("音声ファイルが削除されました。")
                        except FileNotFoundError:
                            print("削除しようとした音声ファイルが見つかりませんでした。")
                        except Exception as e:
                            print(f"音声ファイルの削除中にエラーが発生しました: {e}")
    

if __name__ == "__main__":
    main()
