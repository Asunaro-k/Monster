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
from contents import LLM
device = "cuda:1" if torch.cuda.is_available() else "cpu"

def create_sidebar():
    with st.sidebar:
        st.session_state.levels = st.sidebar.radio(
            "ナビゲーション",
            ["通常モード (usual)", "初心者 (Beginner)", "中級者 (Intermediate)", "上級者 (Advanced)"],
            index=0,
        )
        #st.session_state.levels = "通常モード (usual)"
        with st.sidebar:
            if st.button("新しい問題を生成"):
                # 問題再生成時に状態をリセット
                st.session_state.flag = False
                st.session_state.prompt = None
                st.session_state.reset_audio_input = True
                
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
トピックを1つ考え、何か質問か問いかけを私にしてください。レベルに合わせて日本語訳などを追加してください。

トピック: 

質問: 

答えの例: 
といった形式にしてください

### 例 (ユーザーの英語レベル:Begginerの場合):
トピック: 食べ物

質問: What is your favorite food? (あなたのお気に入りの食べ物は何ですか？)

答えの例: My favorite food is sushi. (私の好きな食べ物は寿司です。)

あなたの番です！あなたのお気に入りの食べ物は何ですか？

"""
llm = LLM.init_ollama()


@st.cache_resource     
def download_and_save_models(save_dir="./models"):
    os.makedirs(save_dir, exist_ok=True)
    
    sttmodel_id = "openai/whisper-large-v3-turbo"
    sttmodel = AutoModelForSpeechSeq2Seq.from_pretrained(
        sttmodel_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    processor = AutoProcessor.from_pretrained(sttmodel_id)

    stt_model = pipeline(
        "automatic-speech-recognition",
        model=sttmodel,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=device,
    )
    stt_model_path = os.path.join(save_dir, "STT_model")
    stt_model.save_pretrained(stt_model_path)



@st.cache_resource(show_spinner="モデルを読み込み中...", max_entries=1)
def load_saved_models(stt_model_path):
    """
    保存したモデルを読み込む関数
    """
    #sttmodel

    stt_model = pipeline(
        "automatic-speech-recognition",
        model=stt_model_path,
        torch_dtype=torch.float16,
        device=device,
    )
    
    # return monsterstdmodel,stdmodel,stt_model
    return stt_model


@st.cache_resource(show_spinner="sttモデルを読み込み中...", max_entries=1)
def load_saved_sttmodels():
    """
    保存したモデルを読み込む関数
    """
    stt_model_path = "./models/STT_model"
    stt_model = pipeline(
        "automatic-speech-recognition",
        model=stt_model_path,
        torch_dtype=torch.float16,
        device=device,
    )
    
    # return monsterstdmodel,stdmodel,stt_model
    return stt_model

@st.cache_resource
def download_and_save_models_if_needed():
    # モデルが保存されるパス
    stt_path = "./models/STT_model"

    # もしモデルが存在しなければダウンロードと保存を実行
    if not os.path.exists(stt_path):
        print("モデルが存在しません。ダウンロードを開始します。")
        download_and_save_models()
    else:
        print("モデルはすでに存在しています。ダウンロードは不要です。")
        print("torch Num GPUs Available: ", torch.cuda.device_count())
           
    
# 初回のみ実行
#monsterStableDef_model_path, StableDef_model_path, stt_model_path = download_and_save_models_if_needed()
download_and_save_models_if_needed()
  
# 保存したモデルの読み込み
stt_model = load_saved_sttmodels()

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
    try:
        with NamedTemporaryFile(delete=True, suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            result = stt_model(
                temp_file.name
            )
            response = result["text"]
            return response
    except Exception as e:
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
    try:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt)
        ]

        # Use the invoke method for chat completions
        response = llm.invoke(messages)
        # Extract and return the generated response from the model
        return response.content
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return None
    

def render():
    st.title("英会話モード / Speaking Talking Mode")
    create_sidebar()
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
    
    
    # レベル別タスク生成
    if st.session_state.levels == "通常モード (usual)":
        user_level = "Usual"
    elif st.session_state.levels == "初心者 (Beginner)":
        user_level = "Beginner"
    elif st.session_state.levels == "中級者 (Intermediate)":
        user_level = "Intermediate"
    elif st.session_state.levels == "上級者 (Advanced)":
        user_level = "Advanced" 
    else:
        user_level = "Usual"
    
    # 難易度変更時の状態リセット
    if user_level != st.session_state.user_level:
        st.session_state.user_level = user_level
        st.session_state.flag = False
        st.session_state.prompt = None
        st.session_state.reset_audio_input = True
    
    #groq_client = ChatGroq(model_name="llama-3.1-70b-versatile")
    ollama_client = llm
    tts_prompt = PromptTemplate(template=TTS_PROMPT, input_variables=["prompt"])
    tts_chain = tts_prompt | ollama_client
    level_prompts = PromptTemplate(template=level_prompt, input_variables=["user_level"])
    level_chain = level_prompts | ollama_client

    # 難易度別タスク生成
    if st.session_state.user_level == "Usual":
        st.write("話しかけてみよう！")
    else:
        if not st.session_state.flag:
            question_prompts = level_chain.invoke(st.session_state.user_level)
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
        pause_threshold=3
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
                tts_result = tts_chain.invoke(generated_text)
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

