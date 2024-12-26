import streamlit as st
from langchain.schema import HumanMessage, SystemMessage, AIMessage, BaseMessage
import mysql.connector
from mysql.connector import Error
import json
import hashlib
import os
import pandas as pd
import plotly.graph_objects as go
from decimal import Decimal, ROUND_HALF_UP
import plotly.graph_objects as go
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, LLMChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
import re
import asyncio
from PIL import Image
import torch
from typing import List
from transformers import pipeline
from duckduckgo_search import DDGS
from requests.exceptions import HTTPError
from langchain_ollama import ChatOllama

st.set_page_config(
    page_title="English Learning App",
    page_icon="📚",
    layout="centered"
)

# グローバル変数としてキャプションモデルを初期化
@st.cache_resource
def load_caption_model():
    # 利用可能なデバイスを動的に判定
    device = 0 if torch.cuda.is_available() else -1
    caption_model = pipeline(
        "image-to-text",
        model="Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device=device,
        max_new_tokens = 100
    )
    return caption_model

# 画像をキャプション化する関数
def generate_image_caption(image_file):
    try:
        # キャプションモデルの取得
        caption_model = st.session_state.image_captioner
        
        # 画像をPILで開く
        image = Image.open(image_file)
        
        # キャプション生成
        captions = caption_model(image)
        
        # キャプションの取得（通常は最初の結果を使用）
        caption = captions[0]['generated_text'] if captions else "画像の説明を生成できませんでした。"
        
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

# URLを検出する関数
def extract_urls(text):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

# Webページの内容を取得する関数
def get_webpage_content(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        return text[:5000]
    except Exception as e:
        return f"Error fetching webpage: {str(e)}"

# 検索クエリ生成のためのプロンプト
QUERY_PROMPT = """
あなたは与えられた質問に対して、以下の3つの判断を行うアシスタントです：
1. 最新の情報が必要かどうか
2. URLが含まれているかどうか
3. 通常の会話で対応可能かどうか

質問: {question}

以下の形式で応答してください：
NEEDS_SEARCH: [true/false] - 最新の情報が必要な場合はtrue
HAS_URL: [true/false] - URLが含まれている場合はtrue
SEARCH_QUERY: [検索クエリ] - NEEDS_SEARCHがtrueの場合のみ必要な検索クエリを書いてください
"""

QUESTION_PROMPT = """
あなたは与えられた文章に対して、以下の判断を行うアシスタントです：
1. 最後に問いかけている文章はどれか

文章: {questionprompt}

以下の形式で応答してください：
NEEDS_QUESTION: [true/false] -問いかけられている場合はtrue
QUESTION_QUERY: [クエリ] - 最後の英語の問いかけの文章を抜き出して書いてください
"""

def init_session_state():
    """セッション状態の初期化"""
    MAX_MEMORY_LIMIT = 10
    #保存済みのモデルをロード
    if "image_captioner" not in st.session_state:
        st.session_state.image_captioner = load_caption_model()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()
    if 'llm' not in st.session_state:
        st.session_state.llm = init_ollama()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatHistory(
            system_prompt='あなたは知識豊富なアシスタントです。会話を良く理解し、適切な返答を行います。'
        )
    # メッセージ上限のチェックと古いメッセージ削除
    if len(st.session_state.memory.chat_memory.messages) > MAX_MEMORY_LIMIT:
        st.session_state.memory.chat_memory.messages = st.session_state.memory.chat_memory.messages[-MAX_MEMORY_LIMIT:]
        
    

class ChatHistory:
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt
        self.messages: List[BaseMessage] = []
        if system_prompt:
            self.messages.append(SystemMessage(content=system_prompt))
    
    async def add_message(self, content: str, is_bot: bool = False, author_name: str = None):
        if is_bot:
            self.messages.append(AIMessage(content=content))
        else:
            prefix = f"{author_name}: " if author_name else ""
            self.messages.append(HumanMessage(content=f"{prefix}{content}"))
    
    async def get_recent_history(self, limit: int = 10) -> List[BaseMessage]:
        start_idx = max(len(self.messages) - limit, 0)
        return self.messages[start_idx:]
    
    def clear_history(self):
        system_message = None
        if self.messages and isinstance(self.messages[0], SystemMessage):
            system_message = self.messages[0]
        self.messages.clear()
        if system_message:
            self.messages.append(system_message)


async def handle_query(prompt, query_chain,question_chain, search, extract_urls, get_webpage_content, chat_history, image_file=None, imageflag=None):
    try:
        chain = ConversationChain(
                    llm=st.session_state.llm,
                    memory=st.session_state.memory,
                    verbose=True
                )
        # 画像がアップロードされている場合
        if image_file is not None:
            #comment_prompt = f"この画像に関する質問/コメント：{prompt}" if prompt is not None else ""
            # 前回の画像と異なる場合はキャプションを生成して会話
            if imageflag:
                # キャプション生成
                with st.spinner('画像を解析中...'):
                    caption = generate_image_caption(image_file)
                st.info(f"画像の説明: {caption}")
                
                prompt_with_image = f"""キーワードに基づいた簡単な英会話をあなたとしたいです。
                以下に例を張ります。例なので猫などの内容は無視してください。\n
                A: Look at the cat! It's sitting on the floor in front of the kitchen. (見て！猫がキッチンの前の床に座ってるよ。)\n
                B: Yeah, it looks so relaxed! (ああ、まったりしてるね！)\n
                A: I know, right? Maybe it's waiting for food. (そうだよね？もしかしたらご飯が食べたいから待ってるのかな。)\n
                B: Do you think the cat is hungry? (猫はおなかすいてるかな？)\n
                A: Hmm, maybe. Cats love food! (えー、もしかしたら。猫は食べ物が大好きなんだよ！)\n\n
                Question: What do you think the cat would say if it could talk? (猫が話すことができたら何と言うかな？)\n
                
                上記のようにキーワードに基づいて何回か日本語訳をつけた英会話をしたうえで、最後に日本語訳をつけた英語で私に何か質問か問いかけをしてください。キーワード：
                {caption}

                """
                
                #response = await st.session_state.llm.apredict(prompt_with_image)
                reply = await chain.ainvoke(prompt_with_image)
                response = reply['response']
                await chat_history.add_message(prompt_with_image, is_bot=False)
                await chat_history.add_message(response, is_bot=True)
                st.session_state['questionprompt'] = response
                st.session_state['previous_uploaded_file'] = image_file
                analysis = await question_chain.ainvoke(st.session_state.get('questionprompt', ''))
                content = analysis.content if hasattr(analysis, 'content') else str(analysis)
                st.session_state['needs_question'] = "NEEDS_QUESTION: true" in content
                if st.session_state['needs_question']:
                    question_query = re.search(r'QUESTION_QUERY: (.*)', content)
                    st.session_state['question_query'] = question_query.group(1)
                    st.session_state['needs_question'] = False
                
            else:
                st.markdown("Question")
                st.markdown(f"{st.session_state['question_query']}")
                # 同じ画像の場合は英会話の正誤判定サポート
                prompt_with_support = f"""
                
                あなたの目標は、ユーザーが楽しく英会話を練習し、上達できるようにサポートすることです。
                次の文章の英語の正しさを日本語で評価し、あっている場合は褒めてください。
                英語ではない場合や間違っている場合は修正を提案し、修正した英語を話してください。
                またその後も会話を続けます。\n
                以下に例を張ります。例なので岩の崖などの内容は無視してください。\n
                Question: What do you think is the most beautiful rocky cliff with a body of water in the world?\n
                上記のように日本語訳をつけた英語で私に何か質問か問いかけをしてください。
                前回の出力で次のような会話を行いました。：{st.session_state.get('questionprompt', '')}\n
                質問の回答: {prompt}"""
                
                
                
                #response = await st.session_state.llm.apredict(prompt_with_support)
                reply = await chain.ainvoke(prompt_with_support)
                response = reply['response']
                #response = await chain.ainvoke(prompt_with_support)
                st.session_state['questionprompt'] = response
                await chat_history.add_message(prompt_with_support, is_bot=False)
                await chat_history.add_message(response, is_bot=True)
                analysis = await question_chain.ainvoke(response)
                content = analysis.content if hasattr(analysis, 'content') else str(analysis)
                st.session_state['needs_question'] = "NEEDS_QUESTION: true" in content
                if st.session_state['needs_question']:
                    question_query = re.search(r'QUESTION_QUERY: (.*)', content)
                    st.session_state['question_query'] = question_query.group(1)
                    st.session_state['needs_question'] = False
        else:
            # 通常のテキストベースの処理
            #recent_history = await chat_history.get_recent_history()
            analysis = await query_chain.ainvoke(prompt)
            content = analysis.content if hasattr(analysis, 'content') else str(analysis)
            needs_search = "NEEDS_SEARCH: true" in content
            has_url = "HAS_URL: true" in content

            if has_url:
                urls = extract_urls(prompt)
                if urls:
                    webpage_content = get_webpage_content(urls[0])
                    prompt_with_content = f"以下のWebページの内容に基づいて適切な返答を考えてください。広告や関連記事などに気を取られないでください。\n\nWebページ内容: {webpage_content}\n\n質問: {prompt}"
                    #response = await st.session_state.llm.apredict(prompt_with_content)
                    #response = await chain.ainvoke(prompt_with_content)
                    reply = await chain.ainvoke(prompt_with_content)
                    response = reply['response']
            elif needs_search:
                st.markdown("""DuckDuckGoで検索中...""")
                search_query = re.search(r'SEARCH_QUERY: (.*)', content)
                if search_query:
                    search_results = search.run(search_query.group(1))
                    prompt_with_search = f"""以下の検索結果の内容に基づいて適切な返答を考えてください。広告や関連記事などに気を取られないでください。
                    できるだけ最新の情報を含めて回答してください。

                    検索結果: {search_results}

                    質問: {prompt}
                    """
                    
                    #response = await st.session_state.llm.apredict(prompt_with_search)
                    reply = await chain.ainvoke(prompt_with_search)
                    response = reply['response']
                else:
                    response = "申し訳ありません。検索クエリの生成に失敗しました。"
            else:
                #st.markdown(st.session_state.memory)
                reply = await chain.ainvoke(prompt)
                response = reply['response']

        # 応答の表示
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")


def init_ollama():
    llm = ChatOllama(
        base_url="http://host.docker.internal:11434/",
        #llama3.1:8b
        #llama3.2:latest
        #7shi/tanuki-dpo-v1.0:latest
        model="gemma2:9b",
        temperature=0.7
    )
    return llm

# Database configuration
DB_CONFIG = {
    'host': os.environ["DB_HOST"],
    'user': os.environ["DB_USER"],
    'password': os.environ["DB_PASSWORD"],
    'database': os.environ["DB_NAME"]
}

# Database initialization
def init_db():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_progress (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                score INT DEFAULT 0,
                total_correct INT DEFAULT 0,
                total_attempts INT DEFAULT 0,
                last_session TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Create history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                japanese_text TEXT,
                english_text TEXT,
                is_correct BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        conn.commit()
    except Error as e:
        st.error(f"Database error: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()


def get_learning_history_data(user_id):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        # 過去30日間のデータを取得
        cursor.execute("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as total_attempts,
                SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers,
                HOUR(created_at) as hour
            FROM learning_history
            WHERE user_id = %s
                AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at), HOUR(created_at)
            ORDER BY date, hour
        """, (user_id,))
        
        return cursor.fetchall()
    except Error as e:
        st.error(f"Error getting history data: {e}")
        return []
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def create_performance_graphs():
    history_data = get_learning_history_data(st.session_state.user['id'])
    
    if not history_data:
        st.info("学習履歴がまだありません。問題を解いて統計を作成しましょう！")
        return
    
    # データをPandas DataFrameに変換
    df = pd.DataFrame(history_data)
    
    # 日別の正解率の計算
    daily_stats = df.groupby('date').agg({
        'total_attempts': 'sum',
        'correct_answers': 'sum'
    }).reset_index()
    daily_stats['accuracy'] = (daily_stats['correct_answers'] / daily_stats['total_attempts'] * 100).round(1)
    
    # 時間帯別の学習パターン
    hourly_stats = df.groupby('hour').agg({
        'total_attempts': 'sum'
    }).reset_index()
    
    # 日別正解率のグラフ
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=daily_stats['date'],
        y=daily_stats['accuracy'],
        mode='lines+markers',
        name='正解率',
        line=dict(color='#2E86C1'),
        marker=dict(size=8)
    ))
    fig1.update_layout(
        title='日別正解率の推移',
        xaxis_title='日付',
        yaxis_title='正解率 (%)',
        yaxis_range=[0, 100],
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    # 時間帯別学習パターンのグラフ
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=hourly_stats['hour'],
        y=hourly_stats['total_attempts'],
        marker_color='#2E86C1'
    ))
    fig2.update_layout(
        title='時間帯別学習回数',
        xaxis_title='時間',
        yaxis_title='学習回数',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    # 統計サマリー
    total_answers = daily_stats['total_attempts'].sum()
    total_correct = daily_stats['correct_answers'].sum()
    #average_accuracy = (total_correct / total_answers * 100).round(1) if total_answers > 0 else 0
    # 精度計算
    if total_answers > 0:
        total_answers = int(total_answers)  # numpy.int64をintに変換
        total_correct = int(total_correct)  # numpy.int64をintに変換
        
        accuracy = Decimal(total_correct) / Decimal(total_answers) * Decimal(100)
        average_accuracy = accuracy.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)  # 小数第1位に丸める
    else:
        average_accuracy = Decimal(0)
    
    # UI表示
    st.write("## 学習統計")
    
    # サマリー統計
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総回答数", f"{total_answers:,}")
    with col2:
        st.metric("総正解数", f"{total_correct:,}")
    with col3:
        st.metric("平均正解率", f"{average_accuracy}%")
    
    # グラフの表示
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)
    
    # 詳細データの表示（展開可能）
    with st.expander("詳細データを表示"):
        st.dataframe(
            daily_stats.rename(columns={
                'date': '日付',
                'total_attempts': '総回答数',
                'correct_answers': '正解数',
                'accuracy': '正解率(%)'
            }),
            hide_index=True
        )
        
def search_images(query, num_results=5):
    try:
        ddgs = DDGS()
        results = ddgs.images(query)
        return results[:num_results]  # 検索結果を指定された数だけ返す
    except HTTPError as e:
        # RatelimitException もしくは他のHTTPエラーが発生した場合
        if e.response.status_code == 429:  # 429はレートリミットエラー
            st.warning("Too many requests. Please wait before trying again.")
        else:
            st.error(f"Error occurred: {e}")
        return []

def setting_confi():
    tmp = st.session_state.generate_toggle
    toggle = st.toggle("画像生成を有効にする")
    if tmp != toggle:
        st.session_state.generate_toggle = toggle
    st.write(st.session_state.generate_toggle)
    st.rerun()
        
# User authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        password_hash = hash_password(password)
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
            (username, password_hash)
        )
        
        user_id = cursor.lastrowid
        cursor.execute(
            "INSERT INTO user_progress (user_id) VALUES (%s)",
            (user_id,)
        )
        
        conn.commit()
        return True
    except Error as e:
        st.error(f"Error creating user: {e}")
        return False
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def verify_user(username, password):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        password_hash = hash_password(password)
        cursor.execute(
            "SELECT * FROM users WHERE username = %s AND password_hash = %s",
            (username, password_hash)
        )
        
        user = cursor.fetchone()
        return user
    except Error as e:
        st.error(f"Error verifying user: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def update_user_progress(user_id, is_correct):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE user_progress 
            SET total_attempts = total_attempts + 1,
                total_correct = total_correct + %s,
                score = score + %s,
                last_session = CURRENT_TIMESTAMP
            WHERE user_id = %s
        """, (1 if is_correct else 0, 1 if is_correct else -1, user_id))
        
        conn.commit()
    except Error as e:
        st.error(f"Error updating progress: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_user_progress(user_id):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute(
            "SELECT * FROM user_progress WHERE user_id = %s",
            (user_id,)
        )
        
        return cursor.fetchone()
    except Error as e:
        st.error(f"Error getting progress: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def save_learning_history(user_id, japanese_text, english_text, is_correct):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO learning_history 
            (user_id, japanese_text, english_text, is_correct)
            VALUES (%s, %s, %s, %s)
        """, (user_id, japanese_text, english_text, is_correct))
        
        conn.commit()
    except Error as e:
        st.error(f"Error saving history: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Login/Signup UI
def show_auth_ui():
    tab1, tab2 = st.tabs(["Login", "Sign Up"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            
            if submit:
                user = verify_user(username, password)
                if user:
                    st.session_state.user = user
                    st.session_state.authenticated = True
                    st.success("Successfully logged in!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
    
    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit = st.form_submit_button("Sign Up")
            
            if submit:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                elif len(new_password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    if create_user(new_username, new_password):
                        st.success("Account created successfully! Please log in.")
                    else:
                        st.error("Username already exists")

# Main app functions (modified to include user data)
def generate_new_sentence():
    # Ollamaの初期化
    chat = init_ollama()
    # LLMに送信するシステムプロンプト
    # system_prompt = """Decide on one random theme and generate a simple Japanese sentence and its English translation. 
    # Return it as a JSON object with format: 
    # {"japanese": "日本語の文", "english": "English sentence"} 
    # Keep sentences simple and suitable for language learners."""
    system_prompt = """Please decide on a random theme and generate one English sentences and their Japanese translations for phrases commonly used in conversation and other situations.
    Return it as a JSON object in the format: 
    {"english": "English sentence", "japanese": "日本語の文"} 
    Keep the sentences simple and appropriate for language learners."""

    # メッセージの作成
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Generate a new sentence")
    ]

    # 応答の処理
    try:
        # Chatモデルからの応答を取得
        response = chat.invoke(messages)

        raw_content = response.content.strip()

        # ` ```json` ブロックがある場合
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_content, re.DOTALL)

        if match:
            # JSON部分だけを抽出
            raw_content = match.group(1)
        
        # JSONデータをロード
        try:
            data = json.loads(raw_content)
            #st.write("Parsed JSON:", data)
        except json.JSONDecodeError as e:
            st.error(f"JSONデコードエラー: {e}")

        # 必要な操作をクライアント側で実行
        # 末尾の英語の句読点を削除
        english_sentence = re.sub(r"[.!?]+$", "", data["english"])

        # 末尾の日本語の句読点を削除
        japanese_sentence = re.sub(r"[。！？]+$", "", data["japanese"])
        

        # 単語の分割とシャッフル
        import random

        # 単語を小文字にして分割
        available_words = english_sentence.lower().split()

        # シャッフル (元のリストは変更される)
        shuffled_words = available_words[:]  # コピーを作成
        random.shuffle(shuffled_words)

        # Streamlit の状態に設定
        st.session_state.current_sentence = english_sentence
        st.session_state.translation = japanese_sentence
        st.session_state.correct_words = available_words
        st.session_state.available_words = shuffled_words  # シャッフルされた単語
        st.session_state.selected_words = []
        # available_words = english_sentence.lower().split()
        # st.session_state.current_sentence = english_sentence
        # st.session_state.translation = japanese_sentence
        # st.session_state.correct_words = available_words
        # st.session_state.available_words = available_words
        # st.session_state.selected_words = []

        # デバッグ表示
        # st.write("English Sentence:", english_sentence)
        # st.write("Japanese Sentence:", japanese_sentence)
        # st.write("Available Words:", available_words)

    except json.JSONDecodeError as e:
        st.error(f"JSONデコードエラー: {e}")
        st.write("レスポンスがJSON形式ではありません。レスポンス内容を確認してください。")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")

def create_chat_ui():
    st.session_state.image_flag = False
    st.title("英語学習チャット")
    # 質問分析のためのチェーン
    query_prompt = PromptTemplate(template=QUERY_PROMPT, input_variables=["question"])
    query_chain = query_prompt | st.session_state.llm

    question_prompt = PromptTemplate(template=QUESTION_PROMPT, input_variables=["questionprompt"])
    question_chain = question_prompt | st.session_state.llm
    
    # Initialize chat history
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if st.session_state.uploaded_file:
        if 'previous_uploaded_file' not in st.session_state:
            st.session_state['previous_uploaded_file'] = None

        # 新しい画像が前回と異なるかを判定
        if st.session_state['previous_uploaded_file'] is None or st.session_state.uploaded_file != st.session_state['previous_uploaded_file']:
            #st.info("新しい画像がアップロードされました")
            st.session_state.image_flag = True
            with st.chat_message("user"):
                st.image(st.session_state.uploaded_file)
                st.session_state['previous_uploaded_file'] = st.session_state.uploaded_file
            asyncio.run(handle_query(None, query_chain,question_chain, st.session_state.search, extract_urls, get_webpage_content, st.session_state.chat_history,st.session_state.uploaded_file,st.session_state.image_flag))
            #st.info("同じ画像で英会話を行います") 
    
    # Chat input
    if prompt := st.chat_input("話しかけてみよう！左にメニューがあるよ"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            if st.session_state.uploaded_file:
                if 'previous_uploaded_file' not in st.session_state:
                    st.session_state['previous_uploaded_file'] = None

                # 新しい画像が前回と異なるかを判定
                if st.session_state['previous_uploaded_file'] is None or st.session_state.uploaded_file != st.session_state['previous_uploaded_file']:
                    #st.info("新しい画像がアップロードされました")
                    st.session_state.image_flag = True
                    st.image(st.session_state.uploaded_file)
                    st.session_state['previous_uploaded_file'] = st.session_state.uploaded_file
                else:
                    image_flag = None
                    st.image(st.session_state.uploaded_file)
                    st.info("同じ画像がアップロードされています")       
            st.markdown(prompt)
        asyncio.run(handle_query(prompt, query_chain,question_chain, st.session_state.search, extract_urls, get_webpage_content, st.session_state.chat_history,st.session_state.uploaded_file,st.session_state.image_flag))
        st.session_state.image_flag = None
            
def create_word_buttons(cols, available_words, selected_words, tab_key=""):
    # インデックス付きの単語リストを作成
    word_objects = [{"id": idx, "word": word} for idx, word in enumerate(available_words)]
    
    # 各単語オブジェクトに対してボタンを表示
    for word_obj in word_objects:
        col_idx = word_obj["id"] % 4
        with cols[col_idx]:
            # この単語のIDがまだ選択されていないか確認
            if not any(selected["id"] == word_obj["id"] for selected in selected_words):
                if st.button(word_obj["word"], key=f"{tab_key}_word_{word_obj['id']}"):
                    selected_words.append(word_obj)
                    st.rerun()

def create_app_ui():
    st.title("英語学習アプリ")
    # Get user progress
    progress = get_user_progress(st.session_state.user['id'])
    # タブを作成
    tab1, tab2, tab3 = st.tabs(["学習", "統計","画像"])
    with tab1:
        # Display user stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Score", progress['score'])
        with col2:
            st.metric("Correct Answers", progress['total_correct'])
        with col3:
            accuracy = (progress['total_correct'] / progress['total_attempts'] * 100) if progress['total_attempts'] > 0 else 0
            st.metric("Accuracy", f"{accuracy:.1f}%")
            
        # if st.session_state.generate_toggle:
        #     if st.session_state.learnimage is None:
        #     # Display Japanese sentence
        #         prompt= f"{st.session_state.translation}:The image is reminiscent of this scene, illustration style, masterpiece"
        #         negative_prompt=""
        #         image = asyncio.run(generate_monster_image(prompt,negative_prompt))
        #         if image:
        #             st.session_state.learnimage = image
                    
        #     if st.session_state.learnimage:
        #             st.image(st.session_state.learnimage)
                  
        st.write("### " + st.session_state.translation)
        
        # selected_wordsの初期化
        if "selected_words" not in st.session_state:
            st.session_state.selected_words = []
            
        # 選択された単語を表示（単語テキストのみを抽出して結合）
        selected_text = " ".join(word_obj["word"] for word_obj in st.session_state.selected_words)
        st.text_area("Your answer:", selected_text, height=100, key="tab1_answer")
        
        # Available words as buttons
        st.write("### Available words:")
        cols = st.columns(4)
        
        # for idx, word in enumerate(st.session_state.available_words):
        #     col_idx = idx % 4
        #     with cols[col_idx]:
        #         if word not in st.session_state.selected_words:
        #             if st.button(word, key=f"word_{idx}"):
        #                 st.session_state.selected_words.append(word)
        #                 st.rerun()
        # Initialize selected_words as a list of word objects if not already done
        # ボタンの作成
        create_word_buttons(cols, st.session_state.available_words, st.session_state.selected_words, tab_key="tab1")

        # Check answer button
        if st.button("Check Answer", type="primary"):
            selected_word_list = [word_obj["word"] for word_obj in st.session_state.selected_words]
            is_correct = " ".join(selected_word_list).lower() == " ".join(st.session_state.correct_words).lower()
            #is_correct = " ".join(st.session_state.selected_words).lower() == " ".join(st.session_state.correct_words).lower()
            #st.write(st.session_state.selected_words)
            #st.write(st.session_state.correct_words)
            if is_correct:
                st.success("Correct! 🎉")
                st.balloons()
            else:
                st.error("Try again!")
            
            # Update progress and save history
            update_user_progress(st.session_state.user['id'], is_correct)
            save_learning_history(
                st.session_state.user['id'],
                st.session_state.translation,
                st.session_state.current_sentence,
                is_correct
            )
            
            if is_correct:
                generate_new_sentence()
                st.session_state.Flag_serachimag = False
                st.session_state.learnimage = None
                st.rerun()

        # Control buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear"):
                st.session_state.selected_words = []
                st.rerun()
        with col2:
            if st.button("New Sentence"):
                generate_new_sentence()
                st.session_state.learnimage = None
                st.rerun()
        
        # Logout button
        # if st.sidebar.button("Logout"):
        #     st.session_state.clear()
        #     st.rerun()
    
    with tab2:
        # 統計グラフの表示
        create_performance_graphs()
    with tab3:
        setting_confi()

        
def create_sidebar():
    with st.sidebar:
        if st.session_state.get("authenticated"):
            st.title("メニュー / Menu")
            
            # Mode selection
            st.write("### モード選択 / Mode Selection")
            if st.button("学習モード / Study Mode", 
                        type="primary" if st.session_state.get("mode", "study") == "study" else "secondary"):
                st.session_state.mode = "study"
                st.rerun()
            
            if st.button("チャットモード / Chat Mode",
                        type="primary" if st.session_state.get("mode", "study") == "chat" else "secondary"):
                st.session_state.mode = "chat"
                st.rerun()

            if st.button("英会話モード / Speaking Talking Mode",
                        type="primary" if st.session_state.get("mode", "study") == "stt" else "secondary"):
                st.session_state.mode = "stt"
                st.rerun()

            if st.session_state.mode=="stt":
                st.session_state.levels = st.sidebar.radio(
                    "ナビゲーション",
                    ["通常モード (usual)", "初心者 (Beginner)", "中級者 (Intermediate)", "上級者 (Advanced)","育成"],
                    index=0,
                )
                #st.session_state.levels = "通常モード (usual)"
                with st.sidebar:
                    if st.button("新しい問題を生成"):
                        # 問題再生成時に状態をリセット
                        st.session_state.flag = False
                        st.session_state.prompt = None
                        st.session_state.reset_audio_input = True
            
            if st.session_state.mode=="chat":
                st.header("About")
                st.markdown("""
                ・ こんにちは！このチャットボットは、さまざまな機能を備えています。  
                ・ どんなことでも気軽に話しかけてくださいね😊  
                ・ 英会話を練習したい時は、画像を送信してください！📸
                どんな質問でもお気軽にどうぞ！🗣️
                """, unsafe_allow_html=True)
                    
                
                if st.button("チャット履歴をクリア"):
                    st.session_state.chat_history.clear_history()
                    st.session_state.messages = []
                    MAX_MEMORY_LIMIT = 0
                    st.session_state.memory.chat_memory.messages = st.session_state.memory.chat_memory.messages[-MAX_MEMORY_LIMIT:]
                    st.session_state.memory = ConversationBufferMemory()

                # 画像アップロード機能を下に配置
                st.session_state.uploaded_file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
        
        # Show user info if logged in
        if st.session_state.get("authenticated"):
            st.write("---")
            st.write(f"### ユーザー / User")
            st.write(f"🧑‍💻 {st.session_state.user['username']}")
            if st.button("ログアウト / Logout"):
                st.session_state.clear()
                st.rerun()

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
llm = ChatOllama(
    base_url="http://host.docker.internal:11434/",
    #llama3.1:8b
    #llama3.2:latest
    #7shi/tanuki-dpo-v1.0:latest
    model="gemma2:9b"
)

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
    
    # Mstdmodel_id = "lambdalabs/sd-pokemon-diffusers"
    # monsterstdmodel = StableDiffusionPipeline.from_pretrained(
    #     Mstdmodel_id,
    #     torch_dtype=torch.float16
    # )
    # monsterStableDef_model_path = os.path.join(save_dir, "STD_model1")
    # monsterstdmodel.save_pretrained(monsterStableDef_model_path)
    
    
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
    
    
    #return monsterStableDef_model_path, StableDef_model_path, stt_model_path
    return StableDef_model_path, stt_model_path

@st.cache_resource(show_spinner="モデルを読み込み中...", max_entries=1)
def load_saved_models(StableDef_model_path, stt_model_path):
    """
    保存したモデルを読み込む関数
    """
    #stdmodel
    # monsterstdmodel = StableDiffusionPipeline.from_pretrained(
    #     pretrained_model_name_or_path=monsterStableDef_model_path,
    #     torch_dtype=torch.float16
    # ).to(device)
    
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

    #sttmodel

    stt_model = pipeline(
        "automatic-speech-recognition",
        model=stt_model_path,
        torch_dtype=torch.float16,
        device=device,
    )
    
    # return monsterstdmodel,stdmodel,stt_model
    return stdmodel,stt_model

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
    std_path = "./models/anylora_diffusers_model"
    #Mstd_path = "./models/STD_model1"
    stt_path = "./models/STT_model"
    
    # 初期化
    #monsterStableDef_model_path = Mstd_path
    StableDef_model_path = std_path
    stt_model_path = stt_path

    # もしモデルが存在しなければダウンロードと保存を実行
    if not os.path.exists(std_path) or not os.path.exists(stt_path):
        print("モデルが存在しません。ダウンロードを開始します。")
        StableDef_model_path, stt_model_path = download_and_save_models()
    else:
        print("モデルはすでに存在しています。ダウンロードは不要です。")
        print("torch Num GPUs Available: ", torch.cuda.device_count())
        
    #return monsterStableDef_model_path, StableDef_model_path, stt_model_path    
    return StableDef_model_path, stt_model_path 
    
# 初回のみ実行
#monsterStableDef_model_path, StableDef_model_path, stt_model_path = download_and_save_models_if_needed()
#StableDef_model_path, stt_model_path = download_and_save_models_if_needed()
  
# 保存したモデルの読み込み
#monsterstdmodel, stdmodel, stt_model = load_saved_models(monsterStableDef_model_path, StableDef_model_path, stt_model_path)
if 'models' not in st.session_state:
    #stdmodel = load_saved_stdmodels()
    #stt_model = load_saved_sttmodels()
    st.session_state.models = True

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
    
# async def generate_image(prompt,negative_prompt):
#     try:
#         # プロンプトに基づいて画像を生成
#         image = stdmodel(
#             prompt=prompt,  # 正しいキーワード引数
#             negative_prompt=negative_prompt,
#             num_inference_steps=20,
#             guidance_scale=7.5
#         ).images[0]
#         return image
#     except Exception as e:
#         st.error(f"画像生成エラー: {str(e)}")
#         return None

def Monster_page():
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


def STT():
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
    # levels = st.sidebar.radio(
    #     "ナビゲーション",
    #     ["通常モード (usual)", "初心者 (Beginner)", "中級者 (Intermediate)", "上級者 (Advanced)","Monster"],
    #     index=0,
    #     key="difficulty_radio"
    # )
    
    # with st.sidebar:
    #     if st.button("新しい問題を生成"):
    #         # 問題再生成時に状態をリセット
    #         st.session_state.flag = False
    #         st.session_state.prompt = None
    #         st.session_state.reset_audio_input = True
        #st.markdown(st.session_state.levels)
    # レベル別タスク生成
    if st.session_state.levels == "通常モード (usual)":
        user_level = "Usual"
    elif st.session_state.levels == "初心者 (Beginner)":
        user_level = "Beginner"
    elif st.session_state.levels == "中級者 (Intermediate)":
        user_level = "Intermediate"
    elif st.session_state.levels == "上級者 (Advanced)":
        user_level = "Advanced" 
    elif st.session_state.levels == "育成":
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
            pause_threshold=4
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

def reset_mode_specific_state(mode):
    """
    Resets states specific to each mode to prevent UI or data conflicts.
    """
    #if mode == "study":
        # Reset study mode specific states
        # st.session_state.selected_words = []
        # st.session_state.learnimage = None
        # st.session_state.Flag_serachimag = False
        # st.session_state.generate_toggle = False
    #elif mode == "chat":
        # Reset chat mode specific states
        # st.session_state.messages = []
        # st.session_state.chat_history = None
    #elif mode == "stt":
        # Reset STT mode specific states
        # st.session_state.levels = "通常モード (usual)"
        # st.session_state.prompt = None
        # st.session_state.reset_audio_input = True

def main():
    st.session_state.image_flag = None
    # Initialize database
    init_db()
    
    # Set default mode if not set
    if 'mode' not in st.session_state:
        st.session_state.mode = "study"
    if 'Flag_serachimag' not in st.session_state:
        st.session_state.Flag_serachimag = False
    if 'image_results' not in st.session_state:
        st.session_state.image_results = []
    if 'image_results' not in st.session_state:
        st.session_state.image_results = []
    if 'learnimage' not in st.session_state:
            st.session_state.learnimage = None
    if 'generate_toggle' not in st.session_state:
        st.session_state.generate_toggle = False
    # if 'last_mode' not in st.session_state:
    #     st.session_state.last_mode = None

    # Check if mode has changed
    # if st.session_state.mode != st.session_state.last_mode:
    #     reset_mode_specific_state(st.session_state.mode)
    #     st.session_state.last_mode = st.session_state.mode
    
    # Create sidebar
    create_sidebar()
    
    # Check authentication
    if 'authenticated' not in st.session_state or not st.session_state.authenticated:
        show_auth_ui()
    else:
        # Display appropriate UI based on mode
        if st.session_state.mode == "study":
            if 'current_sentence' not in st.session_state:
                generate_new_sentence()
            create_app_ui()
        elif st.session_state.mode == "chat":
            # セッション状態の初期化
            init_session_state()

            # DuckDuckGo検索の初期化
            st.session_state.search = DuckDuckGoSearchAPIWrapper(
                backend="api",
                max_results=5,
                region="jp-jp",
                safesearch="off",
                source="text",
                time="w"
            )
            create_chat_ui()
        elif st.session_state.mode == "stt":
            STT()

if __name__ == "__main__":
    main()