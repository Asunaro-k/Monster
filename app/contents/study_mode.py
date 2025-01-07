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
from contents import LLM
import random

DB_CONFIG = {
    'host': os.environ["DB_HOST"],
    'user': os.environ["DB_USER"],
    'password': os.environ["DB_PASSWORD"],
    'database': os.environ["DB_NAME"]
}

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

def setting_confi():
    st.write("## 学習設定 / Learning Settings")
    
    # 現在の設定を取得
    current_settings = get_user_settings(st.session_state.user['id'])
    if current_settings is None:
        # デフォルト設定を作成
        create_default_settings(st.session_state.user['id'])
        current_settings = get_user_settings(st.session_state.user['id'])

    with st.form("settings_form"):
        # 難易度設定
        st.write("### 難易度設定 / Difficulty Settings")
        difficulty = st.select_slider(
            "難易度を選択してください / Select difficulty level",
            options=['easy', 'normal', 'hard'],
            value=current_settings['difficulty']
        )

        # 難易度の説明
        st.write("#### 各難易度の特徴 / Difficulty Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Easy**
            - 短い文章 (3-6語)
            - ダミー単語なし
            - 基本的な語彙
            """)
        
        with col2:
            st.markdown("""
            **Normal**
            - 中程度の文章 (6-9語)
            - 2個のダミー単語
            - 一般的な語彙
            """)
        
        with col3:
            st.markdown("""
            **Hard**
            - 長い文章 (9-12語)
            - 4個のダミー単語
            - 高度な語彙
            """)

        # テーマ設定
        st.write("### テーマ設定 / Theme Settings")
        preset_themes = [
            'ランダム / Random',
            '日常会話 / Daily Conversation',
            'ビジネス / Business',
            '旅行 / Travel',
            '趣味 / Hobbies',
            '文化 / Culture'
        ]
        
        # ユーザーが選択したテーマを取得
        selected_themes = current_settings.get('themes', [])

        # カスタムテーマ
        custom_theme = st.text_input(
            "カスタムテーマを追加 / Add custom theme",
            value=current_settings.get('custom_theme', ''),
            placeholder="Enter your custom theme here"
        )

        # カスタムテーマが入力されていれば、それを選択肢に追加
        if custom_theme and custom_theme not in preset_themes:
            preset_themes.append(custom_theme)

        # 新しいテーマを選択肢に含める
        selected_themes = st.multiselect(
            "学習したいテーマを選択してください / Select themes you want to learn",
            preset_themes,
            default=[theme for theme in selected_themes if theme in preset_themes] + ([custom_theme] if custom_theme else [])
        )

        # ダミー単語設定
        st.write("### ダミー単語設定 / Dummy Words Settings")
        use_dummy = st.toggle(
            "ダミー単語を追加する / Add dummy words",
            value=current_settings.get('use_dummy_words', False)
        )

        # 設定の保存
        submit = st.form_submit_button("設定を保存 / Save Settings", type="primary")
        
        if submit:
            # 設定の保存
            save_user_settings(
                st.session_state.user['id'],
                difficulty,
                selected_themes,
                use_dummy
            )
            st.success("設定が保存されました！ / Settings saved successfully!")
            st.session_state.surrenderFlag = False
            generate_new_sentence()
            st.session_state.learnimage = None
            st.session_state[f"tab1_selected_word_ids"] = []
            st.rerun()

def create_default_settings(user_id):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        default_themes = json.dumps(['ランダム / Random'])
        cursor.execute("""
            INSERT INTO user_settings (user_id, difficulty, themes, use_dummy_words)
            VALUES (%s, 'normal', %s, TRUE)
        """, (user_id, default_themes))
        
        conn.commit()
    except Error as e:
        st.error(f"Error creating default settings: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def get_user_settings(user_id):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True, buffered=True)  # buffered=True を追加
        
        # デバッグ出力を追加
        #print(f"Fetching settings for user_id: {user_id}")
        
        cursor.execute("""
            SELECT difficulty, themes, use_dummy_words 
            FROM user_settings 
            WHERE user_id = %s
        """, (user_id,))
        
        result = cursor.fetchone()  # 結果を取得
        
        # デバッグ出力を追加
        #print("Retrieved settings:", result)
        
        if result:
            # JSON文字列を辞書型に変換
            result['themes'] = json.loads(result['themes']) if result['themes'] else []
            return result
        return None
    except Error as e:
        st.error(f"Error getting settings: {e}")
        return None
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def save_user_settings(user_id, difficulty, themes, use_dummy_words):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        # JSON文字列に変換
        themes_json = json.dumps(themes)
        
        cursor.execute("""
            UPDATE user_settings
            SET difficulty = %s,
                themes = %s,
                use_dummy_words = %s
            WHERE user_id = %s
        """, (difficulty, themes_json, use_dummy_words, user_id))
        
        conn.commit()
        # print(f"User settings updated for user_id: {user_id}")
    except Error as e:
        st.error(f"Error saving user settings: {e}")
        print(f"Error: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
    
# def generate_new_sentence():
#     # Ollamaの初期化
#     chat = LLM.init_ollama()
#     # LLMに送信するシステムプロンプト
#     system_prompt = """Please decide on a random theme and generate one English sentences and their Japanese translations for phrases commonly used in conversation and other situations.
#     Return it as a JSON object in the format: 
#     {"english": "English sentence", "japanese": "日本語の文"} 
#     Keep the sentences simple and appropriate for language learners."""

#     # メッセージの作成
#     messages = [
#         SystemMessage(content=system_prompt),
#         HumanMessage(content="Generate a new sentence")
#     ]

#     # 応答の処理
#     try:
#         # Chatモデルからの応答を取得
#         response = chat.invoke(messages)

#         raw_content = response.content.strip()

#         # ` ```json` ブロックがある場合
#         match = re.search(r"```json\s*(\{.*?\})\s*```", raw_content, re.DOTALL)

#         if match:
#             # JSON部分だけを抽出
#             raw_content = match.group(1)
        
#         # JSONデータをロード
#         try:
#             data = json.loads(raw_content)
#             #st.write("Parsed JSON:", data)
#         except json.JSONDecodeError as e:
#             st.error(f"JSONデコードエラー: {e}")

#         # 必要な操作をクライアント側で実行
#         # 末尾の英語の句読点を削除
#         english_sentence = re.sub(r"[.!?]+$", "", data["english"])

#         # 末尾の日本語の句読点を削除
#         japanese_sentence = re.sub(r"[。！？]+$", "", data["japanese"])
        

#         # 単語の分割とシャッフル
#         import random

#         # 単語を小文字にして分割
#         available_words = english_sentence.lower().split()

#         # シャッフル (元のリストは変更される)
#         shuffled_words = available_words[:]  # コピーを作成
#         random.shuffle(shuffled_words)

#         # Streamlit の状態に設定
#         st.session_state.current_sentence = english_sentence
#         st.session_state.translation = japanese_sentence
#         st.session_state.correct_words = available_words
#         st.session_state.available_words = shuffled_words  # シャッフルされた単語
#         st.session_state.selected_words = []

#     except json.JSONDecodeError as e:
#         st.error(f"JSONデコードエラー: {e}")
#         st.write("レスポンスがJSON形式ではありません。レスポンス内容を確認してください。")
#     except Exception as e:
#         st.error(f"エラーが発生しました: {e}")
  
def generate_new_sentence():
    # ユーザー設定の取得と検証
    user_settings = get_user_settings(st.session_state.user['id'])
    # デバッグ出力を追加
    #print("Retrieved user settings:", user_settings)
    if not user_settings:
        create_default_settings(st.session_state.user['id'])
        user_settings = get_user_settings(st.session_state.user['id'])
        if not user_settings:  # 設定の取得に失敗した場合
            st.error("設定の取得に失敗しました。")
            return

    # セッション状態の初期化
    if not all(key in st.session_state for key in ['current_sentence', 'translation', 'correct_words', 'available_words', 'selected_words']):
        st.session_state.current_sentence = ''
        st.session_state.translation = ''
        st.session_state.correct_words = []
        st.session_state.available_words = []
        st.session_state.selected_words = []

    # 難易度設定の取得と検証
    difficulty = user_settings.get('difficulty', 'normal')
    if difficulty not in ['easy', 'normal', 'hard']:
        st.error("無効な難易度設定です。")
        return

    # 難易度に基づく設定
    difficulty_configs = {
        'easy': {
            'length': '3-6 words',
            'vocab': 'Beginner',
            'complexity': 'simple present tense only',
            'dummy_count': 0
        },
        'normal': {
            'length': '6-9 words',
            'vocab': 'Intermediate',
            'complexity': 'present and past tense',
            'dummy_count': 2
        },
        'hard': {
            'length': '9-12 words',
            'vocab': 'Advanced',
            'complexity': 'various tenses and complex structures',
            'dummy_count': 4
        }
    }

    # テーマの設定と検証
    selected_themes = user_settings.get('themes', ['ランダム / Random'])
    if not isinstance(selected_themes, list):
        st.error("テーマ設定が無効です。")
        return

    # テーマプロンプトの生成
    theme_prompt = ""
    if 'ランダム / Random' not in selected_themes:
        theme_list = [theme.split(' / ')[0] for theme in selected_themes]
        theme_prompt = f"The sentence should be related to one of these themes: {', '.join(theme_list)}. "

    # システムプロンプトの構築
    config = difficulty_configs[difficulty]
    system_prompt = f"""Generate one English sentence and its Japanese translation based on a given theme. 
    The sentence should reflect a common, natural situation or conversation related to the theme. For example, for the 'Travel' theme,
    it could include phrases like asking for directions, booking tickets, or interacting with locals, and hotel conversation, ripped jeans, etc. For 'Business,' it might include meetings, negotiations, or emails. 
    Ensure the sentence fits a realistic context.The sentence should not be limited to generic descriptions but should include common and useful phrases in the specific context. 
    Aim for a natural, conversational tone.
    Interject a little humor in between.:
    
    - Length: {config['length']}
    - Vocabulary level: {config['vocab']}
    - Grammatical complexity: {config['complexity']}
    {theme_prompt}
    The sentence should not be limited to a single context but should reflect a variety of common phrases related to the theme.
    
    Return as JSON: {{"english": "English sentence", "japanese": "日本語の文"}}
    Do not include anything else. The response should only contain the JSON object with the fields "english" and "japanese"."""
    #print(f"systemprompt:{system_prompt}")
    try:
        
        # LLMの初期化と実行
        chat = LLM.init_ollama()
        response = chat.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Generate a new sentence")
        ])

        # レスポンスの処理
        raw_content = response.content.strip()
        # print(raw_content)
        
        match = re.search(r"```json\s*(\{.*?\})\s*```", raw_content, re.DOTALL)
        
        if match:
            raw_content = match.group(1)
            
        # 不要な閉じカギカッコを取り除く
        raw_content = re.sub(r"」.*$", '"}', raw_content)

        data = json.loads(raw_content)
        # 文章の整形
        english_sentence = re.sub(r"[.!?]+$", "", data["english"])
        japanese_sentence = re.sub(r"[。！？」]+$", "", data["japanese"])
        
        # 単語リストの作成
        available_words = english_sentence.lower().split()
        
        # ダミー単語の追加
        if user_settings.get('use_dummy_words', True):
            dummy_count = config['dummy_count']
            if dummy_count > 0:
                dummy_pools = {
                    'easy': [
                        'cat', 'dog', 'book', 'pen', 'apple',
                        'tree', 'house', 'car', 'table', 'banana',
                        'bird', 'fish', 'computer', 'phone', 'water',
                        'rain', 'sun', 'sky', 'food', 'shoe'
                    ],
                    'normal': [
                        'computer', 'beautiful', 'quickly', 'happy', 'important',
                        'software', 'development', 'camera', 'education', 'family',
                        'scientific', 'discussion', 'economy', 'analysis', 'technology',
                        'research', 'design', 'project', 'problem', 'solution'
                    ],
                    'hard': [
                        'nevertheless', 'particularly', 'development', 'essential', 'consideration',
                        'philosophy', 'unprecedented', 'improvisation', 'counterproductive', 'paradoxical',
                        'metaphor', 'infrastructure', 'interdisciplinary', 'prognostication', 'revolutionary',
                        'individuality', 'artificial', 'optimization', 'complexity', 'transcendental'
                    ]
                }
                
                dummy_pool = dummy_pools[difficulty]
                available_dummy_words = [word for word in dummy_pool if word not in available_words]
                if available_dummy_words:
                    dummy_words = random.sample(available_dummy_words, min(dummy_count, len(available_dummy_words)))
                    available_words.extend(dummy_words)

        # 単語のシャッフルと状態の更新
        shuffled_words = available_words[:]
        random.shuffle(shuffled_words)
        
        st.session_state.current_sentence = english_sentence
        st.session_state.translation = japanese_sentence
        st.session_state.correct_words = english_sentence.lower().split()
        st.session_state.available_words = shuffled_words
        st.session_state.selected_words = []

    except Exception as e:
        st.error(f"文章生成中にエラーが発生しました: {e}")
        return     

# def create_word_buttons(cols, available_words, selected_words, tab_key=""):
#     # インデックス付きの単語リストを作成 
#     word_objects = [{"id": idx, "word": word} for idx, word in enumerate(available_words)]
    
#     # 初期化: ボタン押下状態をセッションステートで管理
#     if f"{tab_key}_selected_word_ids" not in st.session_state:
#         st.session_state[f"{tab_key}_selected_word_ids"] = []

#     # 各単語オブジェクトに対してボタンを表示
#     for word_obj in word_objects:
#         col_idx = word_obj["id"] % 4
#         with cols[col_idx]:
#             # ボタンが押されたかどうかを確認
#             is_selected = word_obj["id"] in st.session_state[f"{tab_key}_selected_word_ids"]
            
#             # ボタンを無効化（灰色化）するためのオプションを指定
#             if st.button(
#                 word_obj["word"], 
#                 key=f"{tab_key}_word_{word_obj['id']}", 
#                 disabled=is_selected  # 押された場合はボタンを無効化
#             ):
#                 # ボタンが押されたら選択リストに追加
#                 if not is_selected:
#                     st.session_state[f"{tab_key}_selected_word_ids"].append(word_obj["id"])
#                     selected_words.append(word_obj)
#                     st.rerun()

# 
# def create_word_buttons(available_words, selected_words, tab_key=""):
#     # スタイルを適用するためのCSSを追加
#     st.markdown("""
#         <style>
#             .word-button-container {
#                 display: flex;
#                 flex-wrap: wrap;
#                 gap: 8px;
#                 justify-content: flex-start;
#             }
            
#             .word-button {
#                 flex: 0 1 calc(33.333% - 8px);
#                 min-width: 80px;
#             }
            
#             @media (max-width: 768px) {
#                 .word-button {
#                     flex: 0 1 calc(50% - 8px);
#                 }
#             }
#         </style>
#     """, unsafe_allow_html=True)
    
#     # ボタンコンテナを作成
#     st.markdown('<div class="word-button-container">', unsafe_allow_html=True)
    
#     # インデックス付きの単語リストを作成 
#     word_objects = [{"id": idx, "word": word} for idx, word in enumerate(available_words)]
    
#     # 初期化: ボタン押下状態をセッションステートで管理
#     if f"{tab_key}_selected_word_ids" not in st.session_state:
#         st.session_state[f"{tab_key}_selected_word_ids"] = []

#     # Create columns for buttons
#     col1, col2, col3 = st.columns([1, 1, 1])
    
#     # Split words into groups of 3
#     for i in range(0, len(word_objects), 3):
#         group = word_objects[i:i+3]
        
#         # Display each word in the group in its respective column
#         for j, word_obj in enumerate(group):
#             with [col1, col2, col3][j]:
#                 is_selected = word_obj["id"] in st.session_state[f"{tab_key}_selected_word_ids"]
                
#                 if st.button(
#                     word_obj["word"],
#                     key=f"{tab_key}_word_{word_obj['id']}",
#                     disabled=is_selected,
#                     use_container_width=True
#                 ):
#                     if not is_selected:
#                         st.session_state[f"{tab_key}_selected_word_ids"].append(word_obj["id"])
#                         selected_words.append(word_obj)
#                         st.rerun()
    
#     st.markdown('</div>', unsafe_allow_html=True)

def create_word_buttons(available_words, selected_words, tab_key=""):
    # インデックス付きの単語リストを作成 
    word_objects = [{"id": idx, "word": word} for idx, word in enumerate(available_words)]
    
    # 初期化: ボタン押下状態をセッションステートで管理
    if f"{tab_key}_selected_word_ids" not in st.session_state:
        st.session_state[f"{tab_key}_selected_word_ids"] = []

    # CSS for button styling
    st.markdown("""
        <style>
            .stButton button {
                width: 100%;
                margin: 1px 0;
            }
        </style>
    """, unsafe_allow_html=True)

    # Display buttons in rows of 3
    for i in range(0, len(word_objects), 3):
        cols = st.columns(3)
        # Get current row of words (up to 3)
        row_words = word_objects[i:min(i+3, len(word_objects))]
        
        # Display each word in the row
        for j, word_obj in enumerate(row_words):
            is_selected = word_obj["id"] in st.session_state[f"{tab_key}_selected_word_ids"]
            with cols[j]:
                if st.button(
                    word_obj["word"],
                    key=f"{tab_key}_word_{word_obj['id']}",
                    disabled=is_selected,
                    use_container_width=True
                ):
                    if not is_selected:
                        st.session_state[f"{tab_key}_selected_word_ids"].append(word_obj["id"])
                        selected_words.append(word_obj)
                        st.rerun()

def render():
    st.title("学習モード / Study Mode")
    if 'current_sentence' not in st.session_state:
        generate_new_sentence()
    # Get user progress
    progress = get_user_progress(st.session_state.user['id'])
    # タブを作成
    tab1, tab2, tab3 = st.tabs(["学習", "統計","設定"])
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
        
        st.write("### " + st.session_state.translation)
        
        # selected_wordsの初期化
        if "selected_words" not in st.session_state:
            st.session_state.selected_words = []
            
        if "surrenderFlag" not in st.session_state:
            st.session_state.surrenderFlag = False
        
        if st.session_state.surrenderFlag:
            st.markdown("### 答え:" + st.session_state.current_sentence)
            
        # 選択された単語を表示（単語テキストのみを抽出して結合）
        selected_text = " ".join(word_obj["word"] for word_obj in st.session_state.selected_words)
        st.text_area("Your answer:", selected_text, height=100, key="tab1_answer", disabled=True)
        
        # Available words as buttons
        st.write("### Available words:")
        cols = st.columns(4)

        # ボタンの作成
        #create_word_buttons(cols, st.session_state.available_words, st.session_state.selected_words, tab_key="tab1")
        create_word_buttons(st.session_state.available_words, st.session_state.selected_words, tab_key="tab1")

        if not st.session_state.surrenderFlag:
            # Check answer button
            if st.button("Check Answer", type="primary"):
                selected_word_list = [word_obj["word"] for word_obj in st.session_state.selected_words]
                is_correct = " ".join(selected_word_list).lower() == " ".join(st.session_state.correct_words).lower()
                
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
                    st.session_state.learnimage = None
                    st.session_state[f"tab1_selected_word_ids"] = []
                    st.rerun()
                    
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Clear"):
                # 押下状態をリセット
                st.session_state.selected_words = []
                st.session_state[f"tab1_selected_word_ids"] = []  # 押されたボタンの状態をリセット
                st.rerun()
        # Control buttons
        with col2:
            if st.button("New Sentence"):
                st.session_state.surrenderFlag = False
                generate_new_sentence()
                st.session_state.learnimage = None
                st.session_state[f"tab1_selected_word_ids"] = []
                st.rerun()
        with col3:
            if st.button("Surrender",type="primary"):
                # 押下状態をリセット
                st.session_state.surrenderFlag = True
                st.session_state.learnimage = None
                st.session_state.selected_words = []
                st.session_state[f"tab1_selected_word_ids"] = []
                st.rerun()

            
    
    with tab2:
        # 統計グラフの表示
        create_performance_graphs()
    with tab3:
        setting_confi()