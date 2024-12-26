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
    # tmp = st.session_state.generate_toggle
    # toggle = st.toggle("画像生成を有効にする")
    # if tmp != toggle:
    #     st.session_state.generate_toggle = toggle
    # st.write(st.session_state.generate_toggle)
    # st.rerun()
    st.write("test")
    
def generate_new_sentence():
    # Ollamaの初期化
    chat = LLM.init_ollama()
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

def render():
    st.title("学習モード / Study Mode")
    if 'current_sentence' not in st.session_state:
        generate_new_sentence()
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
        #selected_text = " ".join(st.session_state.selected_words)
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