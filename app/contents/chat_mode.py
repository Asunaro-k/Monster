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
import asyncio


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
                
                # prompt_with_image = f"""キーワードに基づいた簡単な英会話をあなたとしたいです。
                # 以下に例を張ります。例なので猫などの内容は無視してください。\n
                # A: Look at the cat! It's sitting on the floor in front of the kitchen. (見て！猫がキッチンの前の床に座ってるよ。)\n
                # B: Yeah, it looks so relaxed! (ああ、まったりしてるね！)\n
                # A: I know, right? Maybe it's waiting for food. (そうだよね？もしかしたらご飯が食べたいから待ってるのかな。)\n
                # B: Do you think the cat is hungry? (猫はおなかすいてるかな？)\n
                # A: Hmm, maybe. Cats love food! (えー、もしかしたら。猫は食べ物が大好きなんだよ！)\n\n
                # Question: What do you think the cat would say if it could talk? (猫が話すことができたら何と言うかな？)\n
                
                # 上記のようにキーワードに基づいて何回か日本語訳をつけた英会話をしたうえで、最後に日本語訳をつけた英語で私に何か質問か問いかけをしてください。キーワード：
                # {caption}

                # """
                prompt_with_image = f"""
                以下のキーワードを使った日本語訳付きの簡単な英会話を作成してください。  
                1. **キーワードに関連した1つの短い英語の文**（日本語訳を添えること）。  
                2. **その文に基づいた、英語の質問を1つ**（日本語訳を添えること）。  

                例（キーワード: 猫）:  
                英語: Look at the cat! It's sitting on the floor in front of the kitchen. \n 
                日本語訳: 見て！猫がキッチンの前の床に座ってるよ。  \n

                質問: What do you think the cat would say if it could talk?  \n
                日本語訳: 猫が話すことができたら何と言うかな？  \n

                以下の形式に従って作成してください。  

                キーワード: {caption}  
                """
                
                #response = await st.session_state.llm.apredict(prompt_with_image)
                reply = chain.invoke(prompt_with_image)
                #reply = await chain.ainvoke(prompt_with_image)
                response = reply['response']
                await chat_history.add_message(prompt_with_image, is_bot=False)
                await chat_history.add_message(response, is_bot=True)
                st.session_state['questionprompt'] = response
                st.session_state['previous_uploaded_file'] = image_file
                # analysis = await question_chain.ainvoke(st.session_state.get('questionprompt', ''))
                analysis = question_chain.invoke(st.session_state.get('questionprompt', ''))
                content = analysis.content if hasattr(analysis, 'content') else str(analysis)
                st.session_state['needs_question'] = "NEEDS_QUESTION: true" in content
                if st.session_state['needs_question']:
                    question_query = re.search(r'QUESTION_QUERY: (.*)', content)
                    st.session_state['question_query'] = question_query.group(1)
                    st.session_state['needs_question'] = False
                
            else:
                st.markdown("Question")
                if st.session_state['question_query']:
                    st.markdown(f"{st.session_state['question_query']}")
                # 同じ画像の場合は英会話の正誤判定サポート
                # prompt_with_support = f"""
                
                # あなたの目標は、ユーザーが楽しく英会話を練習し、上達できるようにサポートすることです。
                # 次の文章の英語の正しさを日本語で評価し、あっている場合は褒めてください。
                # 英語ではない場合や間違っている場合は修正を提案し、修正した英語を話してください。
                # またその後も会話を続けます。\n
                # 以下に例を張ります。例なので岩の崖などの内容は無視してください。\n
                # Question: What do you think is the most beautiful rocky cliff with a body of water in the world?\n
                # 上記のように日本語訳をつけた英語で私に何か質問か問いかけをしてください。
                # 前回の出力で次のような会話を行いました。：{st.session_state.get('questionprompt', '')}\n
                # 質問の回答: {prompt}"""
                prompt_with_support = f"""
                あなたの目標は、ユーザーが楽しく英会話を練習し、上達できるようにサポートすることです。
                会話の文脈を読み取り、次の質問の回答の文章の英語の正しさを日本語で評価し、あっている場合は、どの部分が良いか具体的に褒めてください。
                英語ではない場合や文脈的に間違っている場合や短すぎて文章として足りない場合は、どの点が問題かを簡潔に説明し、修正案を日本語で提案した後、修正した英語を提示してください。
                "評価は必ず日本語でわかりやすく記述してください。"
                また、その後も会話を続けます。下記のように日本語訳をつけた英語で私に何か質問か問いかけをしてください。

                以下に例を張ります。例なので羊などの内容は無視してください。:
                
                あなたの回答 "sheep's dreams is flying sky" については、いくつか問題があります。

                **問題点:**

                * **文法的エラー:** "sheep's dreams is" は文法的に正しくありません。主語が複数形なので、動詞も複数形にしなければなりません。つまり、"sheep's dreams are" になります。
                * **単語の選択:** "flying sky" という表現は、意味としては理解できますが、自然な英語ではありません。夢の中で飛んでいることを表現したい場合は、"flying in the sky" または "soaring through the air" などの表現がより適しています。

                **修正案:**

                Sheep's dreams are flying in the sky.

                これを英訳すると： 羊の夢の中では空中を飛んでいるようです。

                さて、私から次の質問です：What is the most memorable dream you had while you were sleeping?（あなたは眠っている間に経験した一番印象に残る夢については何ですか？）

                
                前回の出力で次のような会話を行いました：{st.session_state.get('questionprompt', '')}
                質問の回答: {prompt}
                """
                
                #response = await st.session_state.llm.apredict(prompt_with_support)
                reply = chain.invoke(prompt_with_support)
                #reply = asyncio.run(chain.ainvoke(prompt_with_support))
                response = reply['response']
                st.session_state['questionprompt'] = response
                await chat_history.add_message(prompt_with_support, is_bot=False)
                await chat_history.add_message(response, is_bot=True)
                analysis = question_chain.invoke(response)
                #analysis = await question_chain.ainvoke(response)
                content = analysis.content if hasattr(analysis, 'content') else str(analysis)
                st.session_state['needs_question'] = "NEEDS_QUESTION: true" in content
                if st.session_state['needs_question']:
                    question_query = re.search(r'QUESTION_QUERY: (.*)', content)
                    st.session_state['question_query'] = question_query.group(1)
                    st.session_state['needs_question'] = False
        else:
            # 通常のテキストベースの処理
            #recent_history = await chat_history.get_recent_history()
            # analysis = await query_chain.ainvoke(prompt)
            analysis = query_chain.invoke(prompt)
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
                    # reply = await chain.ainvoke(prompt_with_content)
                    reply = chain.invoke(prompt_with_content)
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
                    reply = chain.invoke(prompt_with_search)
                    #reply = await chain.ainvoke(prompt_with_search)
                    response = reply['response']
                else:
                    response = "申し訳ありません。検索クエリの生成に失敗しました。"
            else:
                #st.markdown(st.session_state.memory)
                #reply = await chain.ainvoke(prompt)
                reply = chain.invoke(prompt)
                response = reply['response']

        # 応答の表示
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        
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
            
def init_session_state():
    """セッション状態の初期化"""
    MAX_MEMORY_LIMIT = 10
    #保存済みのモデルをロード
    if "image_captioner" not in st.session_state:
        st.session_state.image_captioner = load_caption_model()
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'memory' not in st.session_state or st.session_state.memory is None:
        st.session_state.memory = ConversationBufferMemory()
    if 'llm' not in st.session_state:
        st.session_state.llm = LLM.init_ollama()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ChatHistory(
            system_prompt='あなたは知識豊富なアシスタントです。会話を良く理解し、適切な返答を行います。'
        )
    # メッセージ上限のチェックと古いメッセージ削除
    if len(st.session_state.memory.chat_memory.messages) > MAX_MEMORY_LIMIT:
        st.session_state.memory.chat_memory.messages = st.session_state.memory.chat_memory.messages[-MAX_MEMORY_LIMIT:]
        
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

def create_sidebar():
    with st.sidebar:
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

def render():
    st.title("チャットモード / Chat Mode")
    # セッション状態の初期化
    init_session_state()
    create_sidebar()

    # DuckDuckGo検索の初期化
    st.session_state.search = DuckDuckGoSearchAPIWrapper(
        backend="api",
        max_results=5,
        region="jp-jp",
        safesearch="off",
        source="text",
        time="w"
    )
    st.session_state.image_flag = False
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