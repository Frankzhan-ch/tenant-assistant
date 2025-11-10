"""
=============================================================================
🏠 TENANCY ASSISTANT - Final Working Version
=============================================================================
最终工作版本 - 修复所有问题
=============================================================================
"""

import os
import re
import unicodedata
import json
import sqlite3
import uuid
import csv
from math import radians, sin, cos, asin, sqrt
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from openai import OpenAI

# ROUGE评分
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

# LangChain
try:
    from langchain_community.vectorstores import FAISS as LCFAISS
except (ModuleNotFoundError, ImportError):
    try:
        from langchain.vectorstores import FAISS as LCFAISS
    except ImportError:
        from langchain_community.vectorstores.faiss import FAISS as LCFAISS

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# =============================================================================
# ⚙️ PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Tenancy Assistant",
    page_icon="🏠",
    layout="wide"
)

# =============================================================================
# 🎨 CUSTOM CSS
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #9c27b0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏠 Tenancy Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Your AI-powered assistant for tenancy questions and property search</div>',
    unsafe_allow_html=True
)

# =============================================================================
# 🎨 SIDEBAR
# =============================================================================
with st.sidebar:
    st.header("⚙️ Settings")
    
    default_key = os.environ.get("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API Key (sk-...)",
        type="password",
        value=default_key
    )
    st.caption("💡 Prefer OPENAI_API_KEY as an environment variable")
    
    MODEL_EMBED = st.selectbox(
        "Embedding model",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0
    )
    MODEL_CHAT = st.selectbox(
        "Chat model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        index=0
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Top-K chunks", 1, 10, 6, 1)
    top_n_listings = st.slider("Max properties to show", 1, 20, 5, 1)
    max_distance_km = st.slider("Max distance (km)", 0.5, 15.0, 5.0, 0.5)
    
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    st.metric("Favorites", len(st.session_state.get("favorites", [])))
    st.metric("Viewing Requests", len(st.session_state.get("viewing_requests", [])))
    
    if st.session_state.get("vectorstore") is not None:
        chunks_count = len(st.session_state.get("doc_chunks", []))
        st.success(f"✅ Contract: {chunks_count} chunks")
    else:
        st.info("📄 No contract uploaded")

if not api_key_input:
    st.warning("⚠️ Please provide your OpenAI API key.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key_input
client = OpenAI(api_key=api_key_input)

# =============================================================================
# 🛠️ UTILITY FUNCTIONS
# =============================================================================
def normalize_plain(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\u00ad", "").replace("\xa0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def clean_page_text(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t).replace("\u00ad", "").replace("\xa0", " ")
    paras = re.split(r"\n{2,}", t)
    out = []
    for p in paras:
        p = re.sub(r"[ \t]*\n[ \t]*", " ", p)
        p = re.sub(r"\s+", " ", p).strip()
        if p:
            out.append(p)
    return "\n\n".join(out)

# =============================================================================
# 📊 ROUGE SCORING
# =============================================================================
def calculate_rouge_scores(generated_answer: str, reference_texts: list) -> dict:
    """Calculate ROUGE scores for answer quality evaluation"""
    if not ROUGE_AVAILABLE:
        return None
    
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        reference = " ".join(reference_texts)
        scores = scorer.score(reference, generated_answer)
        
        return {
            'rouge1': {
                'precision': scores['rouge1'].precision,
                'recall': scores['rouge1'].recall,
                'fmeasure': scores['rouge1'].fmeasure
            },
            'rouge2': {
                'precision': scores['rouge2'].precision,
                'recall': scores['rouge2'].recall,
                'fmeasure': scores['rouge2'].fmeasure
            },
            'rougeL': {
                'precision': scores['rougeL'].precision,
                'recall': scores['rougeL'].recall,
                'fmeasure': scores['rougeL'].fmeasure
            }
        }
    except Exception as e:
        return None

# =============================================================================
# 📄 PDF PROCESSING
# =============================================================================
def read_pdf(file: bytes):
    doc = fitz.open(stream=file, filetype="pdf")
    pages = []
    for i, p in enumerate(doc):
        raw = p.get_text("text")
        text = clean_page_text(raw)
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages

def improved_chunk_by_page(pages, chunk_size=800, overlap=100):
    chunks, metas = [], []
    
    for page_num, page_text in pages:
        paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
        
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                    metas.append({"page": page_num, "chunk_id": f"{page_num}-{chunk_idx}"})
                    chunk_idx += 1
                
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        if current_chunk:
            chunks.append(current_chunk)
            metas.append({"page": page_num, "chunk_id": f"{page_num}-{chunk_idx}"})
    
    return chunks, metas

# =============================================================================
# 🧠 RAG SYSTEM
# =============================================================================
def build_vectorstore_from_chunks(chunks, metas, embed_model_name: str):
    docs = [Document(page_content=c, metadata=m) for c, m in zip(chunks, metas)]
    embeddings = OpenAIEmbeddings(model=embed_model_name)
    vectorstore = LCFAISS.from_documents(docs, embeddings)
    return vectorstore

def make_retriever_chain(vectorstore, model_name: str, temp: float, k: int):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    llm = ChatOpenAI(model=model_name, temperature=temp)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant specializing in tenancy agreements. 
        Use the following context to answer the user's question. 
        If you cannot find the answer in the context, say so clearly.
        
        Context: {context}"""),
        ("human", "{question}")
    ])
    
    return {"retriever": retriever, "llm": llm, "prompt": prompt}

def lc_answer(qa_chain, question: str):
    retriever = qa_chain["retriever"]
    llm = qa_chain["llm"]
    prompt = qa_chain["prompt"]
    
    try:
        docs = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs])
        messages = prompt.format_messages(context=context, question=question)
        response = llm.invoke(messages)
        return response.content, docs
    except AttributeError:
        try:
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])
            messages = prompt.format_messages(context=context, question=question)
            response = llm.invoke(messages)
            return response.content, docs
        except Exception as e:
            raise Exception(f"Retrieval failed: {str(e)}")

# =============================================================================
# 🗺️ LOCATION FUNCTIONS
# =============================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def geocode_location(location_str: str) -> tuple:
    locations = {
        "orchard": (1.3048, 103.8318), "orchard road": (1.3048, 103.8318),
        "marina bay": (1.2781, 103.8540), "chinatown": (1.2830, 103.8440),
        "bugis": (1.2997, 103.8556), "sentosa": (1.2572, 103.8230),
        "clarke quay": (1.2899, 103.8467), "raffles place": (1.2837, 103.8512),
        "newton": (1.3125, 103.8382), "tampines": (1.3496, 103.9456),
        "bedok": (1.3240, 103.9301), "punggol": (1.4051, 103.9021),
        "jurong east": (1.3330, 103.7425), "woodlands": (1.4370, 103.7860),
        "bishan": (1.3507, 103.8487), "toa payoh": (1.3343, 103.8480),
        "nus": (1.2966, 103.7764), "ntu": (1.3483, 103.6831),
        "smu": (1.2969, 103.8517), "sutd": (1.3404, 103.9632),
        "central": (1.3000, 103.8500), "central region": (1.3000, 103.8500),
        "north": (1.4200, 103.8200), "east": (1.3500, 103.9400),
        "west": (1.3500, 103.7500), "north-east": (1.3800, 103.8900),
    }
    
    location_lower = location_str.lower().strip()
    if location_lower in locations:
        return locations[location_lower]
    
    best_match = None
    best_match_len = 0
    for key, coords in locations.items():
        if key in location_lower or location_lower in key:
            if len(key) > best_match_len:
                best_match = coords
                best_match_len = len(key)
    
    return best_match

def extract_location_from_text(text: str) -> tuple:
    text_lower = text.lower()
    
    location_keywords = ["near", "close to", "around", "nearby", "at", "in", "by"]
    
    for keyword in location_keywords:
        pattern = rf'\b{keyword}\s+([a-zA-Z][a-zA-Z\s]+?)(?:,|\.|$|with|budget|price|\d)'
        matches = re.findall(pattern, text_lower)
        
        for match in matches:
            location_str = match.strip()
            coords = geocode_location(location_str)
            if coords:
                return coords[0], coords[1], location_str.title()
    
    all_locations = ["orchard", "marina bay", "chinatown", "newton", "tampines", "nus", "ntu", "central"]
    
    for loc in all_locations:
        if re.search(rf'\b{loc}\b', text_lower):
            coords = geocode_location(loc)
            if coords:
                return coords[0], coords[1], loc.title()
    
    return None, None, None

# =============================================================================
# 🏠 PROPERTY SEARCH
# =============================================================================
def search_properties(query_text, max_monthly_rent, region, room_type, max_distance, top_n, db_path="property_listings.db"):
    user_lat, user_lon, location_name = extract_location_from_text(query_text)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = """
        SELECT id, name, neighbourhood_group, neighbourhood, 
               latitude, longitude, room_type, price,
               number_of_reviews, host_name, minimum_nights
        FROM properties
        WHERE (price * 30) <= ?
    """
    params = [max_monthly_rent]
    
    if region != "All":
        query += " AND neighbourhood_group = ?"
        params.append(region)
    
    if room_type != "All":
        query += " AND room_type = ?"
        params.append(room_type)
    
    cursor.execute(query, params)
    all_results = cursor.fetchall()
    conn.close()
    
    if user_lat and user_lon:
        results_with_distance = []
        for prop in all_results:
            prop_lat, prop_lon = prop[4], prop[5]
            distance = haversine_km(user_lat, user_lon, prop_lat, prop_lon)
            if distance <= max_distance:
                results_with_distance.append((prop, distance))
        
        results_with_distance.sort(key=lambda x: x[1])
        results = [r[0] for r in results_with_distance[:top_n]]
        distances = {r[0][0]: r[1] for r in results_with_distance[:top_n]}
        return results, distances, location_name
    else:
        results = sorted(all_results, key=lambda x: x[8], reverse=True)[:top_n]
        distances = {}
        return results, distances, None

# =============================================================================
# 🤖 QUERY CLASSIFIER
# =============================================================================
def classify_query(query: str) -> dict:
    system_prompt = """You are a query classifier. Classify into:
1. "contract_qa" - Questions about tenancy agreements
2. "property_search" - Looking for rental properties
3. "general" - General conversation

Respond in JSON: {"type": "...", "reasoning": "..."}"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.1,
            max_tokens=200
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"type": "general", "reasoning": "Error"}

# =============================================================================
# 💬 DISPLAY FUNCTIONS - 修复key重复问题
# =============================================================================
def display_property_card(prop, distance=None, location_name=None, unique_id=None):
    """
    显示房源卡片
    
    Args:
        prop: 房源数据
        distance: 距离（可选）
        location_name: 位置名称（可选）
        unique_id: 唯一标识符（用于避免key重复）
    """
    prop_id = prop[0]
    daily_price = prop[7]
    monthly_price = daily_price * 30
    
    # 使用unique_id来创建真正唯一的key
    # unique_id可以是消息索引或其他唯一标识
    key_suffix = f"{unique_id}" if unique_id is not None else ""
    
    is_favorited = prop_id in st.session_state.favorites
    fav_emoji = "⭐" if is_favorited else "🏠"
    
    with st.expander(f"{fav_emoji} {prop[1][:60]}... - ${monthly_price:,.0f}/month", expanded=False):
        col_info, col_actions = st.columns([2, 1])
        
        with col_info:
            st.markdown("### 📍 Location")
            st.markdown(f"**Neighborhood:** {prop[3]}, {prop[2]}")
            
            if distance is not None and location_name:
                st.markdown(f"**Distance:** {distance:.2f} km from {location_name}")
            
            st.markdown(f"[🗺️ Google Maps](https://www.google.com/maps?q={prop[4]},{prop[5]})")
            
            st.markdown("### 💰 Pricing")
            st.markdown(f"**Daily:** ${daily_price}/night")
            st.markdown(f"**Monthly:** ${monthly_price:,.0f}/month")
            st.markdown(f"**Minimum stay:** {prop[10]} nights")
            
            st.markdown("### 🏠 Details")
            st.markdown(f"**Room Type:** {prop[6]}")
            st.markdown(f"**Reviews:** {prop[8]}")
            st.markdown(f"**Host:** {prop[9]}")
        
        with col_actions:
            st.markdown("### 🎯 Actions")
            
            # 收藏按钮 - 使用unique_id避免重复
            if is_favorited:
                if st.button("💔 Remove", key=f"unfav_{prop_id}_{key_suffix}", use_container_width=True):
                    st.session_state.favorites.remove(prop_id)
                    st.success("✅ Removed from favorites!")
                    st.rerun()
            else:
                if st.button("⭐ Add to Favorites", key=f"fav_{prop_id}_{key_suffix}", use_container_width=True):
                    if prop_id not in st.session_state.favorites:
                        st.session_state.favorites.append(prop_id)
                    st.success(f"✅ Added to favorites!")
                    st.balloons()
            
            # 预约按钮 - 使用unique_id避免重复
            if st.button("📅 Schedule Viewing", key=f"schedule_{prop_id}_{key_suffix}", use_container_width=True):
                st.session_state[f"booking_open_{prop_id}_{key_suffix}"] = True
                st.rerun()
        
        # 预约表单
        if st.session_state.get(f"booking_open_{prop_id}_{key_suffix}", False):
            st.markdown("---")
            st.markdown("### 📅 Schedule a Viewing")
            
            with st.form(key=f"form_{prop_id}_{key_suffix}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Your Name*")
                    email = st.text_input("Email*")
                    phone = st.text_input("Phone")
                
                with col2:
                    date = st.date_input("Preferred Date*")
                    time = st.selectbox("Time*", 
                                       ["Morning (9am-12pm)", "Afternoon (12pm-3pm)", 
                                        "Evening (3pm-6pm)", "Flexible"])
                    duration = st.selectbox("Rental Duration",
                                          ["1-3 months", "3-6 months", "6-12 months", "1+ year"])
                
                notes = st.text_area("Notes (optional)")
                
                col_submit, col_cancel = st.columns(2)
                
                with col_submit:
                    submitted = st.form_submit_button("📤 Submit", use_container_width=True, type="primary")
                
                with col_cancel:
                    cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)
                
                if cancelled:
                    st.session_state[f"booking_open_{prop_id}_{key_suffix}"] = False
                    st.rerun()
                
                if submitted:
                    if name and email and date:
                        request = {
                            "property_id": prop_id,
                            "property_name": prop[1],
                            "name": name,
                            "email": email,
                            "phone": phone or "Not provided",
                            "date": str(date),
                            "time": time,
                            "duration": duration,
                            "notes": notes,
                            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        st.session_state.viewing_requests.append(request)
                        st.session_state[f"booking_open_{prop_id}_{key_suffix}"] = False
                        
                        st.success("✅ Viewing request submitted!")
                        st.info(f"Host will contact you at {email}")
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("⚠️ Please fill in all required fields (marked with *)")

# =============================================================================
# 🎯 SESSION STATE INITIALIZATION
# =============================================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "viewing_requests" not in st.session_state:
    st.session_state.viewing_requests = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =============================================================================
# 🏷️ MAIN TABS
# =============================================================================
tab_chat, tab_favorites, tab_viewings = st.tabs([
    "💬 Chat Assistant", 
    "⭐ My Favorites", 
    "📅 Viewing Requests"
])

# =============================================================================
# TAB 1: UNIFIED CHAT INTERFACE
# =============================================================================
with tab_chat:
    st.header("💬 AI Assistant")
    st.caption("Ask about your contract or search for properties")
    
    # PDF上传
    with st.expander("📄 Upload Tenancy Contract (Optional)", expanded=False):
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        
        if uploaded:
            with st.spinner("📖 Processing PDF..."):
                pages = read_pdf(uploaded.read())
                chunks, metas = improved_chunk_by_page(pages, chunk_size=800, overlap=100)
                
                st.session_state.doc_chunks = chunks
                st.session_state.vectorstore = build_vectorstore_from_chunks(chunks, metas, MODEL_EMBED)
                st.session_state.qa_chain = make_retriever_chain(
                    st.session_state.vectorstore, MODEL_CHAT, temperature, top_k
                )
                
                st.success(f"✅ Loaded {len(pages)} pages → {len(chunks)} chunks")
    
    # 搜索参数
    with st.expander("⚙️ Property Search Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_monthly_rent = st.number_input("Max rent (SGD)", 1000, 30000, 3000, 500)
        
        with col2:
            region_filter = st.selectbox("Region", ["All", "Central Region", "North Region", "East Region", "West Region", "North-East Region"])
        
        with col3:
            room_type_filter = st.selectbox("Room Type", ["All", "Private room", "Entire home/apt", "Shared room"])
    
    st.markdown("---")
    
    # 显示聊天历史
    for msg_idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">👤 <b>You:</b> {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 <b>Assistant:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            
            # 显示引用来源
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(msg["sources"], 1):
                        st.markdown(f"**Source {i}** (Page {source['page']})")
                        st.text(source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])
                        st.markdown("---")
            
            # 显示ROUGE分数 - 简化版，无中文
            if "rouge_scores" in msg and msg["rouge_scores"]:
                with st.expander("📊 Answer Quality (ROUGE Scores)", expanded=False):
                    scores = msg["rouge_scores"]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("ROUGE-1", f"{scores['rouge1']['fmeasure']:.3f}")
                        st.caption("Unigram overlap")
                    
                    with col2:
                        st.metric("ROUGE-2", f"{scores['rouge2']['fmeasure']:.3f}")
                        st.caption("Bigram overlap")
                    
                    with col3:
                        st.metric("ROUGE-L", f"{scores['rougeL']['fmeasure']:.3f}")
                        st.caption("Longest common subsequence")
                    
                    st.info("💡 Higher scores (closer to 1.0) indicate better answer quality")
            
            # 显示房源 - 传入msg_idx作为unique_id避免key重复
            if "properties" in msg:
                for prop_idx, prop_data in enumerate(msg["properties"]):
                    # 使用消息索引和房源索引组合成唯一ID
                    unique_id = f"msg{msg_idx}_prop{prop_idx}"
                    display_property_card(
                        prop_data["property"],
                        prop_data.get("distance"),
                        prop_data.get("location_name"),
                        unique_id=unique_id  # 传入唯一ID
                    )
    
    # 聊天输入
    st.markdown("---")
    user_input = st.chat_input("Ask about your contract or search for properties...")
    
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.spinner("🤔 Thinking..."):
            classification = classify_query(user_input)
            query_type = classification.get("type", "general")
            
            if query_type == "contract_qa":
                if st.session_state.vectorstore is None:
                    response = "Please upload your tenancy PDF first. 📄"
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    try:
                        answer, sources = lc_answer(st.session_state.qa_chain, user_input)
                        
                        source_data = [
                            {"page": doc.metadata.get("page", "?"), "content": doc.page_content}
                            for doc in sources[:3]
                        ]
                        
                        rouge_scores = None
                        if ROUGE_AVAILABLE:
                            reference_texts = [doc.page_content for doc in sources]
                            rouge_scores = calculate_rouge_scores(answer, reference_texts)
                        
                        response = f"{answer}\n\n📚 *Based on your contract*"
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response,
                            "sources": source_data,
                            "rouge_scores": rouge_scores
                        })
                    except Exception as e:
                        response = f"Error: {str(e)}"
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            elif query_type == "property_search":
                db_path = "property_listings.db"
                if not os.path.exists(db_path):
                    response = "⚠️ Database not found. Run: python init_db_from_csv.py"
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                else:
                    try:
                        results, distances, location_name = search_properties(
                            user_input, max_monthly_rent, region_filter, 
                            room_type_filter, max_distance_km, top_n_listings
                        )
                        
                        if results:
                            if location_name:
                                response = f"🎉 Found {len(results)} properties near **{location_name}**! (sorted by distance)"
                            else:
                                response = f"🎉 Found {len(results)} properties!"
                            
                            properties_data = [
                                {
                                    "property": prop,
                                    "distance": distances.get(prop[0]),
                                    "location_name": location_name
                                }
                                for prop in results
                            ]
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": response,
                                "properties": properties_data
                            })
                        else:
                            response = "😕 No properties found. Try adjusting filters."
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                    except Exception as e:
                        response = f"❌ Search error: {str(e)}"
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            else:
                response = """Hello! 👋 I can help with:

1. **Contract Questions** 📄 - Upload PDF and ask anything
2. **Property Search** 🏠 - Describe what you need (e.g., "near Orchard, $3000/month")

What would you like help with?"""
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        st.rerun()
    
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

# =============================================================================
# TAB 2: MY FAVORITES
# =============================================================================
with tab_favorites:
    st.header("⭐ My Favorite Properties")
    
    if not st.session_state.favorites:
        st.info("📭 No favorites yet. Add some from the search results!")
    else:
        st.success(f"You have {len(st.session_state.favorites)} favorite(s)")
        
        db_path = "property_listings.db"
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join('?' * len(st.session_state.favorites))
            query = f"""
                SELECT id, name, neighbourhood_group, neighbourhood, 
                       latitude, longitude, room_type, price,
                       number_of_reviews, host_name, minimum_nights
                FROM properties
                WHERE id IN ({placeholders})
            """
            cursor.execute(query, st.session_state.favorites)
            favorite_props = cursor.fetchall()
            conn.close()
            
            for fav_idx, prop in enumerate(favorite_props):
                # 使用"fav"前缀和索引作为unique_id
                display_property_card(prop, unique_id=f"fav{fav_idx}")
            
            st.markdown("---")
            if st.button("🗑️ Clear All", type="secondary"):
                st.session_state.favorites = []
                st.success("Cleared!")
                st.rerun()

# =============================================================================
# TAB 3: VIEWING REQUESTS
# =============================================================================
with tab_viewings:
    st.header("📅 My Viewing Requests")
    
    if not st.session_state.viewing_requests:
        st.info("📭 No viewing requests yet!")
    else:
        st.success(f"You have {len(st.session_state.viewing_requests)} request(s)")
        
        for i, req in enumerate(st.session_state.viewing_requests):
            with st.expander(f"🏠 {req['property_name'][:50]}... - {req['date']}", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📋 Details")
                    st.markdown(f"**Property:** {req['property_name']}")
                    st.markdown(f"**Date:** {req['date']}")
                    st.markdown(f"**Time:** {req['time']}")
                    st.markdown(f"**Duration:** {req['duration']}")
                
                with col2:
                    st.markdown("### 👤 Your Info")
                    st.markdown(f"**Name:** {req['name']}")
                    st.markdown(f"**Email:** {req['email']}")
                    st.markdown(f"**Phone:** {req['phone']}")
                
                if req['notes']:
                    st.markdown("### 📝 Notes")
                    st.text(req['notes'])
                
                st.caption(f"Submitted: {req['submitted_at']}")
                
                if st.button("🗑️ Cancel", key=f"cancel_{i}"):
                    st.session_state.viewing_requests.pop(i)
                    st.rerun()
        
        st.markdown("---")
        if st.button("💾 Export CSV"):
            import io
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["Property", "Date", "Time", "Name", "Email", "Phone", "Duration", "Notes", "Submitted"])
            
            for req in st.session_state.viewing_requests:
                writer.writerow([
                    req['property_name'], req['date'], req['time'],
                    req['name'], req['email'], req['phone'],
                    req['duration'], req['notes'], req['submitted_at']
                ])
            
            st.download_button("⬇️ Download", buf.getvalue(), "viewing_requests.csv", "text/csv")

st.markdown("---")
st.caption("🏠 Tenancy Assistant | Built with Streamlit & OpenAI")
