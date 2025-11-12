import os
import re
import unicodedata
import json
import sqlite3
import uuid
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from math import radians, sin, cos, asin, sqrt
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import fitz  # PyMuPDF

from openai import OpenAI

# ROUGE scoring library
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
    .form-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
    .citation-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.3rem;
    }
    .rouge-score {
        background-color: #d1ecf1;
        border-left: 4px solid #0c5460;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 0.3rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏠 Tenancy Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Your AI-powered assistant for tenancy questions, property search, and tenant services</div>',
    unsafe_allow_html=True
)

# =============================================================================
# 🔧 SESSION STATE INITIALIZATION
# =============================================================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "favorites" not in st.session_state:
    st.session_state.favorites = []
if "viewing_requests" not in st.session_state:
    st.session_state.viewing_requests = []
if "maintenance_requests" not in st.session_state:
    st.session_state.maintenance_requests = []
if "rent_reminders" not in st.session_state:
    st.session_state.rent_reminders = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "show_maintenance_form" not in st.session_state:
    st.session_state.show_maintenance_form = False
if "show_reminder_form" not in st.session_state:
    st.session_state.show_reminder_form = False

# =============================================================================
# =============================================================================
# SMTP CONFIGURATION
# =============================================================================
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_FROM = os.getenv("SMTP_FROM", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_USE_SSL = (SMTP_PORT == 465)

def send_email(to_email: str, subject: str, body_html: str) -> bool:
    """
    Send email via SMTP using environment variables.
    Supports SSL (465) and STARTTLS (587).
    """
    if not (SMTP_SERVER and SMTP_PORT and SMTP_FROM and SMTP_PASS):
        st.warning("⚠️ SMTP not configured. Check environment variables: SMTP_SERVER, SMTP_PORT, SMTP_FROM, SMTP_PASS")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = SMTP_FROM
        msg['To'] = to_email
        msg['Reply-To'] = SMTP_FROM

        # Attach plain text and HTML versions for better compatibility
        plain_text = re.sub(r'<[^>]+>', '', body_html or "")
        msg.attach(MIMEText(plain_text, 'plain', 'utf-8'))
        msg.attach(MIMEText(body_html or "", 'html', 'utf-8'))

        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
                server.login(SMTP_FROM, SMTP_PASS)
                server.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=20) as server:
                server.ehlo()
                server.starttls()
                server.login(SMTP_FROM, SMTP_PASS)
                server.send_message(msg)
        return True
    except Exception as e:
        st.warning(f"⚠️ Failed to send email: {e}")
        return False

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
    st.metric("Favorites", len(st.session_state.favorites))
    st.metric("Viewing Requests", len(st.session_state.viewing_requests))
    st.metric("Maintenance Requests", len(st.session_state.maintenance_requests))
    st.metric("Rent Reminders", len(st.session_state.rent_reminders))

    if st.session_state.vectorstore is not None:
        chunks_count = len(st.session_state.doc_chunks)
        st.success(f"✅ Contract: {chunks_count} chunks")
    else:
        st.info("📄 No contract uploaded")

    st.markdown("---")
    st.markdown("### 📧 Email")
    if SMTP_FROM:
        st.write(f"From: {SMTP_FROM}")
    else:
        st.warning("SMTP_FROM not configured. Email functionality disabled.")

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
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def chunk_text(text: str, max_chunk_size=800, overlap=100):
    words = text.split()
    if len(words) <= max_chunk_size:
        return [text]

    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_len = len(word) + 1
        if current_length + word_len > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_words + [word]
            current_length = sum(len(w) + 1 for w in current_chunk)
        else:
            current_chunk.append(word)
            current_length += word_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def parse_pdf(pdf_file):
    pages = []
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            text = normalize_plain(raw_text)
            pages.append((page_num, text))
        doc.close()
    except Exception as e:
        st.error(f"Error parsing PDF: {e}")
    return pages

def extract_rent_info(docs: list) -> dict:
    combined_text = " ".join([d.page_content for d in docs])

    try:
        prompt = f"""
Based on the following contract text, extract:
1. Monthly rent amount (numeric value)
2. Rent due date (day of month, e.g., "1st" or "15th")

Text: {combined_text[:3000]}

Return JSON: {{"amount": <number or null>, "due_date": <day or null>}}
"""
        response = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        rent_info = json.loads(response.choices[0].message.content)
        rent_info["found"] = (rent_info.get("amount") is not None and rent_info.get("due_date") is not None)
        return rent_info
    except Exception as e:
        return {"amount": None, "due_date": None, "found": False, "error": str(e)}

# =============================================================================
# 📊 ROUGE SCORE CALCULATION
# =============================================================================
def calculate_rouge_scores(reference_text: str, generated_answer: str) -> dict:
    if not ROUGE_AVAILABLE:
        return {"error": "ROUGE library not available"}

    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, generated_answer)

        return {
            "ROUGE-L-Precision": round(scores['rougeL'].precision, 4)
        }
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# LLM & RAG - Enhanced Intent Classification
# =============================================================================
def classify_query(query: str, has_contract: bool = False) -> dict:
    prompt = f"""
You are an intent classifier for a tenancy assistant chatbot. Analyze the user's query and classify it into ONE category.

CRITICAL DISTINCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔍 ASKING vs 📝 DOING - This is the KEY distinction!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

A. MAINTENANCE/REPAIR:
   🔍 "contract_qa" - questions about policies
   📝 "maintenance" - submit an actual problem

B. RENT/PAYMENT:
   🔍 "contract_qa" - ask about rent terms
   📝 "rent_reminder" - create reminders

CATEGORIES:
1. "contract_qa"  2. "property_search"  3. "maintenance"  4. "rent_reminder"  5. "general"

Context:
- Contract uploaded: {has_contract}

User Query: "{query}"

Return JSON ONLY: {{"type": "contract_qa|property_search|maintenance|rent_reminder|general", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)

        if has_contract:
            question_words = ["what", "when", "who", "how", "which", "where", "does", "is", "are", "can i", "should i", "do i"]
            query_lower = query.lower()
            if any(qw in query_lower for qw in question_words):
                action_words = ["submit", "report", "create", "set up", "remind me", "send", "fix my", "repair my"]
                if not any(aw in query_lower for aw in action_words):
                    if result["type"] in ["maintenance", "rent_reminder"]:
                        result["type"] = "contract_qa"
                        result["reasoning"] = "Question words without action intent - information seeking"
        return result
    except Exception as e:
        return {"type": "general", "confidence": 0.5, "reasoning": f"Error: {str(e)}"}

def build_rag_system(pages: list):
    documents = []
    all_chunks = []

    for page_num, page_text in pages:
        if not page_text.strip():
            continue
        chunks = chunk_text(page_text, max_chunk_size=800, overlap=100)
        for chunk in chunks:
            if len(chunk.strip()) > 50:
                doc = Document(page_content=chunk, metadata={"page": page_num})
                documents.append(doc)
                all_chunks.append(chunk)

    embeddings = OpenAIEmbeddings(model=MODEL_EMBED)
    vectorstore = LCFAISS.from_documents(documents, embeddings)

    llm = ChatOpenAI(model=MODEL_CHAT, temperature=temperature)

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant. Answer based on the context provided.
If the information is not in the context, say so clearly.

Context:
{context}"""),
        ("human", "{question}")
    ])

    qa_chain = {
        "vectorstore": vectorstore,
        "llm": llm,
        "prompt": prompt_template,
        "top_k": top_k
    }

    return vectorstore, qa_chain, all_chunks

def lc_answer(qa_chain, question: str):
    vectorstore = qa_chain["vectorstore"]
    llm = qa_chain["llm"]
    prompt = qa_chain["prompt"]
    k = qa_chain["top_k"]

    docs = vectorstore.similarity_search(question, k=k)
    context_str = "\n\n".join([d.page_content for d in docs])

    messages = prompt.format_messages(context=context_str, question=question)
    response = llm.invoke(messages)
    answer = response.content

    rouge_scores = calculate_rouge_scores(context_str, answer)

    citations = []
    for doc in docs:
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        citations.append({
            "page": page_num,
            "preview": preview,
            "full_text": doc.page_content
        })

    return {
        "answer": answer,
        "citations": citations,
        "rouge_scores": rouge_scores,
        "source_docs": docs
    }

# =============================================================================
# 🏠 PROPERTY SEARCH
# =============================================================================
DB_PATH = "property_listings.db"

def check_database_exists():
    if not os.path.exists(DB_PATH):
        st.error(f"❌ Database not found: {DB_PATH}")
        return False
    return True

def get_property_by_id(property_id):
    if not check_database_exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = """
            SELECT id, name, neighbourhood_group, neighbourhood, 
                   latitude, longitude, room_type, price,
                   number_of_reviews, host_name, minimum_nights
            FROM properties
            WHERE id = ?
        """
        cursor.execute(query, (property_id,))
        result = cursor.fetchone()
        conn.close()
        return result
    except Exception as e:
        st.error(f"❌ Database query error: {e}")
        return None

def get_host_info(host_name):
    if not check_database_exists():
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = """
            SELECT 
                COUNT(*) as total_listings,
                AVG(number_of_reviews) as avg_reviews,
                AVG(price) as avg_price,
                MIN(price) as min_price,
                MAX(price) as max_price,
                GROUP_CONCAT(DISTINCT room_type) as room_types
            FROM properties
            WHERE host_name = ?
            GROUP BY host_name
        """
        cursor.execute(query, (host_name,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                "name": host_name,
                "total_listings": result[0],
                "avg_reviews": round(result[1], 1) if result[1] else 0,
                "avg_price": round(result[2], 2) if result[2] else 0,
                "min_price": round(result[3], 2) if result[3] else 0,
                "max_price": round(result[4], 2) if result[4] else 0,
                "room_types": result[5] if result[5] else "N/A"
            }
        return None
    except Exception as e:
        st.error(f"❌ Error fetching host info: {e}")
        return None

def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

SINGAPORE_LOCATIONS = {
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
    "little india": (1.3067, 103.8518), "city hall": (1.2930, 103.8520),
}

def geocode_location(location_str: str) -> tuple:
    location_lower = location_str.lower().strip()
    if location_lower in SINGAPORE_LOCATIONS:
        return SINGAPORE_LOCATIONS[location_lower]
    best_match = None
    best_match_len = 0
    for key, coords in SINGAPORE_LOCATIONS.items():
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
    all_locations = ["orchard", "marina bay", "chinatown", "newton", "tampines",
                     "nus", "ntu", "central", "bugis", "jurong east"]
    for loc in all_locations:
        if re.search(rf'\b{loc}\b', text_lower):
            coords = geocode_location(loc)
            if coords:
                return coords[0], coords[1], loc.title()
    return None, None, None

def search_properties(query: str, top_n=5, max_dist_km=5.0):
    if not check_database_exists():
        return [], {}, None
    user_lat, user_lon, location_name = extract_location_from_text(query)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query_sql = """
            SELECT id, name, neighbourhood_group, neighbourhood, 
                   latitude, longitude, room_type, price,
                   number_of_reviews, host_name, minimum_nights
            FROM properties
            WHERE price IS NOT NULL
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL
        """
        cursor.execute(query_sql)
        all_results = cursor.fetchall()
        conn.close()

        if user_lat and user_lon:
            results_with_distance = []
            for prop in all_results:
                prop_lat, prop_lon = prop[4], prop[5]
                distance = haversine_km(user_lat, user_lon, prop_lat, prop_lon)
                if distance <= max_dist_km:
                    results_with_distance.append((prop, distance))
            results_with_distance.sort(key=lambda x: x[1])
            results = [r[0] for r in results_with_distance[:top_n]]
            distances = {r[0][0]: r[1] for r in results_with_distance[:top_n]}
            return results, distances, location_name
        else:
            results = sorted(all_results, key=lambda x: x[8], reverse=True)[:top_n]
            distances = {}
            return results, distances, None

    except Exception as e:
        st.error(f"❌ Database query error: {e}")
        return [], {}, None

# =============================================================================
# 📋 FORM FUNCTIONS
# =============================================================================
def show_maintenance_form():
    st.markdown("---")
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown("## 🔧 Submit Maintenance Request")

    with st.form(key="maintenance_form"):
        st.markdown("### Property Information")
        col1, col2 = st.columns(2)

        with col1:
            property_address = st.text_input("Property Address*", placeholder="123 Main St, Apt 4B")
            issue_type = st.selectbox(
                "Issue Type*",
                ["Plumbing", "Electrical", "HVAC/Air Conditioning", "Appliances",
                 "Structural", "Pest Control", "Other"]
            )
            urgency = st.selectbox("Urgency Level*", ["Low", "Medium", "High", "Emergency"])

        with col2:
            email = st.text_input("Contact Email*", placeholder="your.email@example.com")
            preferred_date = st.date_input("Preferred Date*")
            preferred_time = st.selectbox(
                "Preferred Time*",
                ["Morning (9am-12pm)", "Afternoon (12pm-3pm)", "Evening (3pm-6pm)", "Flexible"]
            )

        st.markdown("### Issue Details")
        description = st.text_area(
            "Describe the issue*",
            placeholder="Please provide a detailed description of the maintenance issue...",
            height=150
        )

        uploaded_photos = st.file_uploader(
            "Upload Photos (optional)",
            type=["jpg", "jpeg", "png", "gif"],
            accept_multiple_files=True,
            help="You can upload multiple photos of the issue"
        )

        col_submit, col_cancel = st.columns(2)

        with col_submit:
            submit = st.form_submit_button("📤 Submit Request", use_container_width=True, type="primary")

        with col_cancel:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)

        if submit:
            if property_address and issue_type and email and description:
                # Process uploaded photos
                photo_filenames = []
                if uploaded_photos:
                    for photo in uploaded_photos:
                        photo_filenames.append(photo.name)
                
                request = {
                    "id": str(uuid.uuid4())[:8],
                    "property_address": property_address,
                    "type": issue_type,
                    "urgency": urgency,
                    "email": email,
                    "preferred_date": str(preferred_date),
                    "preferred_time": preferred_time,
                    "description": description,
                    "photos": photo_filenames,
                    "status": "Pending",
                    "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                st.session_state.maintenance_requests.append(request)

                # Send confirmation email
                email_body = f"""
                <html>
                <body>
                    <h2>🔧 Maintenance Request Submitted</h2>
                    <p>Dear Tenant,</p>
                    <p>Your maintenance request has been submitted successfully. Here are the details:</p>
                    <table style="border-collapse: collapse; width: 100%;">
                        <tr style="background-color: #f2f2f2;">
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Request ID:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{request['id']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Property:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{property_address}</td>
                        </tr>
                        <tr style="background-color: #f2f2f2;">
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Issue Type:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{issue_type}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Urgency:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{urgency}</td>
                        </tr>
                        <tr style="background-color: #f2f2f2;">
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Preferred Date:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{preferred_date}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Preferred Time:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{preferred_time}</td>
                        </tr>
                    </table>
                    <p><strong>Description:</strong><br>{description}</p>
                    <p>We will contact you shortly to schedule the repair.</p>
                    <p>Best regards,<br>Tenancy Assistant</p>
                </body>
                </html>
                """
                _ = send_email(email, "Maintenance Request Received", email_body)

                st.success(f"✅ Maintenance request submitted! Request ID: {request['id']}")
                st.session_state.show_maintenance_form = False
                st.balloons()
                st.rerun()
            else:
                st.error("⚠️ Please fill in all required fields marked with *")

        if cancel:
            st.session_state.show_maintenance_form = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

def show_rent_reminder_form():
    st.markdown("---")
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown("## 💰 Set Up Rent Payment Reminder")

    rent_info = {"amount": None, "due_date": None}
    if st.session_state.vectorstore is not None:
        docs = st.session_state.vectorstore.similarity_search("rent payment amount due date", k=3)
        rent_info = extract_rent_info(docs)

    with st.form(key="rent_reminder_form"):
        st.markdown("### Rent Information")
        col1, col2 = st.columns(2)

        with col1:
            property_address = st.text_input("Property Address*", placeholder="123 Main St, Apt 4B")
            default_amount = rent_info.get("amount") if rent_info.get("found") else None
            rent_amount = st.number_input(
                "Monthly Rent Amount ($)*",
                min_value=0.0,
                value=float(default_amount) if default_amount else 0.0,
                step=50.0
            )
            default_due = rent_info.get("due_date") if rent_info.get("found") else 1
            due_date = st.number_input(
                "Rent Due Date (Day of Month)*",
                min_value=1,
                max_value=31,
                value=int(default_due) if default_due else 1
            )

        with col2:
            reminder_email = st.text_input("Reminder Email*", placeholder="your.email@example.com")
            reminder_days = st.multiselect(
                "Remind Me (Days Before Due Date)*",
                [1, 2, 3, 5, 7, 10, 14],
                default=[7, 3, 1]
            )
            reminder_duration = st.number_input(
                "Reminder Duration (Months)*",
                min_value=1,
                max_value=60,
                value=12,
                help="How many months should this reminder continue?"
            )

        st.markdown("### Additional Information (Optional)")
        landlord_info = st.text_area(
            "Landlord/Payment Details",
            placeholder="Bank account, payment method, landlord contact info...",
            height=100
        )
        notes = st.text_area(
            "Notes",
            placeholder="Any additional notes about rent payment...",
            height=80
        )

        col_submit, col_cancel = st.columns(2)
        with col_submit:
            submit = st.form_submit_button("💾 Create Reminder", use_container_width=True, type="primary")
        with col_cancel:
            cancel = st.form_submit_button("❌ Cancel", use_container_width=True)

        if submit:
            if property_address and rent_amount > 0 and reminder_email and reminder_days and reminder_duration > 0:
                today = datetime.now()
                current_month_due = datetime(today.year, today.month, due_date)

                if current_month_due < today:
                    if today.month == 12:
                        next_due = datetime(today.year + 1, 1, due_date)
                    else:
                        next_due = datetime(today.year, today.month + 1, due_date)
                else:
                    next_due = current_month_due

                earliest_reminder_days = max(reminder_days)
                next_reminder_date = next_due - timedelta(days=earliest_reminder_days)

                reminder = {
                    "id": str(uuid.uuid4())[:8],
                    "property_address": property_address,
                    "rent_amount": rent_amount,
                    "due_date": due_date,
                    "email": reminder_email,
                    "reminder_days": sorted(reminder_days, reverse=True),
                    "reminder_duration": reminder_duration,
                    "landlord_info": landlord_info,
                    "notes": notes,
                    "next_reminder": str(next_reminder_date),
                    "status": "Active",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.rent_reminders.append(reminder)

                email_body = f"""
                <html>
                <body>
                    <h2>💰 Rent Reminder Created Successfully</h2>
                    <p>Dear Tenant,</p>
                    <p>Your rent payment reminder has been set up. Here are the details:</p>
                    <table style="border-collapse: collapse; width: 100%;">
                        <tr style="background-color: #f2f2f2;">
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Reminder ID:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{reminder['id']}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Property:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{property_address}</td>
                        </tr>
                        <tr style="background-color: #f2f2f2;">
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Monthly Rent:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">${rent_amount}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Due Date:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">Day {due_date} of each month</td>
                        </tr>
                        <tr style="background-color: #f2f2f2;">
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Reminder Days:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{', '.join([str(d) + ' days before' for d in sorted(reminder_days, reverse=True)])}</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Duration:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{reminder_duration} months</td>
                        </tr>
                        <tr style="background-color: #f2f2f2;">
                            <td style="padding: 8px; border: 1px solid #ddd;"><strong>Next Reminder:</strong></td>
                            <td style="padding: 8px; border: 1px solid #ddd;">{next_reminder_date}</td>
                        </tr>
                    </table>
                    <p>You will receive email reminders at: {reminder_email}</p>
                    <p>Best regards,<br>Tenancy Assistant</p>
                </body>
                </html>
                """
                _ = send_email(reminder_email, "Rent Reminder Created", email_body)

                st.success(f"✅ Rent reminder created! Next reminder: {next_reminder_date}")
                st.session_state.show_reminder_form = False
                st.balloons()
                st.rerun()
            else:
                st.error("⚠️ Please fill in all required fields marked with *")

        if cancel:
            st.session_state.show_reminder_form = False
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# 🏠 PROPERTY DISPLAY
# =============================================================================
def display_property_card(prop, distance=None, location_name=None, key_suffix="main"):
    prop_id = str(prop[0])
    daily_price = prop[7]
    monthly_price = daily_price * 30
    host_name = prop[9]

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

            st.markdown("### 👤 Host Information")
            st.markdown(f"**Host Name:** {host_name}")

            host_info = get_host_info(host_name)
            if host_info:
                # Initialize host details display state
                host_key = f"show_host_{prop_id}_{key_suffix}"
                if host_key not in st.session_state:
                    st.session_state[host_key] = False
                
                # Toggle button for host details
                button_label = "🔍 Hide Host Details" if st.session_state[host_key] else "🔍 View Host Details"
                if st.button(button_label, key=f"btn_{host_key}"):
                    st.session_state[host_key] = not st.session_state[host_key]
                    st.rerun()
                
                # Display host details if enabled
                if st.session_state[host_key]:
                    st.markdown("---")
                    col_host1, col_host2 = st.columns(2)
                    with col_host1:
                        st.metric("Total Listings", host_info['total_listings'])
                        st.metric("Avg Reviews", host_info['avg_reviews'])
                        st.metric("Avg Price/night", f"${host_info['avg_price']:.0f}")
                    with col_host2:
                        st.metric("Price Range", f"${host_info['min_price']:.0f} - ${host_info['max_price']:.0f}")
                        st.markdown(f"**Property Types:** {host_info['room_types']}")

                    if host_info['total_listings'] >= 10:
                        experience_level = "🌟🌟🌟 Experienced Superhost"
                    elif host_info['total_listings'] >= 5:
                        experience_level = "🌟🌟 Established Host"
                    else:
                        experience_level = "🌟 New Host"
                    st.markdown(f"**Experience Level:** {experience_level}")
                    st.markdown("---")
            else:
                st.info("Host information not available")

        with col_actions:
            st.markdown("### 🎯 Actions")

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

            if st.button("📅 Schedule Viewing", key=f"schedule_{prop_id}_{key_suffix}", use_container_width=True):
                st.session_state[f"booking_open_{prop_id}_{key_suffix}"] = True
                st.rerun()

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
                            "id": str(uuid.uuid4())[:8],
                            "property_id": prop_id,
                            "property_name": prop[1],
                            "name": name,
                            "email": email,
                            "phone": phone,
                            "date": str(date),
                            "time": time,
                            "duration": duration,
                            "notes": notes,
                            "status": "Pending",
                            "submitted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                        st.session_state.viewing_requests.append(request)
                        
                        # Send confirmation email
                        email_subject = f"Viewing Appointment Confirmed - Request #{request['id']}"
                        email_body = f"""
                        <html>
                        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                            <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 10px;">
                                <h2 style="color: #667eea; border-bottom: 2px solid #667eea; padding-bottom: 10px;">
                                    🏠 Viewing Appointment Confirmed
                                </h2>
                                
                                <p>Dear {name},</p>
                                
                                <p>Your property viewing appointment has been successfully scheduled!</p>
                                
                                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0;">
                                    <h3 style="margin-top: 0; color: #667eea;">📋 Appointment Details:</h3>
                                    <p><strong>Request ID:</strong> {request['id']}</p>
                                    <p><strong>Property:</strong> {request['property_name']}</p>
                                    <p><strong>Date:</strong> {request['date']}</p>
                                    <p><strong>Time:</strong> {request['time']}</p>
                                    <p><strong>Duration:</strong> {request['duration']}</p>
                                    {f"<p><strong>Phone:</strong> {phone}</p>" if phone else ""}
                                    {f"<p><strong>Notes:</strong> {notes}</p>" if notes else ""}
                                </div>
                                
                                <p><strong>Status:</strong> <span style="color: #ffc107;">⏳ Pending</span></p>
                                
                                <p>The property owner will review your request and contact you soon to confirm the viewing.</p>
                                
                                <p style="margin-top: 30px;">
                                    Best regards,<br>
                                    <strong>Tenancy Assistant Team</strong>
                                </p>
                            </div>
                        </body>
                        </html>
                        """
                        
                        send_email(email, email_subject, email_body)
                        
                        st.success(f"✅ Viewing request submitted! ID: {request['id']}")
                        st.info("📧 Confirmation email sent to your address.")
                        st.session_state[f"booking_open_{prop_id}_{key_suffix}"] = False
                        st.balloons()
                        st.rerun()
                    else:
                        st.error("⚠️ Please fill in required fields (Name, Email, Date)")

# =============================================================================
# 🗣️ MAIN CHAT INTERFACE
# =============================================================================
st.markdown("---")

tab1, tab2 = st.tabs(["💬 Chat Assistant", "📊 My Dashboard"])

with tab1:
    st.header("💬 Chat with Your Tenancy Assistant")

    col_upload, col_clear = st.columns([3, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "📄 Upload your tenancy contract (PDF)",
            type=["pdf"],
            help="Upload your contract to ask questions about its terms"
        )

    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if uploaded_file:
        if st.session_state.vectorstore is None:
            with st.spinner("🔄 Processing your contract..."):
                pages = parse_pdf(uploaded_file)
                if pages:
                    vectorstore, qa_chain, doc_chunks = build_rag_system(pages)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = qa_chain
                    st.session_state.doc_chunks = doc_chunks
                    st.success(f"✅ Contract processed! {len(doc_chunks)} chunks indexed.")
                else:
                    st.error("❌ Failed to process PDF")

    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            st.markdown(f'<div class="chat-message user-message">👤 <strong>You:</strong><br>{content}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 <strong>Assistant:</strong><br>{content}</div>',
                        unsafe_allow_html=True)

            if "citations" in msg and msg["citations"]:
                with st.expander("📚 References from Contract", expanded=False):
                    for i, citation in enumerate(msg["citations"], 1):
                        st.markdown(f'<div class="citation-box">', unsafe_allow_html=True)
                        st.markdown(f"**Reference {i} (Page {citation['page']}):**")
                        st.markdown(f"_{citation['preview']}_")
                        st.markdown('</div>', unsafe_allow_html=True)

            if "rouge_scores" in msg and msg["rouge_scores"] and "error" not in msg["rouge_scores"]:
                with st.expander("📊 ROUGE-L Precision)", expanded=False):
                    st.markdown('<div class="rouge-score">', unsafe_allow_html=True)
                    rouge_l_precision = msg["rouge_scores"]["ROUGE-L-Precision"]
                    
                    # Display ROUGE-L Precision with 3-tier color coding
                    if rouge_l_precision > 0.6:
                        st.success(f"✅ **ROUGE-L Precision: {rouge_l_precision:.4f}** (High)")
                        st.caption("Answer is well-grounded in source text")
                    elif rouge_l_precision >= 0.3:
                        st.info(f"🟡 **ROUGE-L Precision: {rouge_l_precision:.4f}** (Medium)")
                        st.caption("Answer partially uses source text")
                    else:
                        st.warning(f"⚠️ **ROUGE-L Precision: {rouge_l_precision:.4f}** (Low)")
                        st.caption("Answer may not be well-grounded - please verify")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption("💡 ROUGE-L Precision measures the longest common subsequence overlap between the answer and source text. Higher scores indicate better grounding in retrieved context.")

            if "properties" in msg:
                st.markdown("### 🏠 Recommended Properties")
                distances = msg.get("distances", {})
                location_name = msg.get("location_name")
                for i, prop in enumerate(msg["properties"]):
                    prop_id = prop[0]
                    distance = distances.get(prop_id)
                    display_property_card(prop, distance, location_name, key_suffix=f"chat_{i}")

    if st.session_state.show_maintenance_form:
        show_maintenance_form()

    if st.session_state.show_reminder_form:
        show_rent_reminder_form()

    user_input = st.chat_input("💭 Ask me anything about your contract, search properties, or request services...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        has_contract = st.session_state.vectorstore is not None
        classification = classify_query(user_input, has_contract)
        query_type = classification["type"]

        with st.expander("🔍 Intent Detection (Debug Info)", expanded=False):
            st.json(classification)

        if query_type == "contract_qa":
            if not has_contract:
                response = "📄 Please upload your tenancy contract first so I can answer your questions about it."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                with st.spinner("🔍 Searching contract..."):
                    result = lc_answer(st.session_state.qa_chain, user_input)
                    response = result["answer"]
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "citations": result["citations"],
                        "rouge_scores": result["rouge_scores"]
                    })

        elif query_type == "property_search":
            with st.spinner("🏘️ Searching properties..."):
                results, distances, location_name = search_properties(
                    user_input,
                    top_n=top_n_listings,
                    max_dist_km=max_distance_km
                )
                if results:
                    response = f"🏘️ I found {len(results)} properties"
                    if location_name:
                        response += f" near {location_name}"
                    response += " that match your search:"
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response,
                        "properties": results,
                        "distances": distances,
                        "location_name": location_name
                    })
                else:
                    response = "😔 Sorry, I couldn't find any properties matching your criteria. Try adjusting your location or increasing the search distance."
                    st.session_state.chat_history.append({"role": "assistant", "content": response})

        elif query_type == "maintenance":
            response = "🔧 I'll help you submit a maintenance request. Please fill out the form below:"
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.show_maintenance_form = True

        elif query_type == "rent_reminder":
            response = "💰 I'll help you set up a rent payment reminder. Please fill out the form below:"
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.session_state.show_reminder_form = True

        else:
            try:
                completion = client.chat.completions.create(
                    model=MODEL_CHAT,
                    messages=[
                        {"role": "system", "content": "You are a helpful tenancy assistant. Be friendly and helpful."},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=0.7
                )
                response = completion.choices[0].message.content
            except Exception as e:
                response = f"I'm having trouble processing your request. Error: {str(e)}"

            st.session_state.chat_history.append({"role": "assistant", "content": response})

        st.rerun()

# =============================================================================
# 📊 DASHBOARD TAB
# =============================================================================
with tab2:
    st.header("📊 My Dashboard")

    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "⭐ Favorites",
        "📅 Viewing Requests",
        "🔧 Maintenance",
        "💰 Rent Reminders"
    ])

    # SUB-TAB 1: FAVORITES
    with subtab1:
        st.subheader("⭐ My Favorite Properties")

        if not st.session_state.favorites:
            st.info("❤️ No favorites yet. Start searching and add properties to your favorites!")
        else:
            st.success(f"You have {len(st.session_state.favorites)} favorite(s)")

            for prop_id in st.session_state.favorites:
                prop = get_property_by_id(prop_id)
                if prop:
                    display_property_card(prop, key_suffix=f"fav_{prop_id}")
                else:
                    st.warning(f"⚠️ Property {prop_id} not found in database")

    # SUB-TAB 2: VIEWING REQUESTS
    with subtab2:
        st.subheader("📅 Viewing Request History")

        if not st.session_state.viewing_requests:
            st.info("📭 No viewing requests yet. Schedule one from the property search!")
        else:
            st.success(f"You have {len(st.session_state.viewing_requests)} request(s)")

            for i, req in enumerate(st.session_state.viewing_requests):
                status_color = {
                    "Pending": "🟡",
                    "Confirmed": "🟢",
                    "Completed": "✅",
                    "Cancelled": "⚫"
                }.get(req["status"], "⚪")

                with st.expander(
                    f"{status_color} {req['property_name'][:50]}... - {req['date']} ({req['status']})",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 📋 Request Details")
                        st.markdown(f"**Request ID:** {req['id']}")
                        st.markdown(f"**Property:** {req['property_name']}")
                        st.markdown(f"**Name:** {req['name']}")
                        st.markdown(f"**Email:** {req['email']}")
                        if req.get('phone'):
                            st.markdown(f"**Phone:** {req['phone']}")

                    with col2:
                        st.markdown("### 📅 Schedule")
                        st.markdown(f"**Date:** {req['date']}")
                        st.markdown(f"**Time:** {req['time']}")
                        st.markdown(f"**Duration:** {req['duration']}")
                        st.markdown(f"**Status:** {status_color} {req['status']}")
                        st.markdown(f"**Submitted:** {req['submitted_at']}")

                    if req.get('notes'):
                        st.markdown("### 📝 Notes")
                        st.text(req['notes'])

                    col_status, col_delete = st.columns([3, 1])

                    with col_status:
                        new_status = st.selectbox(
                            "Update Status",
                            ["Pending", "Confirmed", "Completed", "Cancelled"],
                            index=["Pending", "Confirmed", "Completed", "Cancelled"].index(req["status"]),
                            key=f"status_{req['id']}"
                        )
                        if new_status != req["status"]:
                            if st.button("💾 Save Status", key=f"save_{req['id']}"):
                                st.session_state.viewing_requests[i]["status"] = new_status
                                st.success(f"Status updated to: {new_status}")
                                st.rerun()

                    with col_delete:
                        if st.button("🗑️ Delete", key=f"delete_{i}"):
                            st.session_state.viewing_requests.pop(i)
                            st.rerun()

            st.markdown("---")
            if st.button("💾 Export to CSV"):
                import io
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([
                    "Request ID", "Property", "Name", "Email", "Phone",
                    "Date", "Time", "Duration", "Status", "Notes", "Submitted"
                ])
                for req in st.session_state.viewing_requests:
                    writer.writerow([
                        req['id'], req['property_name'], req['name'], req['email'],
                        req.get('phone', ''), req['date'], req['time'],
                        req['duration'], req['status'], req.get('notes', ''),
                        req['submitted_at']
                    ])
                st.download_button("⬇️ Download CSV", buf.getvalue(), "viewing_requests.csv", "text/csv")

    # SUB-TAB 3: MAINTENANCE HISTORY
    with subtab3:
        st.subheader("🔧 Maintenance Request History")

        if not st.session_state.maintenance_requests:
            st.info("📭 No maintenance requests yet. Submit one from the Chat Assistant!")
        else:
            st.success(f"You have {len(st.session_state.maintenance_requests)} request(s)")

            for i, req in enumerate(st.session_state.maintenance_requests):
                status_color = {
                    "Pending": "🟡",
                    "In Progress": "🔵",
                    "Completed": "🟢",
                    "Cancelled": "⚫"
                }.get(req["status"], "⚪")

                with st.expander(
                    f"{status_color} {req['type']} - {req['property_address'][:40]}... ({req['submitted_at'].split()[0]})",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 📋 Request Details")
                        st.markdown(f"**Request ID:** {req['id']}")
                        st.markdown(f"**Property:** {req['property_address']}")
                        st.markdown(f"**Type:** {req['type']}")
                        st.markdown(f"**Urgency:** {req['urgency']}")
                        st.markdown(f"**Status:** {status_color} {req['status']}")

                    with col2:
                        st.markdown("### 📅 Schedule")
                        st.markdown(f"**Preferred Date:** {req['preferred_date']}")
                        st.markdown(f"**Preferred Time:** {req['preferred_time']}")
                        st.markdown(f"**Contact Email:** {req['email']}")
                        st.markdown(f"**Submitted:** {req['submitted_at']}")

                    st.markdown("### 📝 Description")
                    st.text(req['description'])

                    if req['photos']:
                        st.markdown("### 📸 Photos")
                        # Initialize photo display state for each maintenance request
                        if f"show_photo_{req['id']}" not in st.session_state:
                            st.session_state[f"show_photo_{req['id']}"] = {}
                        
                        # Display photo buttons and images
                        cols = st.columns(min(len(req['photos']), 4))
                        for idx, photo_name in enumerate(req['photos']):
                            with cols[idx % 4]:
                                # Toggle button for photo display
                                photo_key = f"photo_{req['id']}_{idx}"
                                is_showing = st.session_state[f"show_photo_{req['id']}"].get(photo_key, False)
                                
                                button_label = "🖼️ Hide Photo" if is_showing else f"🖼️ {photo_name}"
                                if st.button(button_label, key=f"btn_{photo_key}"):
                                    st.session_state[f"show_photo_{req['id']}"][photo_key] = not is_showing
                                    st.rerun()
                                
                                # Display photo if enabled
                                if st.session_state[f"show_photo_{req['id']}"].get(photo_key, False):
                                    try:
                                        if os.path.exists(photo_name):
                                            st.image(photo_name, use_container_width=True, caption=photo_name)
                                        else:
                                            st.info(f"📄 File: {photo_name}")
                                    except Exception as e:
                                        st.warning(f"⚠️ Cannot display: {photo_name}")

                    col_status, col_delete = st.columns([3, 1])

                    with col_status:
                        new_status = st.selectbox(
                            "Update Status",
                            ["Pending", "In Progress", "Completed", "Cancelled"],
                            index=["Pending", "In Progress", "Completed", "Cancelled"].index(req["status"]),
                            key=f"status_{req['id']}"
                        )
                        if new_status != req["status"]:
                            if st.button("💾 Save Status", key=f"save_status_{req['id']}"):
                                st.session_state.maintenance_requests[i]["status"] = new_status
                                st.success(f"Status updated to: {new_status}")
                                st.rerun()

                    with col_delete:
                        if st.button("🗑️ Delete", key=f"delete_maint_{i}"):
                            st.session_state.maintenance_requests.pop(i)
                            st.rerun()

            st.markdown("---")
            if st.button("💾 Export Maintenance History"):
                import io
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([
                    "Request ID", "Property", "Type", "Urgency", "Status",
                    "Preferred Date", "Preferred Time", "Email", "Description",
                    "Photos", "Submitted"
                ])
                for req in st.session_state.maintenance_requests:
                    writer.writerow([
                        req['id'], req['property_address'], req['type'],
                        req['urgency'], req['status'], req['preferred_date'],
                        req['preferred_time'], req['email'], req['description'],
                        "; ".join(req['photos']), req['submitted_at']
                    ])
                st.download_button(
                    "⬇️ Download CSV",
                    buf.getvalue(),
                    "maintenance_history.csv",
                    "text/csv"
                )

    # SUB-TAB 4: REMINDER HISTORY
    with subtab4:
        st.subheader("💰 Rent Reminder History")

        if not st.session_state.rent_reminders:
            st.info("📭 No rent reminders set yet. Create one from the Chat Assistant!")
        else:
            st.success(f"You have {len(st.session_state.rent_reminders)} reminder(s)")

            for i, reminder in enumerate(st.session_state.rent_reminders):
                status_icon = "🟢" if reminder["status"] == "Active" else "⚫"

                with st.expander(
                    f"{status_icon} ${reminder['rent_amount']}/month - Due Day {reminder['due_date']} ({reminder['property_address'][:40]}...)",
                    expanded=False
                ):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### 💰 Payment Details")
                        st.markdown(f"**Reminder ID:** {reminder['id']}")
                        st.markdown(f"**Property:** {reminder['property_address']}")
                        st.markdown(f"**Monthly Rent:** ${reminder['rent_amount']}")
                        st.markdown(f"**Due Date:** Day {reminder['due_date']} of each month")
                        st.markdown(f"**Status:** {status_icon} {reminder['status']}")

                    with col2:
                        st.markdown("### 📧 Reminder Settings")
                        st.markdown(f"**Email:** {reminder['email']}")
                        reminder_days_str = ", ".join([f"{d} days" for d in sorted(reminder['reminder_days'])])
                        st.markdown(f"**Remind Before:** {reminder_days_str}")
                        st.markdown(f"**Duration:** {reminder.get('reminder_duration', 'N/A')} months")
                        st.markdown(f"**Next Reminder:** {reminder['next_reminder']}")
                        st.markdown(f"**Created:** {reminder['created_at']}")

                    if reminder['landlord_info']:
                        st.markdown("### 🏦 Landlord/Payment Info")
                        st.text(reminder['landlord_info'])

                    if reminder['notes']:
                        st.markdown("### 📝 Notes")
                        st.text(reminder['notes'])

                    col_status, col_delete = st.columns([3, 1])

                    with col_status:
                        new_status = st.selectbox(
                            "Status",
                            ["Active", "Paused", "Cancelled"],
                            index=["Active", "Paused", "Cancelled"].index(reminder["status"]),
                            key=f"reminder_status_{reminder['id']}"
                        )
                        if new_status != reminder["status"]:
                            if st.button("💾 Save Status", key=f"save_reminder_status_{reminder['id']}"):
                                st.session_state.rent_reminders[i]["status"] = new_status
                                st.success(f"Reminder status updated to: {new_status}")
                                st.rerun()

                    with col_delete:
                        if st.button("🗑️ Delete", key=f"delete_reminder_{i}"):
                            st.session_state.rent_reminders.pop(i)
                            st.rerun()

            st.markdown("---")
            if st.button("💾 Export Reminder History"):
                import io
                buf = io.StringIO()
                writer = csv.writer(buf)
                writer.writerow([
                    "Reminder ID", "Property", "Rent Amount", "Due Date",
                    "Reminder Email", "Reminder Days Before", "Duration (Months)", "Next Reminder",
                    "Landlord Info", "Status", "Notes", "Created"
                ])
                for reminder in st.session_state.rent_reminders:
                    writer.writerow([
                        reminder['id'], reminder['property_address'],
                        reminder['rent_amount'], reminder['due_date'],
                        reminder['email'],
                        "; ".join([str(d) for d in reminder['reminder_days']]),
                        reminder.get('reminder_duration', 'N/A'),
                        reminder['next_reminder'], reminder['landlord_info'],
                        reminder['status'], reminder['notes'], reminder['created_at']
                    ])
                st.download_button(
                    "⬇️ Download CSV",
                    buf.getvalue(),
                    "rent_reminders.csv",
                    "text/csv"
                )

st.markdown("---")
