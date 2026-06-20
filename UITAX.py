import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
from datetime import datetime


# =========================================================
# STREAMLIT PAGE CONFIG (must be first st. call)
# =========================================================
st.set_page_config(
    page_title="FinTax AI",
    page_icon="💬",
    layout="centered",
)

# =========================================================
# THEME TOGGLE (light/dark) — session state
# =========================================================
if "theme" not in st.session_state:
    st.session_state.theme = "light"

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, time}


def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


# Colors based on the screenshot's purple chat UI
if st.session_state.theme == "light":
    bg_color = "#F3EEFC"
    card_bg = "#FFFFFF"
    bot_bubble = "#F1EEF7"
    user_bubble = "#A98EE0"
    user_text = "#FFFFFF"
    bot_text = "#1E1E1E"
    text_color = "#1E1E1E"
    subtext_color = "#6B6B6B"
    header_bg = "#FFFFFF"
    input_bg = "#FFFFFF"
    border_color = "#E5DEF5"
    # FIX: explicit light-mode input text color
    input_text_color = "#1E1E1E"
    input_placeholder_color = "#9B8EC4"
else:
    bg_color = "#1B1626"
    card_bg = "#241E33"
    bot_bubble = "#2E2640"
    user_bubble = "#7C5CC4"
    user_text = "#FFFFFF"
    bot_text = "#EDEAF5"
    text_color = "#EDEAF5"
    subtext_color = "#A89FC0"
    header_bg = "#241E33"
    input_bg = "#241E33"
    border_color = "#3A3050"
    input_text_color = "#EDEAF5"
    input_placeholder_color = "#7A6FA0"

# =========================================================
# CSS — styled after the NOVA chat screenshot
# =========================================================
st.markdown(
    f"""
    <style>
    /* Page background */
    .stApp {{
        background-color: {bg_color};
    }}

    /* Center everything into a phone-width column like the reference screenshot */
    .block-container {{
        padding-top: 0 !important;
        padding-bottom: 6rem !important;
        max-width: 480px;
        margin: 0 auto;
    }}

    /* Hide default Streamlit chrome for a cleaner app feel */
    header[data-testid="stHeader"] {{
        background: transparent;
    }}
    #MainMenu, footer {{
        visibility: hidden;
    }}

    /* Sticky top header bar */
    .fintax-header {{
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: {header_bg};
        border-radius: 0 0 18px 18px;
        padding: 16px 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        margin: 0 -1rem 20px -1rem;
        border-bottom: 1px solid {border_color};
    }}
    .fintax-avatar {{
        width: 38px;
        height: 38px;
        border-radius: 50%;
        background: linear-gradient(135deg, #A98EE0, #7C5CC4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 17px;
        flex-shrink: 0;
        box-shadow: 0 2px 6px rgba(124,92,196,0.4);
    }}
    .fintax-header-text {{
        flex: 1;
        min-width: 0;
    }}
    .fintax-header-title {{
        font-size: 16px;
        font-weight: 700;
        color: {text_color};
        margin: 0;
        line-height: 1.3;
    }}
    .fintax-header-sub {{
        font-size: 12px;
        color: #8E7CC3;
        margin: 2px 0 0 0;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 5px;
    }}
    .fintax-status-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #4CD37C;
        display: inline-block;
    }}

    /* Theme toggle button styling */
    div[data-testid="stButton"] button {{
        border-radius: 20px;
        border: 1px solid {border_color};
        background-color: {card_bg};
        color: {text_color};
        font-size: 13px;
        font-weight: 600;
        padding: 6px 14px;
        box-shadow: none;
    }}
    div[data-testid="stButton"] button:hover {{
        border-color: {user_bubble};
        color: {user_bubble};
    }}

    /* Chat rows + bubbles */
    .chat-row {{
        display: flex;
        flex-direction: column;
        margin-bottom: 16px;
        padding: 0 4px;
    }}
    .chat-row-user {{
        align-items: flex-end;
    }}
    .chat-row-bot {{
        align-items: flex-start;
    }}
    .chat-bubble-bot {{
        background-color: {bot_bubble};
        color: {bot_text};
        padding: 11px 16px;
        border-radius: 4px 16px 16px 16px;
        max-width: 82%;
        font-size: 14.5px;
        line-height: 1.55;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        word-wrap: break-word;
    }}
    .chat-bubble-user {{
        background: linear-gradient(135deg, #B49AE8, {user_bubble});
        color: {user_text};
        padding: 11px 16px;
        border-radius: 16px 4px 16px 16px;
        max-width: 82%;
        font-size: 14.5px;
        line-height: 1.55;
        box-shadow: 0 2px 6px rgba(124,92,196,0.25);
        word-wrap: break-word;
    }}
    .chat-meta {{
        font-size: 10.5px;
        color: {subtext_color};
        margin-top: 4px;
        padding: 0 6px;
        letter-spacing: 0.2px;
    }}

    /* =====================================================
       FIX: Chat input box — text visible in BOTH themes
       ===================================================== */
    .stChatInput, div[data-testid="stChatInput"] {{
        background-color: {input_bg} !important;
        max-width: 480px;
        margin: 0 auto;
    }}

    /* The textarea itself — color must be forced explicitly */
    div[data-testid="stChatInput"] textarea {{
        background-color: {input_bg} !important;
        color: {input_text_color} !important;
        border-radius: 22px !important;
        caret-color: {input_text_color} !important;
    }}

    /* Placeholder text color */
    div[data-testid="stChatInput"] textarea::placeholder {{
        color: {input_placeholder_color} !important;
        opacity: 1 !important;
    }}

    /* The wrapper div border */
    div[data-testid="stChatInput"] > div {{
        border-radius: 22px !important;
        border: 1px solid {border_color} !important;
        background-color: {input_bg} !important;
    }}

    /* Send button inside chat input */
    div[data-testid="stChatInput"] button {{
        color: {user_bubble} !important;
    }}

    /* Typing indicator / spinner text */
    .stSpinner > div > div {{
        color: {subtext_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HEADER with theme toggle
# =========================================================
header_col1, header_col2 = st.columns([5, 1])
with header_col1:
    st.markdown(
        f"""
        <div class="fintax-header">
            <div class="fintax-avatar">💬</div>
            <div class="fintax-header-text">
                <p class="fintax-header-title">Chat with FinTax AI</p>
                <p class="fintax-header-sub"><span class="fintax-status-dot"></span> Online</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    toggle_label = "🌙 Dark" if st.session_state.theme == "light" else "☀️ Light"
    st.button(toggle_label, on_click=toggle_theme, use_container_width=True)


# =========================================================
# RAG PIPELINE — unchanged logic, cached
# =========================================================

@st.cache_resource(show_spinner="Setting up FinTax AI (first run only)...")
def load_rag_pipeline():
    load_dotenv()

    docs = []
    pdf_folder = "documents"

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(pdf_folder, file))
                docs.extend(loader.load())
            except Exception as e:
                print(f"Failed: {file}")
                print(e)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = []
    for chunk in chunks:
        texts.append(chunk.page_content)

    embeds = embeddings.embed_documents(texts)
    print(len(embeds[0]))

    if os.path.exists("faiss_index"):
        vectorstore = FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local("faiss_index")

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3}
    )

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are FinTax AI, a specialized assistant for Indian taxation, accounting, and finance.

            Answer questions strictly using the information provided in the retrieved context.
            Do not rely on outside knowledge or make assumptions.

            Guidelines:
            - If the answer is not available in the retrieved documents, clearly state that the information could not be found.
            - Never fabricate GST rates, tax slabs, deductions, TDS percentages, or financial figures.
            - Provide concise yet accurate explanations.
            - Include step-by-step calculations whenever applicable.
            - Mention the source document and page reference when available.
            - Highlight any compliance requirements, limitations, or potential risks relevant to the query.
            - Present monetary amounts in Indian Rupees (₹).

            Prioritize factual accuracy, transparency, and source-backed responses.
            """
        ),
        ("human", "{data}")
    ])

    llm = ChatMistralAI(model="mistral-small-latest")

    return retriever, template, llm


retriever, template, llm = load_rag_pipeline()


# =========================================================
# HELPER: render a single message bubble
# =========================================================
def render_bubble(msg):
    row_class = "chat-row-user" if msg["role"] == "user" else "chat-row-bot"
    bubble_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
    label = "You" if msg["role"] == "user" else "FinTax AI"
    st.markdown(
        f"""
        <div class="chat-row {row_class}">
            <div class="{bubble_class}">{msg['content']}</div>
            <div class="chat-meta">{label} · {msg['time']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# RENDER FULL CHAT HISTORY
# =========================================================
for msg in st.session_state.messages:
    render_bubble(msg)

# =========================================================
# CHAT INPUT — FIX: show user message IMMEDIATELY,
# then fetch AI response, then rerun
# =========================================================
query = st.chat_input("Type your message...")

if query:
    now = datetime.now().strftime("%I:%M %p")

    if query.lower() in ["exit", "quit", "bye"]:
        user_msg = {"role": "user", "content": query, "time": now}
        bot_msg = {"role": "assistant", "content": "Goodbye!", "time": now}
        st.session_state.messages.extend([user_msg, bot_msg])
        render_bubble(user_msg)
        render_bubble(bot_msg)
        st.stop()

    else:
        # 1. Save & immediately render user bubble — no waiting
        user_msg = {"role": "user", "content": query, "time": now}
        st.session_state.messages.append(user_msg)
        render_bubble(user_msg)

        # 2. Retrieve context + call LLM (spinner shows during this)
        finds = retriever.invoke(query)
        context = "\n\n".join([find.page_content for find in finds])

        prompt = template.format_messages(
            data=f"""
        Context : {context}

        Question: {query}
        """
        )

        with st.spinner("FinTax AI is typing..."):
            response = llm.invoke(prompt)

        print(response.content)

        # 3. Save & render bot reply
        bot_msg = {"role": "assistant", "content": response.content, "time": datetime.now().strftime("%I:%M %p")}
        st.session_state.messages.append(bot_msg)
        render_bubble(bot_msg)

        # 4. Rerun to clean up state (input box reset etc.)
        st.rerun()