### Código desenvolvido em aula e interface desenvolvida com ajuda do chatgpt-5

import streamlit as st
from io import BytesIO

# LlamaIndex
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# PDF -> texto (PyMuPDF)
import fitz  # pip install pymupdf

st.set_page_config(page_title="Chat RAG (LlamaIndex + Ollama)", page_icon="🧠", layout="centered")
st.title("🧠 Chat RAG — LlamaIndex + Ollama (simples)")

with st.sidebar:
    st.markdown("**Como usar**")
    st.markdown("1) Envie **TXT** ou **PDF com texto** (não escaneado).\n2) Clique **Construir índice**.\n3) Converse.")
    st.markdown("**Modelos locais:** `llama3` (LLM) e `nomic-embed-text` (embeddings).")
    st.divider()
    chunk_size = st.slider("Tamanho do chunk", 600, 2000, 1200, 100)
    chunk_overlap = st.slider("Overlap", 0, 400, 200, 20)
    top_k = st.slider("Trechos recuperados (k)", 1, 12, 8, 1)

uploads = st.file_uploader("Envie 1+ arquivos (.txt ou .pdf com texto)", type=["txt", "pdf"], accept_multiple_files=True)

# Estado da sessão
if "chat" not in st.session_state:
    st.session_state.chat = None
if "msgs" not in st.session_state:
    st.session_state.msgs = []

def extract_text_from_pdf(file_bytes: bytes) -> str:
    doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    return "".join(page.get_text() for page in doc)

def build_chat_engine(files):
    # Documentos
    docs = []
    for f in files:
        name = f.name
        ext = name.lower().rsplit(".", 1)[-1]
        if ext == "txt":
            text = f.getvalue().decode("utf-8", errors="ignore")
        elif ext == "pdf":
            text = extract_text_from_pdf(f.getvalue())
        else:
            continue
        if text.strip():
            docs.append(Document(text=text, metadata={"filename": name}))
    if not docs:
        raise RuntimeError("Nenhum texto válido encontrado. Use TXT ou PDF com texto selecionável.")

    # Chunking (explícito e estável)
    nodes = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator="\n\n"
    ).get_nodes_from_documents(docs)

    # Índice + LLM locais (PT-BR)
    index = VectorStoreIndex(nodes, embed_model=OllamaEmbedding("nomic-embed-text"))
    llm = Ollama(
        model="llama3:8b",
        temperature=0,
        system_prompt=(
            "Você é um assistente que responde SEMPRE em português do Brasil. "
            "Baseie-se APENAS nos trechos recuperados. "
            "Se faltar evidência, diga 'Não sei'. Seja claro e objetivo."
        ),
    )

    # Chat conversacional com memória + RAG
    chat = index.as_chat_engine(
        llm=llm,
        chat_mode="condense_question",  # reescreve follow-ups usando histórico
        similarity_top_k=top_k,
    )
    return chat

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Construir índice", use_container_width=True):
        if not uploads:
            st.warning("Envie pelo menos um arquivo TXT ou PDF com texto.")
        else:
            try:
                st.session_state.chat = build_chat_engine(uploads)
                st.session_state.msgs = []  # reinicia o histórico visual
                st.success("Índice criado. Pode começar a conversar.")
            except Exception as e:
                st.session_state.chat = None
                st.error(f"Erro ao construir índice: {e}")

with col2:
    if st.button("🧹 Limpar conversa", use_container_width=True):
        st.session_state.msgs = []

st.divider()

# Mostra histórico
for m in st.session_state.msgs:
    st.chat_message(m["role"]).markdown(m["content"])

# Entrada do usuário
user_input = st.chat_input("Digite sua pergunta (ex.: Princípios do SUS na Lei 8.080)")
if user_input:
    st.session_state.msgs.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    if not st.session_state.chat:
        st.chat_message("assistant").markdown("⚠️ Construa o índice antes (envie arquivos e clique em **Construir índice**).")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Consultando…"):
                resp = st.session_state.chat.chat(user_input)
            answer = getattr(resp, "text", None) or getattr(resp, "response", "") or "Não sei."
            st.markdown(answer)
            # Fontes usadas neste turno
            srcs = getattr(resp, "source_nodes", []) or []
            if srcs:
                with st.expander("Fontes usadas"):
                    for i, s in enumerate(srcs, 1):
                        meta = getattr(s.node, "metadata", {})
                        st.markdown(f"{i}. **{meta.get('filename', 'arquivo')}**")
            # Guarda no histórico visual
            st.session_state.msgs.append({"role": "assistant", "content": answer})