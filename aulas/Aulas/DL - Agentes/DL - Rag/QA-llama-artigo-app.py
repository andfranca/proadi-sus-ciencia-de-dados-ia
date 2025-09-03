import logging, re
from io import BytesIO
import streamlit as st
import fitz  # PyMuPDF
from collections import defaultdict, Counter

# === Logs silenciosos para a demo ===
logging.basicConfig(level=logging.WARNING)
for n in ("httpx","httpcore","langchain","llama_index","llama_index.core"):
    lg = logging.getLogger(n); lg.setLevel(logging.WARNING); lg.propagate = False

# ===== LangChain (APIs atuais) =====
from typing import Any, List, Optional
from langchain_ollama import ChatOllama
from langchain_core.documents import Document as LCDocument
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ===== LlamaIndex (camada de dados) =====
from llama_index.core import VectorStoreIndex, Document as LIDocument
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding

# ---------------------------
# PDF -> texto
# ---------------------------
def pdf_to_text(file_bytes: bytes) -> str:
    doc = fitz.open(stream=BytesIO(file_bytes), filetype="pdf")
    texts = []
    for page in doc:
        t = page.get_text("text")
        if t: texts.append(t)
    return "\n".join(texts)

# ---------------------------
# Seções estilo PubMed (heurística leve)
# ---------------------------
HEAD_RX = re.compile(
    r"^\s*(\d+(\.\d+)?\s+)?("
    r"ABSTRACT|RESUMO|BACKGROUND|INTRODUCTION|INTRODUÇÃO|METHODS?|MATERIALS AND METHODS|MÉTODOS|"
    r"RESULTS?|DISCUSSION|DISCUSSÃO|CONCLUSION(S)?|CONCLUSÕES|ACKNOWLEDGMENTS|AGRADECIMENTOS|"
    r"REFERENCES|REFERÊNCIAS|BIBLIOGRAPHY"
    r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

def _canon(h: str) -> str:
    u = h.upper()
    if "ABSTRACT" in u or "RESUMO" in u: return "Abstract"
    if "BACKGROUND" in u: return "Background"
    if "INTRODU" in u: return "Introduction"
    if "METHOD" in u or "MATERIALS" in u or "MÉTODO" in u: return "Methods"
    if "RESULT" in u: return "Results"
    if "DISCUSS" in u: return "Discussion"
    if "CONCLUSION" in u or "CONCLUS" in u: return "Conclusions"
    if "ACKNOWLEDG" in u or "AGRADEC" in u: return "Acknowledgments"
    if "REFER" in u or "BIBLIO" in u: return "References"
    return h.title()

def split_pubmed_sections(full_text: str) -> dict:
    lines = full_text.splitlines()
    idxs = [i for i, line in enumerate(lines) if HEAD_RX.match(line.strip())]
    if not idxs:
        return {"Body": full_text}
    idxs.append(len(lines))
    out = {}
    for i in range(len(idxs)-1):
        header = lines[idxs[i]].strip()
        body = "\n".join(lines[idxs[i]+1: idxs[i+1]]).strip()
        if body:
            out[_canon(header)] = (out.get(_canon(header), "") + ("\n" if _canon(header) in out else "") + body).strip()
    return out or {"Body": full_text}

# ---------------------------
# Referências (heurística simples para PDF)
# ---------------------------
HDR_REF = re.compile(r"(?mi)^\s*(REFERENCES|REFERÊNCIAS|BIBLIOGRAPHY)\s*:?\s*$")
def extract_references(text: str) -> list[str]:
    m = HDR_REF.search(text)
    if not m:
        return []
    block = text[m.end():].strip()
    block = re.sub(r"-\s*\n\s*", "", block)          # des-hifenizar
    block = re.sub(r"(?<!\n)\n(?!\n)", " ", block)  # juntar linhas simples
    parts = re.split(r"(?m)^\s*(?:\[\d+\]|\d+[.)-])\s+|\n\s*\n", block)
    refs, seen = [], set()
    for p in parts:
        s = p.strip().strip(" .;")
        if len(s) >= 30:
            k = s.lower()[:160]
            if k not in seen:
                refs.append(s); seen.add(k)
    return refs

# ---------------------------
# LlamaIndex -> BaseRetriever (aplica filtro por seção pós-retrieval)
# ---------------------------
class LlamaIndexRetriever(BaseRetriever):
    li_retriever: Any
    section_eq: Optional[str] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[LCDocument]:
        results = self.li_retriever.retrieve(query)
        docs: List[LCDocument] = []
        for r in results:
            node = getattr(r, "node", r)
            text = getattr(node, "get_content", lambda: None)() or getattr(node, "text", "") or ""
            meta = dict(getattr(node, "metadata", {}) or {})
            # filtro pós-retrieval por seção (robusto com SimpleVectorStore)
            if self.section_eq and meta.get("section") != self.section_eq:
                continue
            docs.append(LCDocument(page_content=text, metadata=meta))
        return docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[LCDocument]:
        return self._get_relevant_documents(query, run_manager=run_manager)

# ---------------------------
# Construir índice (LlamaIndex)
# ---------------------------
def build_index_from_sections(sections: dict, source_name: str, chunk_size=1200, overlap=150):
    li_docs = []
    for sec, text in sections.items():
        if text and len(text.strip()) >= 20:
            li_docs.append(LIDocument(text=text, metadata={"section": sec, "source": source_name}))
    nodes = SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, paragraph_separator="\n\n"
    ).get_nodes_from_documents(li_docs)
    index = VectorStoreIndex(nodes, embed_model=OllamaEmbedding("nomic-embed-text"))
    return index

# ---------------------------
# Prompt PT-BR (para create_stuff_documents_chain)
# ---------------------------
RAG_PROMPT = ChatPromptTemplate.from_template(
    "Você é um assistente para revisão rápida de literatura, em PT-BR.\n"
    "Responda de forma objetiva **APENAS** com base no CONTEXTO.\n"
    "Se faltar evidência, responda 'Não sei'. Quando possível, cite a seção (Methods, Results etc.).\n\n"
    "Pergunta: {input}\n\n"
    "CONTEXTO:\n{context}\n\nResposta:"
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Revisão Rápida — LangChain + LlamaIndex + Ollama", page_icon="📄", layout="centered")
st.title("📄 Revisão Rápida de Literatura — LangChain + LlamaIndex + Ollama")

with st.sidebar:
    st.markdown("**Como usar**")
    st.markdown("1) Envie 1 PDF com **texto selecionável**.\n2) Clique **Processar**.\n3) Pergunte.")
    st.divider()
    chunk_size = st.slider("chunk_size", 600, 2000, 1200, 100)
    overlap    = st.slider("overlap", 0, 400, 150, 10)
    top_k      = st.slider("top_k (trechos)", 1, 12, 6, 1)
    st.caption("Dica: ↑k = mais completude; ↓k = mais foco.")

uploaded = st.file_uploader("Envie um artigo (PDF)", type=["pdf"])

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.sections = {}
    st.session_state.refs = []
    st.session_state.source_name = ""

col1, col2 = st.columns(2)
with col1:
    if st.button("⚙️ Processar", use_container_width=True):
        if not uploaded:
            st.warning("Envie um PDF primeiro.")
        else:
            raw = uploaded.getvalue()
            text = pdf_to_text(raw)
            if not text or len(text.strip()) < 100:
                st.error("Não foi possível extrair texto (PDF pode estar escaneado).")
            else:
                secs = split_pubmed_sections(text)
                index = build_index_from_sections(secs, uploaded.name, chunk_size, overlap)
                st.session_state.index = index
                st.session_state.sections = secs
                st.session_state.refs = extract_references(text)
                st.session_state.source_name = uploaded.name
                st.success("Artigo processado e indexado.")

with col2:
    if st.session_state.index:
        st.success("Pronto para perguntas.")
    else:
        st.info("Aguardando processamento…")

st.divider()

# Seções detectadas
if st.session_state.sections:
    st.subheader("Seções detectadas")
    st.write(" • ".join(f"`{k}`" for k in st.session_state.sections.keys()))

# Referências extraídas
if st.session_state.refs:
    st.subheader("Referências (heurística de PDF)")
    for i, r in enumerate(st.session_state.refs, 1):
        st.markdown(f"{i}. {r}")

# Q&A
st.subheader("Pergunte ao artigo")
question = st.text_input("Ex.: Qual a principal conclusão do estudo?")
sec_list = ["(todas)"] + list(st.session_state.sections.keys())
sec_sel  = st.selectbox("Escopo (seção)", sec_list, index=0)

if st.button("Responder", type="primary"):
    if not st.session_state.index:
        st.error("Processe o PDF primeiro.")
    elif not question.strip():
        st.error("Digite uma pergunta.")
    else:
        # LlamaIndex retriever: se filtrar por seção, pegue um pouco mais de candidatos
        base_k = top_k * 3 if sec_sel != "(todas)" else top_k
        li_retriever = st.session_state.index.as_retriever(similarity_top_k=base_k)

        retriever = LlamaIndexRetriever(li_retriever=li_retriever, section_eq=None if sec_sel=="(todas)" else sec_sel)

        # LLM + chains atuais (sem deprecations)
        llm = ChatOllama(model="llama3:8b", temperature=0)
        combine_docs_chain = create_stuff_documents_chain(llm, RAG_PROMPT)
        rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

        with st.spinner("Consultando…"):
            result = rag_chain.invoke({"input": question})

        answer = result.get("answer", "Não sei.")
        st.markdown("### Resposta")
        st.write(answer)



        st.caption("Fontes")
        srcs = result.get("context") or []  # lista de Documents

        if srcs:
            for i, d in enumerate(srcs, 1):
                src = d.metadata.get("source", st.session_state.source_name)
                sec = d.metadata.get("section", "")
                if sec:
                    st.markdown(f"- {i}. **{src}** — _{sec}_")
                else:
                    st.markdown(f"- {i}. **{src}**")
        else:
            st.write("—")