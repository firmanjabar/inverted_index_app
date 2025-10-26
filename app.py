# app.py
# Streamlit Inverted Index App (no external NLP deps; only Streamlit)
# Features:
# - Load corpus from CSV (choose column), TXT (one doc per line), or pasted text
# - Preprocessing (lowercase, remove digits/punct, stopwords EN/ID built-in)
# - Build inverted index with positions: term -> {df, postings: {doc_id: [positions,...]}}
# - Explore vocabulary & postings
# - Boolean search (AND, OR, NOT) and phrase search with quotes: "natural language"
# - Save/Load index as JSON; export vocabulary CSV
#
# Run: streamlit run app.py

import csv
import io
import json
import math
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

import streamlit as st

st.set_page_config(page_title="Inverted Index (Streamlit)", page_icon="📚", layout="wide")

# -------------------- Built-in Stopwords (small, EN & ID) --------------------
STOPWORDS_EN = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it",
    "its","of","on","that","the","to","was","were","will","with","this","these","those",
    "or","not","but","we","you","your","our"
}

STOPWORDS_ID = {
    "yang","dan","di","ke","dari","untuk","pada","dengan","adalah","itu","ini","ada",
    "atau","tidak","saya","kami","kita","anda","dia","mereka","sebagai","sebuah","para",
    "dalam","ke","sebuah","akan","lagi","serta","atau"
}

# -------------------- Tokenization & Preprocess --------------------
def tokenize(text: str, lowercase=True, remove_digits=False, remove_punct=True,
             lang="id", use_stop=True) -> List[str]:
    if lowercase:
        text = text.lower()
    if remove_digits:
        text = re.sub(r"\d+", " ", text)
    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split() if text else []
    if use_stop:
        sw = STOPWORDS_ID if lang == "id" else STOPWORDS_EN
        tokens = [t for t in tokens if t not in sw]
    return tokens

# -------------------- Inverted Index Builder --------------------
def build_inverted_index(docs: List[str], **tokkw) -> Dict:
    vocab = defaultdict(lambda: {"df": 0, "postings": defaultdict(list)})
    for doc_id, text in enumerate(docs):
        tokens = tokenize(text, **tokkw)
        for pos, term in enumerate(tokens):
            postings = vocab[term]["postings"]
            postings[doc_id].append(pos)
        # set df
    for term, entry in vocab.items():
        entry["df"] = len(entry["postings"])
        entry["postings"] = dict(entry["postings"])  # make JSON-serializable
    return {"index": dict(vocab), "num_docs": len(docs)}

def vocabulary_stats(index: Dict) -> List[Tuple[str, int, int]]:
    rows = []
    for term, entry in index["index"].items():
        df = entry["df"]
        cf = sum(len(positions) for positions in entry["postings"].values())
        rows.append((term, df, cf))
    rows.sort(key=lambda x: (-x[1], x[0]))  # sort by df desc, then term
    return rows

# -------------------- Search utils --------------------
def boolean_search(query: str, index: Dict) -> Set[int]:
    """
    Simple parser for operators: AND, OR, NOT (uppercase). Tokens default to OR if only one?
    We implement left-to-right with precedence: NOT > AND > OR
    """
    tokens = query.strip().split()
    if not tokens:
        return set()

    def term_docs(term: str) -> Set[int]:
        entry = index["index"].get(term)
        return set(entry["postings"].keys()) if entry else set()

    # Shunting-yard style precedence
    out_stack = []
    op_stack = []

    def apply_op():
        op = op_stack.pop()
        if op == "NOT":
            a = out_stack.pop()
            universe = set(range(index["num_docs"]))
            out_stack.append(universe - a)
        else:
            b = out_stack.pop()
            a = out_stack.pop()
            if op == "AND":
                out_stack.append(a & b)
            elif op == "OR":
                out_stack.append(a | b)

    prec = {"NOT":3, "AND":2, "OR":1}
    for tok in tokens:
        if tok in ("AND","OR","NOT"):
            while op_stack and prec.get(op_stack[-1],0) >= prec[tok]:
                apply_op()
            op_stack.append(tok)
        else:
            out_stack.append(term_docs(tok))
    while op_stack:
        apply_op()
    return out_stack[-1] if out_stack else set()

def phrase_search(phrase: str, index: Dict) -> Set[int]:
    """
    Phrase format: words separated by space. Use positional intersection.
    """
    words = phrase.split()
    if not words:
        return set()
    # Start with documents containing first word
    first = index["index"].get(words[0])
    if not first:
        return set()
    candidate_docs = set(first["postings"].keys())
    for w in words[1:]:
        entry = index["index"].get(w)
        if not entry:
            return set()
        candidate_docs &= set(entry["postings"].keys())
        if not candidate_docs:
            return set()

    result = set()
    for d in candidate_docs:
        positions_lists = [index["index"][w]["postings"][d] for w in words]
        # Check for positions where p2=p1+1, p3=p2+1, ...
        # Use set intersection trick
        shifted_sets = [set(positions_lists[0])]
        for i in range(1, len(words)):
            shifted_sets.append({p - i for p in positions_lists[i]})
        if set.intersection(*shifted_sets):
            result.add(d)
    return result

def highlight_snippet(text: str, terms: List[str], radius: int = 40) -> str:
    # Simple highlight by surrounding terms with ** ** and trimming radius around first match
    pattern = re.compile(r"(" + "|".join(map(re.escape, terms)) + r")", re.IGNORECASE)
    m = pattern.search(text)
    if not m:
        return text[:radius*2] + ("..." if len(text) > radius*2 else "")
    start = max(0, m.start() - radius)
    end = min(len(text), m.end() + radius)
    snippet = text[start:end]
    return pattern.sub(r"**\1**", snippet) + ("..." if end < len(text) else "")

# -------------------- UI --------------------
st.title("📚 Inverted Index — Streamlit App")

st.markdown(
    """
    Aplikasi ini membangun **Inverted Index** dari korpus Anda dan mendukung:
    - Eksplorasi kosakata (DF/CF) dan postings list
    - **Boolean search** (`AND`, `OR`, `NOT`)
    - **Phrase search** dengan tanda kutip, misalnya: `"natural language processing"`
    - Simpan/Muat index sebagai **JSON**
    """
)

# -------- Sidebar: Preprocess options --------
with st.sidebar:
    st.header("⚙️ Pengaturan Pra-proses")
    lang = st.selectbox("Bahasa (stopwords built-in)", ["id", "en"], index=0)
    lowercase = st.checkbox("Lowercase", True)
    remove_digits = st.checkbox("Hapus angka", False)
    remove_punct = st.checkbox("Hapus tanda baca", True)
    use_stop = st.checkbox("Buang stopwords (built-in)", True)

# -------- Tabs: Input data --------
st.subheader("1) Muat Korpus")
tab_csv, tab_txt, tab_paste, tab_load_index = st.tabs(["📤 CSV", "📄 TXT (per baris)", "📝 Tempel Teks", "📦 Muat Index JSON"])

docs: List[str] = []
doc_ids: List[str] = []

with tab_csv:
    st.caption("Unggah CSV berisi teks. Pilih nama kolom teks.")
    up = st.file_uploader("Upload CSV", type=["csv"], key="csv_up")
    col_name = st.text_input("Nama kolom teks", value="text", key="csv_col")
    if up is not None:
        try:
            # Try utf-8 then fallback
            content = up.read()
            try:
                s = content.decode("utf-8")
            except UnicodeDecodeError:
                s = content.decode("latin-1", errors="ignore")
            reader = csv.DictReader(io.StringIO(s))
            if col_name not in reader.fieldnames:
                st.error(f"Kolom '{col_name}' tidak ada. Kolom tersedia: {reader.fieldnames}")
            else:
                tmp_docs = []
                for row in reader:
                    tmp_docs.append(str(row.get(col_name, "")))
                if tmp_docs:
                    docs = tmp_docs
                    doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
                    st.success(f"Memuat {len(docs)} dokumen dari CSV.")
        except Exception as e:
            st.error(f"Gagal memuat CSV: {e}")

with tab_txt:
    st.caption("Unggah TXT dengan **satu dokumen per baris**.")
    up_txt = st.file_uploader("Upload TXT", type=["txt"], key="txt_up")
    if up_txt is not None:
        try:
            t = up_txt.read().decode("utf-8", errors="ignore")
            lines = [line.strip() for line in t.splitlines() if line.strip()]
            if lines:
                docs = lines
                doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
                st.success(f"Memuat {len(docs)} dokumen dari TXT.")
        except Exception as e:
            st.error(f"Gagal memuat TXT: {e}")

with tab_paste:
    st.caption("Tempel teks Anda di bawah ini (satu dokumen per baris).")
    txt = st.text_area("Tempel korpus", height=180, placeholder="Dokumen 1...\nDokumen 2...\nDokumen 3...")
    if txt.strip():
        docs = [line.strip() for line in txt.splitlines() if line.strip()]
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        st.success(f"Memuat {len(docs)} dokumen dari input tempel.")

loaded_index = None
with tab_load_index:
    st.caption("Muat file index JSON yang sebelumnya diunduh.")
    up_idx = st.file_uploader("Upload Index JSON", type=["json"], key="idx_up")
    if up_idx is not None:
        try:
            data = json.loads(up_idx.read().decode("utf-8"))
            if "index" in data and "num_docs" in data:
                loaded_index = data
                st.success("Index JSON dimuat.")
            else:
                st.error("Berkas bukan format index yang valid.")
        except Exception as e:
            st.error(f"Gagal memuat JSON: {e}")

# -------- Build Index --------
st.subheader("2) Bangun Inverted Index")
if not docs and not loaded_index:
    st.info("Muat korpus (CSV/TXT/Tempel) atau muat index JSON terlebih dahulu.")
    st.stop()

if loaded_index:
    index = loaded_index
    st.success(f"Index dimuat dari JSON (num_docs={index['num_docs']}).")
else:
    if st.button("🚀 Bangun Index dari Korpus"):
        index = build_inverted_index(
            docs,
            lowercase=lowercase,
            remove_digits=remove_digits,
            remove_punct=remove_punct,
            lang=lang,
            use_stop=use_stop
        )
        st.session_state["index"] = index
        st.session_state["docs"] = docs
        st.session_state["doc_ids"] = doc_ids
        st.success(f"Sukses membangun index: {len(index['index'])} terms dari {len(docs)} dokumen.")
    else:
        # try restore from session if exists
        index = st.session_state.get("index")
        docs = st.session_state.get("docs", docs)
        doc_ids = st.session_state.get("doc_ids", doc_ids)
        if not index:
            st.stop()

# Ensure we have index at this point
if not index:
    st.stop()

# -------- Explore Vocabulary --------
st.subheader("3) Eksplorasi Kosakata & Postings")
rows = vocabulary_stats(index)
st.caption(f"Vocabulary size: **{len(rows)}** terms")

k = st.slider("Tampilkan Top‑K terms (urut DF)", min_value=10, max_value=500, value=50, step=10)
show_rows = rows[:k]
st.dataframe(
    {"term": [t for t,_,__ in show_rows],
     "df": [d for _,d,__ in show_rows],
     "cf": [c for __,_,c in show_rows]},
    use_container_width=True, height=320
)

# Download vocab CSV
csvbuf = io.StringIO()
csvbuf.write("term,df,cf\n")
for t, d, c in rows:
    csvbuf.write(f"{t},{d},{c}\n")
st.download_button("⬇️ Unduh Vocabulary (CSV)", data=csvbuf.getvalue().encode("utf-8"),
                   file_name="vocabulary.csv", mime="text/csv")

# Select term to view postings
term_query = st.text_input("Lihat postings untuk term tertentu:", value="language")
if term_query:
    entry = index["index"].get(term_query)
    if entry:
        st.write(f"**{term_query}** → df={entry['df']}")
        st.json(entry["postings"])
    else:
        st.info("Term tidak ditemukan.")

# -------- Search --------
st.subheader("4) Pencarian")
st.markdown(
    """
    - **Boolean:** gunakan operator `AND`, `OR`, `NOT` (huruf besar). Contoh: `nlp AND language NOT image`  
    - **Phrase:** gunakan tanda kutip. Contoh: `"natural language processing"`
    """
)

colA, colB = st.columns([1.2, .8])
with colA:
    q = st.text_input("Masukkan query")
with colB:
    mode = st.selectbox("Mode", ["Boolean", "Phrase"], index=0)

hits: Set[int] = set()
if q.strip():
    if mode == "Boolean":
        hits = boolean_search(q, index)
    else:
        phrase = q.strip().strip('"')
        hits = phrase_search(phrase, index)

    if hits:
        st.success(f"Ditemukan {len(hits)} dokumen.")
        # show list with snippets
        for didx in sorted(list(hits)):
            original = st.session_state.get("docs", docs)[didx] if not loaded_index else "(Teks tidak tersedia — muat korpus untuk snippet)"
            title = st.session_state.get("doc_ids", doc_ids)[didx] if doc_ids else f"doc_{didx+1}"
            st.markdown(f"**{title}**")
            if original and original != "(Teks tidak tersedia — muat korpus untuk snippet)":
                terms = [t for t in re.findall(r'\w+', q.lower()) if t not in {"and","or","not"}]
                st.write(highlight_snippet(original, terms))
            st.markdown("---")
    else:
        st.warning("Tidak ada hasil.")
else:
    st.info("Masukkan query dan pilih mode pencarian.")

# -------- Save Index --------
st.subheader("5) Simpan Index")
index_json = json.dumps(index, ensure_ascii=False).encode("utf-8")
st.download_button("⬇️ Unduh Index JSON", data=index_json, file_name="inverted_index.json", mime="application/json")

st.caption("Tip: Anda dapat memuat kembali index JSON di tab 'Muat Index JSON' untuk pencarian tanpa membangun ulang.")
