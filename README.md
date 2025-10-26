# Inverted Index App (Streamlit)

Aplikasi web untuk membangun **Inverted Index** dari korpus/dataset dan melakukan **Boolean** serta **Phrase search**.

## Fitur
- Input data: **CSV** (pilih kolom teks), **TXT** (satu dokumen per baris), atau **tempel teks**
- Preprocess: lowercase, hapus angka/tanda baca, stopwords (EN/ID bawaan)
- Inverted index dengan **positional postings**
- Eksplorasi kosakata (DF & CF) + lihat postings list per term
- **Boolean search** (`AND`, `OR`, `NOT`) dan **Phrase search** (pakai tanda kutip)
- Simpan/Muat index sebagai **JSON** (pencarian tanpa rebuild)

## Instalasi
Disarankan memakai virtual environment.

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan
```bash
streamlit run app.py
```

Buka URL yang tampil (biasanya `http://localhost:8501`).

## Cara Pakai Singkat
1. Muat korpus dari tab **CSV / TXT / Tempel Teks**.  
2. Atur **pra-proses** di sidebar.  
3. Klik **Bangun Index** (atau muat JSON index di tab khusus).  
4. Eksplor kosakata dan postings.  
5. Coba **Boolean** (`nlp AND language NOT image`) atau **Phrase** (`"natural language processing"`).  
6. Unduh **Vocabulary CSV** & **Index JSON** bila perlu.

## Catatan
- Stopwords yang digunakan hanyalah daftar kecil built-in untuk **EN** dan **ID** (tanpa NLTK). Anda bisa menambah manual di kode bila diperlukan.
