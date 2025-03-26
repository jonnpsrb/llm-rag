import sqlite3
import shutil
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

base_dir = Path(__file__).resolve().parent
sql_file_path = base_dir.parent / "data/ibutler_sqlite.sql"
sqlite_file_path = base_dir.parent / "data/ibutler_sqlite.db"
pdf_file_path = base_dir.parent / "data/ibutler.pdf"
chroma_dir_path = base_dir.parent / "data/ibutler_chroma"

def remove_dbs(sqlite_file: Path, chroma_path: Path):
    if sqlite_file.exists():
        sqlite_file.unlink()
        print(f"🗑️ Removed existing SQLite DB: {sqlite_file}")
    if chroma_path.exists() and chroma_path.is_dir():
        shutil.rmtree(chroma_path)
        print(f"🗑️ Removed existing Chroma DB: {chroma_path}")

def initialize_sqlite_from_sqlfile(sql_file: Path, db_file: Path):
    with open(sql_file, "r") as f:
        sql_script = f.read()
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.executescript(sql_script)
    conn.commit()
    conn.close()
    print(f"🗄 Created SQLite DB at {db_file}")


def load_document_from_pdf(pdf_file_path: Path):
    document_loader = PyPDFLoader(pdf_file_path)
    doc = document_loader.load()
    print(f"📄 Loaded {len(doc)} pages")
    return doc

def initialize_chromadb_from_pdf(pdf_file_path: Path, chroma_dir_path: Path):
    doc = load_document_from_pdf(pdf_file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(doc)
    print(f"📄 Split into {len(chunks)} chunks")
    embed_with_chroma(chunks, chroma_dir_path)
    return chunks

def calculate_chunk_ids(chunks):
    # This will create IDs like "data/ibutler.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def get_embedding_function():
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def embed_with_chroma(chunks: list[Document], persist_dir: Path):
    db = Chroma(
        persist_directory=str(persist_dir), embedding_function=get_embedding_function()
    )
    chunks_with_ids = calculate_chunk_ids(chunks)
    chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    db.add_documents(chunks_with_ids, ids=chunk_ids)
    print(f"👉 Documents added: {len(chunks_with_ids)}")

if __name__ == "__main__":
    remove_dbs(sqlite_file_path, chroma_dir_path)
    initialize_sqlite_from_sqlfile(sql_file_path, sqlite_file_path)
    initialize_chromadb_from_pdf(pdf_file_path, chroma_dir_path)

def generate():
    remove_dbs(sqlite_file_path, chroma_dir_path)
    initialize_sqlite_from_sqlfile(sql_file_path, sqlite_file_path)
    initialize_chromadb_from_pdf(pdf_file_path, chroma_dir_path)
