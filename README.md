## Usage

### 1. Create venv and install dependencies

- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

- `npm install`

### 2. Generate SQLITE and ChromaDB
Based on `data/ibutler_sqlite.sql` sqlite3 database will be generated and based on the PDF `data/ibutler.pdf` chromadb will be generated

1. Run generate script `python scripts/generate.py`
   * This will remove sqlite3 and chromadb under `/data` folder
   * Then, generate `ibutler_sqlite.db` and `ibutler_chroma` folder

### 3. Run local server

- `npm run dev`