## Vercel Deployments
[Vercel Link](https://llm-rag.vercel.app/)

[!NOTE]  
Because of the free tier restrictions VERCEL deployed version could be outdated, or not working at all. Please try in your local environment, for better results.

## Usage

### 0. Export GROQ_API_KEY
- export GROQ_API_KEY=YOUR_API_KEY_HERE

### 1. Create venv and install dependencies

- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`

- `npm install`

### 2. Generate SQLITE and ChromaDB (Optional, generated files already committed)
Based on `data/ibutler_sqlite.sql` sqlite3 database will be generated and based on the PDF `data/ibutler.pdf` chromadb will be generated

1. Run generate script `python scripts/generate.py`
   * This will remove sqlite3 and chromadb under `/data` folder
   * Then, generate `ibutler_sqlite.db` and `ibutler_chroma` folder

### 3. Run local server

- `npm run dev`
- Local server available at http://localhost:3000


### 4. Changing embedding/chunking etc.

1. Run generate script `python scripts/generate.py`
2. Run python backend and js frontend again `npm run dev`
3. Test @ http://localhost:3000