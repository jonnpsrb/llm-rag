from flask import Flask, request, jsonify
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
app = Flask(__name__)

@app.route("/api/chat", methods=["POST"])
def hello_world():
    try:
        db = SQLDatabase.from_uri(f"sqlite:///./data/ibutler_sqlite.db", sample_rows_in_table_info=0 )

        data = request.get_json()
        query = data.get("query")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        # Load the chat model
        model = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama3-8b-8192"
        )

        # Load embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        try:
            vector_store = Chroma(
                persist_directory="./data/ibutler_chroma",
                embedding_function=embeddings
            )
        except Exception as e:
            print(f"Error loading Chroma: {e}")
            raise

        print("Current working directory:", os.getcwd())

        # Setup prompt template
        prompt = PromptTemplate(
            input_variables=["context", "context_doc", "question"],
            template="""
You are an AI assistant for iButler product. 
iButler is a personal humanoid robot can help on various chores like cooking, cleaning, laundry.
Use the provided context to answer the user’s question. If something is not directly answerable from the context, state that you do not have enough information.

Context from SQL: If SQL query and result does not make sense for the user question do not use it!
{context_sql}

Context from Document:
{context}

Question:
{question}

Reply by using Indonesian language and do not give context about SQL query or Document, just state that 'Berdasarkan pengetahuan saya'! 
"""
        )


        qa_chain = load_qa_chain(
            llm=model,
            chain_type="stuff",
            prompt=prompt,
        )

        db_chain = SQLDatabaseChain.from_llm(model, db, verbose=True, return_direct=True, return_intermediate_steps=True)
        context_doc = vector_store.as_retriever().get_relevant_documents(query)
        context_sql = None
        try:
            context_sql = db_chain.invoke({"query": query})
            print(context_sql)
        except Exception as e:
            print(f"Error in db_chain: {str(e)}")

        result = qa_chain.invoke({
            "input_documents": context_doc,
            "question": query,
            "context_sql": str(context_sql)[:512],
        }, return_only_outputs=True)
        return jsonify({"response": result["output_text"]}), 200

    except Exception as e:
        print("Error processing chat request:", e, flush=True)
        return jsonify({"error": "Internal Server Error"}), 500