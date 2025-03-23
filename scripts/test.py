import argparse
import os
import time
import shutil
import sqlite3
from pathlib import Path
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Import functions from generate.py
from generate import (
    get_embedding_function,
    base_dir,
    sql_file_path,
    sqlite_file_path,
    pdf_file_path,
    chroma_dir_path,
    remove_dbs,
    initialize_sqlite_from_sqlfile,
    load_document_from_pdf,
    calculate_chunk_ids,
    generate
)

# Prompt template with language instruction
PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
---
Answer the question based on the above context: {question}

IMPORTANT: Please provide your answer in the same language as the question. 
If the question is in Indonesian, answer in Indonesian. 
If the question is in English, answer in English.
"""

def custom_initialize_chromadb(pdf_file_path, chroma_dir_path, chunk_size=1000, chunk_overlap=100):
    """Custom implementation to allow parameterization of chunk size and overlap"""
    doc = load_document_from_pdf(pdf_file_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(doc)
    print(f"📄 Split into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap})")
    
    # Get embedding function
    embedding_function = get_embedding_function()
    
    # Initialize Chroma
    db = Chroma(
        persist_directory=str(chroma_dir_path), embedding_function=embedding_function
    )
    
    # Add documents
    chunks_with_ids = calculate_chunk_ids(chunks)
    chunk_ids = [chunk.metadata["id"] for chunk in chunks_with_ids]
    db.add_documents(chunks_with_ids, ids=chunk_ids)
    print(f"👉 Documents added: {len(chunks_with_ids)}")
    
    return chunks

def test_retrieval(query, chroma_path=chroma_dir_path, top_k=5):
    """Test retrieval quality for a specific query"""
    print(f"\n===== Testing query: '{query}' =====")
    
    # Get embedding function
    embedding_function = get_embedding_function()
    
    # Initialize Chroma with the embedding function
    db = Chroma(persist_directory=str(chroma_path), embedding_function=embedding_function)
    
    # Perform search
    start_time = time.time()
    results = db.similarity_search_with_score(query, k=top_k)
    search_time = time.time() - start_time
    
    # Print results
    print(f"Found {len(results)} results in {search_time:.4f} seconds")
    for i, (doc, score) in enumerate(results):
        print(f"\nResult #{i+1} [Score: {score:.4f}]")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
        print("-" * 40)
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
        print(content)
    
    return results

def test_rag_pipeline(query, chroma_path=chroma_dir_path, top_k=5, groq_api_key=None):
    """Test the full RAG pipeline including LLM response"""
    # Get retrieval results
    results = test_retrieval(query, chroma_path, top_k)
    
    if not groq_api_key:
        print("\nSkipping LLM generation (no API key provided)")
        return results
    
    # Prepare context from results
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    
    # Create prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    
    # Call LLM
    print("\n===== Generating LLM Response =====")
    try:
        model = ChatGroq(
            model_name="llama3-8b-8192",
            groq_api_key=groq_api_key
        )
        
        start_time = time.time()
        response = model.invoke(prompt)
        generation_time = time.time() - start_time
        
        print(f"Response generated in {generation_time:.4f} seconds:")
        print("-" * 40)
        print(response.content)
        print("-" * 40)
        
        return results, response.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return results, None

def create_test_config(config_name, chunk_size=1000, chunk_overlap=100):
    """Create a test configuration with specified parameters"""
    config_path = base_dir.parent / f"data/test_{config_name}"
    
    # Initialize database
    remove_dbs(sqlite_file_path, config_path)
    initialize_sqlite_from_sqlfile(sql_file_path, sqlite_file_path)
    
    # Create embeddings using our custom function
    start_time = time.time()
    custom_initialize_chromadb(pdf_file_path, config_path, chunk_size, chunk_overlap)
    indexing_time = time.time() - start_time
    
    return config_path, indexing_time

def compare_chunk_sizes(query, sizes=[500, 1000, 1500], overlap=100):
    """Compare different chunk sizes for the same query"""
    print(f"\n===== Comparing chunk sizes for query: '{query}' =====")
    
    results = {}
    for size in sizes:
        # Create a directory for this configuration
        config_name = f"size_{size}_overlap_{overlap}"
        print(f"\n----- Testing chunk size: {size} -----")
        
        # Create test config
        config_path, indexing_time = create_test_config(config_name, size, overlap)
        
        # Test retrieval
        retrieval_results = test_retrieval(query, config_path)
        
        # Store results
        avg_score = sum(score for _, score in retrieval_results) / len(retrieval_results) if retrieval_results else float('inf')
        results[size] = {
            "indexing_time": indexing_time,
            "avg_score": avg_score,
            "num_results": len(retrieval_results)
        }
    
    # Print comparison
    print("\n===== Chunk Size Comparison =====")
    print(f"{'Size':<8} | {'Indexing Time':<15} | {'Avg Score':<12} | {'# Results':<10}")
    print("-" * 50)
    for size, data in sorted(results.items()):
        print(f"{size:<8} | {data['indexing_time']:.2f}s {'':8} | {data['avg_score']:.4f} {'':4} | {data['num_results']:<10}")
    
    # Identify best size
    best_size = min(results.items(), key=lambda x: x[1]['avg_score'])[0]
    print(f"\nBest chunk size: {best_size} (lowest average score)")
    
    return results

def compare_overlaps(query, chunk_size=1000, overlaps=[50, 100, 150, 200]):
    """Compare different overlap sizes for the same query"""
    print(f"\n===== Comparing overlap sizes (chunk size: {chunk_size}) for query: '{query}' =====")
    
    results = {}
    for overlap in overlaps:
        # Create a directory for this configuration
        config_name = f"size_{chunk_size}_overlap_{overlap}"
        print(f"\n----- Testing overlap: {overlap} -----")
        
        # Create test config
        config_path, indexing_time = create_test_config(config_name, chunk_size, overlap)
        
        # Test retrieval
        retrieval_results = test_retrieval(query, config_path)
        
        # Store results
        avg_score = sum(score for _, score in retrieval_results) / len(retrieval_results) if retrieval_results else float('inf')
        results[overlap] = {
            "indexing_time": indexing_time,
            "avg_score": avg_score,
            "num_results": len(retrieval_results)
        }
    
    # Print comparison
    print("\n===== Chunk Overlap Comparison =====")
    print(f"{'Overlap':<8} | {'Indexing Time':<15} | {'Avg Score':<12} | {'# Results':<10}")
    print("-" * 50)
    for overlap, data in sorted(results.items()):
        print(f"{overlap:<8} | {data['indexing_time']:.2f}s {'':8} | {data['avg_score']:.4f} {'':4} | {data['num_results']:<10}")
    
    # Identify best overlap
    best_overlap = min(results.items(), key=lambda x: x[1]['avg_score'])[0]
    print(f"\nBest overlap: {best_overlap} (lowest average score)")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG system with different configurations")
    parser.add_argument("--query", type=str, default="Apa itu iButler?", help="Query to test")
    parser.add_argument("--compare-chunks", action="store_true", help="Compare different chunk sizes")
    parser.add_argument("--compare-overlaps", action="store_true", help="Compare different overlap values")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the database before testing")
    parser.add_argument("--groq-key", type=str, help="Groq API key for LLM testing")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for testing")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap for testing")
    
    args = parser.parse_args()
    
    # Rebuild database if requested
    if args.rebuild:
        generate()
    
    # Run comparisons if requested
    if args.compare_chunks:
        compare_chunk_sizes(args.query, sizes=[500, 800, 1000, 1500], overlap=args.chunk_overlap)
    elif args.compare_overlaps:
        compare_overlaps(args.query, chunk_size=args.chunk_size, overlaps=[50, 100, 150, 200])
    else:
        # Run regular test
        if args.groq_key:
            test_rag_pipeline(
                args.query, 
                chroma_dir_path, 
                top_k=5, 
                groq_api_key=args.groq_key
            )
        else:
            test_retrieval(
                args.query, 
                chroma_dir_path, 
                top_k=5
            )