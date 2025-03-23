import os
import unittest

from generate import generate, get_embedding_function

# Add the scripts directory to the path
scripts_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(scripts_dir)

from langchain_chroma import Chroma
from langchain.schema.document import Document


class TestRAGSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Directory structure
        cls.scripts_dir = scripts_dir
        cls.parent_dir = parent_dir
        cls.chroma_path = os.path.join(parent_dir, "data/ibutler_chroma")
        print(f"Data directory: {cls.chroma_path}")
        generate()

    def test_1_chroma_content(self):
        """Test that Chroma database has content."""
        print("\n=== Testing Chroma database content ===")
        try:
            embedding_function = get_embedding_function()
            db = Chroma(persist_directory=self.chroma_path,
                       embedding_function=embedding_function)
            
            # Get count of documents
            count = db._collection.count()
            self.assertTrue(count > 0, "No documents found in Chroma database")
            print(f"✅ Chroma database contains {count} documents")
            
            # Get some documents
            docs = db.get(limit=3)
            print(f"✅ Sample document metadata: {docs['metadatas'][0]}")
        except Exception as e:
            self.fail(f"Chroma content test failed with error: {str(e)}")
    
    def test_2_direct_search(self):
        """Test direct search using the embedding function and Chroma."""
        print("\n=== Testing direct vector search ===")

        try:
            embedding_function = get_embedding_function()
            db = Chroma(persist_directory=self.chroma_path,
                       embedding_function=embedding_function)
            
            # Perform a similarity search directly
            test_query = "What is iButler?"
            results = db.similarity_search_with_score(test_query, k=3)
            
            self.assertTrue(len(results) > 0, "Similarity search returned no results")
            self.assertIsInstance(results[0][0], Document, "Result is not a Document object")
            
            print(f"✅ Similarity search returned {len(results)} documents")
            print(f"✅ First result content preview: {results[0][0].page_content[:100]}...")
            print(f"✅ Score: {results[0][1]}")
        except Exception as e:
            self.fail(f"Direct vector search test failed with error: {str(e)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)