"""
Test full pipeline: Load → Chunk → Embed
"""
from app.rag.document_loader import DocumentLoader
from app.rag.text_splitter import TextChunker
from app.rag.embeddings import EmbeddingsManager
from dotenv import load_dotenv

load_dotenv()

print("\n🚀 Testing Full RAG Pipeline\n")
print("="*60)

# Step 1: Load document
print("\n1️⃣ Loading document...")
with open("test_sample.txt", "w") as f:
    f.write("""
RAG System Documentation

RAG (Retrieval-Augmented Generation) is an AI technique that combines 
information retrieval with text generation. It helps reduce hallucinations 
by grounding responses in actual documents.

Key components include document loading, text chunking, embedding generation, 
vector storage, and answer generation. The system uses ChromaDB for efficient 
similarity search and OpenAI for embeddings and generation.
    """)

loader = DocumentLoader()
docs = loader.load_document("test_sample.txt")
print(f"✅ Loaded {len(docs)} document(s)")

# Step 2: Chunk text
print("\n2️⃣ Chunking text...")
chunker = TextChunker(chunk_size=200, chunk_overlap=50)
chunks = chunker.split_documents(docs)
print(f"✅ Created {len(chunks)} chunks")

# Step 3: Create embeddings
print("\n3️⃣ Creating embeddings...")
em = EmbeddingsManager(persist_directory="./data/test_pipeline")
vectorstore = em.create_vectorstore(chunks, collection_name="pipeline_test")
print(f"✅ Stored {em.get_collection_info()['count']} vectors")

# Step 4: Test search
print("\n4️⃣ Testing search...")
query = "What is RAG?"
results = em.similarity_search(query, k=2)
print(f"\n🔍 Query: '{query}'")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}: {doc.page_content[:100]}...")

print("\n" + "="*60)
print("✅ Full pipeline working!\n")