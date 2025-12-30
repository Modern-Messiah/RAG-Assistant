from app.rag.document_loader import DocumentLoader

print("\n🔄 Testing Document Loader...\n")

loader = DocumentLoader()

# Test with TXT first (no PDF needed)
try:
    # Create a test TXT file
    with open("test_sample.txt", "w") as f:
        f.write("""
This is a sample document about RAG systems.

RAG (Retrieval-Augmented Generation) combines:
1. Document retrieval
2. Language generation

It helps reduce AI hallucinations by grounding 
responses in actual documents.
        """)
    
    # Load it
    docs = loader.load_document("test_sample.txt")
    
    print(f"✅ Loaded document successfully!")
    print(f"   Source: {docs[0].metadata['source']}")
    print(f"   Type: {docs[0].metadata['type']}")
    print(f"   Characters: {len(docs[0].page_content)}")
    print(f"\n📄 Content preview:")
    print(docs[0].page_content[:200] + "...\n")
    
except Exception as e:
    print(f"❌ Error: {e}\n")

print("✅ Test complete!\n")
