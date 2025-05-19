Below is the complete implementation of `RAG_CLI.py`, a command-line interface (CLI) version of the original `RAG.py` Streamlit app, incorporating all the requested changes. This script processes documents, stores data in MongoDB, generates a table of contents, and handles topic-based retrieval with page references and image saving.

### Overview of Changes
1. **CLI Conversion**: Replaced Streamlit with CLI using `argparse` for file input and an interactive query loop.
2. **Progress Tracking**: Added `print` statements to track key steps like file processing, text extraction, and answer generation.
3. **Page Number Tagging**: Tagged each chunk with its page number in metadata during vectorization.
4. **LLM Page Analysis**: Used Groq's LLM to analyze each page's topic.
5. **Table of Contents**: Generated and stored a table of contents with topics and page ranges in MongoDB.
6. **MongoDB Storage**: Stored chunks (text, embeddings, metadata) and table of contents in MongoDB collections.
7. **Topic-based Retrieval**: Queries first match topics, then retrieve relevant pages.
8. **Page Number Retrieval**: Retrieved page numbers for the matched topic.
9. **Context Provision**: Devectorized (extracted text) chunks from relevant pages, printed as context, and passed to the LLM.
10. **Answer with References**: Instructed the LLM to include page references in answers, listed at the end.
11. **Image Processing**: Saved screenshots of relevant PDF pages in a `context` folder, organized by query.
12. **Vision LLM Note**: Prepared to use "meta-llama/llama-4-scout-17b-16e-instruct" if needed, but focused on text for this version.

### Prerequisites
- Install required packages: `pip install PyPDF2 langchain langchain-community pymongo pdf2image langchain-groq python-dotenv numpy`
- Install `poppler` for `pdf2image` (e.g., `brew install poppler` on macOS or equivalent).
- Set the `GROQ_API_KEY` environment variable (e.g., in a `.env` file).
- Ensure MongoDB is running locally on `mongodb://localhost:27017/`.

### Full Code

```python
import os
import sys
import argparse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient
from pdf2image import convert_from_path
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import numpy as np
import datetime
import glob

load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="RAG CLI App")
    parser.add_argument("files", nargs="+", help="Files or directories to process")
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        print("Error: Groq API key not found. Please set GROQ_API_KEY environment variable.")
        sys.exit(1)

    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["rag_db"]
        chunks_collection = db["chunks"]
        toc_collection = db["toc"]
        print("Connected to MongoDB successfully.")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        sys.exit(1)

    # Create context folder if it doesn't exist
    os.makedirs("context", exist_ok=True)

    process_files(args.files, chunks_collection, toc_collection)
    query_loop(chunks_collection, toc_collection)

def process_files(paths, chunks_collection, toc_collection):
    files = []
    for path in paths:
        if os.path.isdir(path):
            files.extend(glob.glob(os.path.join(path, "*.pdf")))
            files.extend(glob.glob(os.path.join(path, "*.txt")))
            files.extend(glob.glob(os.path.join(path, "*.md")))
        elif os.path.isfile(path) and path.lower().endswith(('.pdf', '.txt', '.md')):
            files.append(path)
        else:
            print(f"Ignoring unsupported file: {path}")

    if not files:
        print("No supported files found.")
        return

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Initialized embeddings model.")

    for file in files:
        print(f"Processing {file}...")
        if file.lower().endswith('.pdf'):
            process_pdf(file, embeddings, chunks_collection, toc_collection)
        else:
            process_text_file(file, embeddings, chunks_collection, toc_collection)

def process_pdf(file, embeddings, chunks_collection, toc_collection):
    try:
        pdf_reader = PdfReader(file)
        pages = []
        for page_num in range(len(pdf_reader.pages)):
            page_text = pdf_reader.pages[page_num].extract_text() or ""
            pages.append((page_num + 1, page_text))
            print(f"Extracted text from page {page_num + 1} of {file}")
    except Exception as e:
        print(f"Error extracting text from PDF {file}: {e}")
        return

    topics = []
    for page_num, page_text in pages:
        topic = generate_topic(page_text)
        topics.append((page_num, topic))
        print(f"Generated topic for page {page_num}: {topic}")

    grouped_topics = group_similar_topics(topics, embeddings)

    toc_entries = []
    for topic, page_nums in grouped_topics.items():
        topic_embedding = embeddings.embed_query(topic)
        toc_entry = {
            "file": file,
            "topic": topic,
            "pages": page_nums,
            "embedding": topic_embedding
        }
        toc_entries.append(toc_entry)
    toc_collection.insert_many(toc_entries)
    print(f"Stored table of contents for {file} with {len(toc_entries)} topics.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    for page_num, page_text in pages:
        for entry in toc_entries:
            if page_num in entry["pages"]:
                topic = entry["topic"]
                break
        else:
            topic = "Unknown"

        chunks = text_splitter.split_text(page_text)
        for chunk_id, chunk_text in enumerate(chunks):
            embedding = embeddings.embed_query(chunk_text)
            chunk_doc = {
                "file": file,
                "page": page_num,
                "chunk_id": chunk_id,
                "text": chunk_text,
                "embedding": embedding,
                "topic": topic
            }
            chunks_collection.insert_one(chunk_doc)
            print(f"Stored chunk {chunk_id} for page {page_num} of {file}")
    print(f"Processed {len(pages)} pages from {file}")

def process_text_file(file, embeddings, chunks_collection, toc_collection):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"Extracted text from {file}")
    except Exception as e:
        print(f"Error reading file {file}: {e}")
        return

    topic = generate_topic(text)
    print(f"Generated topic for {file}: {topic}")

    topic_embedding = embeddings.embed_query(topic)
    toc_entry = {
        "file": file,
        "topic": topic,
        "pages": [1],
        "embedding": topic_embedding
    }
    toc_collection.insert_one(toc_entry)
    print(f"Stored table of contents for {file}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    for chunk_id, chunk_text in enumerate(chunks):
        embedding = embeddings.embed_query(chunk_text)
        chunk_doc = {
            "file": file,
            "page": 1,
            "chunk_id": chunk_id,
            "text": chunk_text,
            "embedding": embedding,
            "topic": topic
        }
        chunks_collection.insert_one(chunk_doc)
        print(f"Stored chunk {chunk_id} for {file}")
    print(f"Processed {file}")

def generate_topic(text):
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = f"Summarize the main topic of this text in one sentence:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content.strip()

def group_similar_topics(topics, embeddings, threshold=0.8):
    grouped = {}
    current_topic = None
    current_pages = []
    for page_num, topic in topics:
        if current_topic is None:
            current_topic = topic
            current_pages = [page_num]
        else:
            similarity = compute_similarity(current_topic, topic, embeddings)
            if similarity > threshold:
                current_pages.append(page_num)
            else:
                grouped[current_topic] = current_pages
                current_topic = topic
                current_pages = [page_num]
    if current_pages:
        grouped[current_topic] = current_pages
    return grouped

def compute_similarity(topic1, topic2, embeddings):
    emb1 = embeddings.embed_query(topic1)
    emb2 = embeddings.embed_query(topic2)
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return cos_sim

def query_loop(chunks_collection, toc_collection):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    while True:
        query = input("Enter your question (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        print(f"Processing query: {query}")
        query_embedding = embeddings.embed_query(query)
        print("Embedded the query.")

        topics = list(toc_collection.find())
        if not topics:
            print("No topics found. Please process documents first.")
            continue

        similarities = []
        for topic in topics:
            topic_emb = topic["embedding"]
            sim = np.dot(query_embedding, topic_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(topic_emb))
            similarities.append((sim, topic))

        if not similarities:
            print("No topics available.")
            continue

        similarities.sort(reverse=True, key=lambda x: x[0])
        top_topic = similarities[0][1]
        file = top_topic["file"]
        pages = top_topic["pages"]
        print(f"Selected topic '{top_topic['topic']}' from {file}, pages: {pages}")

        chunks = list(chunks_collection.find({"file": file, "page": {"$in": pages}}))
        if not chunks:
            print("No chunks found for the selected topic.")
            continue

        chunks.sort(key=lambda x: (x["page"], x["chunk_id"]))

        context = ""
        current_page = None
        for chunk in chunks:
            if chunk["page"] != current_page:
                context += f"\n[Page {chunk['page']}]\n"
                current_page = chunk["page"]
            context += chunk["text"] + " "

        print("Retrieved context:")
        print(context)

        answer = generate_answer(query, context)
        print("Answer:")
        print(answer)

        if file.lower().endswith('.pdf'):
            save_page_images(file, pages, query)

def generate_answer(query, context):
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0.2,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )
    prompt = f"Based on the following context, answer the question: {query}\n\nContext:\n{context}\n\nPlease include references to the page numbers where the information is found, like [Page X]."
    response = llm.invoke(prompt)
    print("Generated answer using LLM.")
    return response.content.strip()

def save_page_images(file, pages, query):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"context/{timestamp}_{query.replace(' ', '_')}"
    os.makedirs(folder_name, exist_ok=True)

    for page_num in pages:
        try:
            images = convert_from_path(file, first_page=page_num, last_page=page_num)
            if images:
                images[0].save(os.path.join(folder_name, f"page_{page_num}.png"))
                print(f"Saved screenshot for page {page_num}")
        except Exception as e:
            print(f"Error saving page {page_num}: {e}")
    print(f"Saved page images in {folder_name}")

if __name__ == "__main__":
    main()
```

### How to Use
1. **Run the Script**: `python RAG_CLI.py path/to/file.pdf path/to/dir/`
   - Provide file paths or directories containing `.pdf`, `.txt`, or `.md` files.
2. **Processing**: The script processes each file, extracts text, generates topics, and stores data in MongoDB.
3. **Querying**: After processing, enter questions at the prompt. Type `exit` to quit.
   - The script retrieves relevant pages, prints the context, generates an answer with references, and saves PDF page screenshots.

### Notes
- **Image Processing**: Screenshots are saved only for PDFs. For TXT/MD files, this step is skipped as they lack page structure.
- **Vision LLM**: The "meta-llama/llama-4-scout-17b-16e-instruct" model is not used here, as the focus is on text-based retrieval. It can be integrated later for image content analysis if needed.
- **Performance**: For large documents, similarity computation might be slow. Consider indexing or optimizing MongoDB queries for production use.

This implementation fulfills all requirements while maintaining the core functionality of the original `RAG.py`. Let me know if you need further adjustments!
