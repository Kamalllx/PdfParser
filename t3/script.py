import os
import tempfile
import json
from typing import List, Dict
from pymongo import MongoClient
from dotenv import load_dotenv
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
COLLECTION_META_FILE = "pdf_collections_groq.json"

client = Groq(api_key=GROQ_API_KEY)

def load_collections():
    if os.path.exists(COLLECTION_META_FILE):
        with open(COLLECTION_META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_collections(collections):
    with open(COLLECTION_META_FILE, "w", encoding="utf-8") as f:
        json.dump(collections, f, indent=2)

class PDFProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)

    def summarize_chunk(self, chunk: str) -> str:
        # Use Groq Llama-4 to summarize or annotate the chunk
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Summarize the following textbook chunk in one sentence or extract its main topic or heading. If it's a table or diagram, describe its main idea. Chunk:\n\n{chunk}"}
                ]
            }
        ]
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=prompt,
            max_completion_tokens=128
        )
        return response.choices[0].message.content.strip()

    def process_pdf(self, pdf_path: str) -> (List[Dict], List[str]):
        documents = []
        chunk_topics = []

        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        images = convert_from_path(pdf_path)
        print(f"\n[PDFProcessor] Loaded {len(pages)} pages from {pdf_path}")

        for i, (page, image) in enumerate(zip(pages, images)):
            print(f"\n[PDFProcessor] Processing page {i+1}...")
            img_path = f"page_images/{os.path.basename(pdf_path).replace('.pdf','')}_page_{i+1}.jpg"
            os.makedirs("page_images", exist_ok=True)
            image.save(img_path, format="JPEG")

            combined_content = page.page_content
            splits = self.text_splitter.split_text(combined_content)
            print(f"[PDFProcessor] Split into {len(splits)} chunks on page {i+1}")

            for split in splits:
                topic = self.summarize_chunk(split)
                print(f"[PDFProcessor] Chunk topic/summary: {topic}")
                chunk_topics.append(topic)
                documents.append({
                    "chunk": split,
                    "summary": topic,
                    "page": i+1,
                    "source": pdf_path,
                    "img_path": img_path
                })

        print(f"\n[PDFProcessor] Total chunks created: {len(documents)}")
        return documents, chunk_topics

class NormalDBManager:
    def __init__(self):
        self.client = MongoClient(MONGODB_URI)
        self.db = self.client["ncert_chatbot_groq"]

    def store_documents(self, documents: List[Dict], collection_name: str):
        print(f"[NormalDBManager] Storing {len(documents)} chunks in MongoDB collection '{collection_name}'...")
        collection = self.db[collection_name]
        collection.delete_many({})  # Clear old
        collection.insert_many(documents)
        print("[NormalDBManager] Storage complete.")

    def list_collections(self):
        return [name for name in self.db.list_collection_names() if not name.startswith("system.")]

    def get_topics_for_collection(self, collection_name: str):
        collection = self.db[collection_name]
        topics = collection.distinct("summary")
        return sorted(set([t for t in topics if t and isinstance(t, str)]))

    def get_chunks(self, collection_name: str):
        return list(self.db[collection_name].find({}))

class GroqRetriever:
    def __init__(self, db_manager: NormalDBManager):
        self.db_manager = db_manager

    def select_relevant_chunks(self, query: str, collection_name: str, top_k=5):
        # Retrieve all summaries
        chunks = self.db_manager.get_chunks(collection_name)
        summaries = [c['summary'] for c in chunks]
        # Use Groq to select the most relevant summaries
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You are given a user's question and a list of textbook chunk summaries. Select the {top_k} most relevant summaries for answering the question. Return only the exact summaries as a numbered list. \n\nQuestion:\n{query}\n\nSummaries:\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(summaries))}
                ]
            }
        ]
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=prompt,
            max_completion_tokens=256
        )
        answer = response.choices[0].message.content
        print(f"[GroqRetriever] Groq selected summaries:\n{answer}")
        # Parse the numbers from the returned list
        selected = []
        for line in answer.splitlines():
            if line.strip() and line[0].isdigit():
                idx = int(line.split('.', 1)[0]) - 1
                if 0 <= idx < len(chunks):
                    selected.append(chunks[idx])
        return selected

class GroqChatbot:
    def __init__(self, db_manager: NormalDBManager):
        self.db_manager = db_manager
        self.retriever = GroqRetriever(db_manager)
        self.memory = []

    def generate_response(self, query: str, collection_name: str):
        print(f"[GroqChatbot] Selecting relevant chunks using Groq for query: '{query}'")
        relevant_chunks = self.retriever.select_relevant_chunks(query, collection_name)
        if not relevant_chunks:
            print("\n[!] No relevant context found for your query in the indexed PDF.")
            return "The answer is not available in the provided material."

        print("\n--- Context Chunks Used ---")
        page_numbers = set()
        img_paths = set()
        for idx, c in enumerate(relevant_chunks):
            print(f"Chunk {idx+1}:")
            print(f"Page: {c['page']}, Source: {c['source']}")
            print(f"Summary: {c['summary']}")
            print(f"Content (first 500 chars):\n{c['chunk'][:500]}")
            print("-" * 40)
            page_numbers.add(c['page'])
            img_paths.add(c['img_path'])

        # Save the relevant pages as PDFs
        sources = set(c['source'] for c in relevant_chunks)
        os.makedirs("extracted_pages", exist_ok=True)
        for source in sources:
            try:
                reader = PdfReader(source)
                for page_num in page_numbers:
                    if 0 <= page_num - 1 < len(reader.pages):
                        writer = PdfWriter()
                        writer.add_page(reader.pages[page_num - 1])
                        base_name = os.path.splitext(os.path.basename(source))[0]
                        out_path = f"extracted_pages/{base_name}_page_{page_num}.pdf"
                        with open(out_path, "wb") as out_f:
                            writer.write(out_f)
                        print(f"Saved page {page_num} from {source} to {out_path}")
            except Exception as e:
                print(f"Error extracting pages from {source}: {e}")

        # Save page images used for context
        os.makedirs("extracted_page_images", exist_ok=True)
        for img_path in img_paths:
            if os.path.exists(img_path):
                base = os.path.basename(img_path)
                out_img = os.path.join("extracted_page_images", base)
                with open(img_path, "rb") as f_in, open(out_img, "wb") as f_out:
                    f_out.write(f_in.read())
                print(f"Saved page image: {out_img}")

        context = "\n\n".join([c['chunk'] for c in relevant_chunks])
        print("\n--- Context Provided to LLM ---\n")
        print(context[:2000])

        # Use Groq to answer using the selected context
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You are a strict NCERT textbook assistant. Answer ONLY using the provided context. If the answer is not in the context, say 'The answer is not available in the provided material.'\n\nContext:\n{context}\n\nQuestion:\n{query}"}
                ]
            }
        ]
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=prompt,
            max_completion_tokens=512
        )
        answer = response.choices[0].message.content.strip()
        self.memory.append({"user": query, "context": context, "answer": answer})
        print("[GroqChatbot] Answer built.")
        return answer

def main():
    pdf_processor = PDFProcessor()
    db_manager = NormalDBManager()
    chatbot = GroqChatbot(db_manager)
    collections_meta = load_collections()

    while True:
        print("\n1. Process PDF\n2. Ask Question\n3. Exit")
        choice = input("Enter choice: ")

        if choice == "1":
            pdf_path = input("Enter PDF path: ")
            if not os.path.exists(pdf_path):
                print("Invalid path")
                continue

            collection_name = os.path.basename(pdf_path).split('.')[0]
            print("Processing PDF, this may take a while...")
            documents, chunk_topics = pdf_processor.process_pdf(pdf_path)
            db_manager.store_documents(documents, collection_name)
            collections_meta[collection_name] = {
                "pdf_path": pdf_path,
                "topics": sorted(list(set(chunk_topics)))
            }
            save_collections(collections_meta)
            print(f"Processed {len(documents)} chunks into collection '{collection_name}'")

        elif choice == "2":
            collections_meta = load_collections()
            available_collections = list(collections_meta.keys())
            if not available_collections:
                print("No PDF collections found. Please process a PDF first.")
                continue
            print("\nAvailable PDF Collections:")
            for idx, cname in enumerate(available_collections):
                print(f"{idx+1}. {cname}")
            sel = input("Select a collection by number: ")
            try:
                sel_idx = int(sel) - 1
                if sel_idx < 0 or sel_idx >= len(available_collections):
                    raise ValueError
                collection_name = available_collections[sel_idx]
            except Exception:
                print("Invalid selection.")
                continue

            print(f"\nTopics in '{collection_name}':")
            topics = collections_meta[collection_name].get("topics", [])
            for t in topics:
                print(f"- {t}")
            print("\n[Chat mode started. Type your questions. Press Ctrl+C to exit chat mode.]")
            chatbot.memory = []
            try:
                while True:
                    query = input("\nYour question: ")
                    response = chatbot.generate_response(query, collection_name)
                    print("\nAssistant:", response)
            except KeyboardInterrupt:
                print("\n[Exited chat mode for this collection.]")

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
