import os
import base64
import tempfile
import json
from typing import List, Dict
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from pdf2image import convert_from_path
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfWriter
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

COLLECTION_META_FILE = "pdf_collections.json"

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
        self.groq_vision = ChatGroq(
            temperature=0.2,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        self.groq_topic = ChatGroq(
            temperature=0.0,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

    def extract_chunk_topic(self, text: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the following textbook chunk, extract a short heading or main topic (max 10 words) that best describes its content. If not clear, guess based on the text."),
            ("user", "{chunk}")
        ])
        chain = prompt | self.groq_topic | StrOutputParser()
        topic = chain.invoke({"chunk": text})
        return topic.strip().replace('\n', ' ')

    def process_pdf(self, pdf_path: str) -> List[Document]:
        documents = []
        chunk_topics = []

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        images = convert_from_path(pdf_path)
        print(f"\n[PDFProcessor] Loaded {len(pages)} pages from {pdf_path}")

        for i, (page, image) in enumerate(zip(pages, images)):
            print(f"\n[PDFProcessor] Processing page {i+1}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as buffered:
                image.save(buffered, format="JPEG")
                buffered.seek(0)
                base64_image = base64.b64encode(buffered.read()).decode('utf-8')

            print("[PDFProcessor] Sending image to Groq Vision for description...")
            vision_response = self.groq_vision.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": "Describe this image in detail for OCR purposes"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ])
            ])

            combined_content = f"{page.page_content}\n[Image Description: {vision_response.content}]"
            splits = self.text_splitter.split_text(combined_content)
            print(f"[PDFProcessor] Split into {len(splits)} chunks on page {i+1}")

            for split in splits:
                topic = self.extract_chunk_topic(split)
                print(f"[PDFProcessor] Chunk topic: {topic}")
                chunk_topics.append(topic)
                # Prepend topic to chunk content for better retrieval!
                combined = f"{topic}\n{split}"
                documents.append(Document(
                    page_content=combined,
                    metadata={
                        "source": pdf_path,
                        "page": i+1,
                        "images_processed": True,
                        "topic": topic
                    }
                ))


        print(f"\n[PDFProcessor] Total chunks created: {len(documents)}")
        return documents, chunk_topics

class VectorDBManager:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client["ncert_chatbot"]
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def store_documents(self, documents: List[Document], collection_name: str):
        print(f"[VectorDBManager] Storing {len(documents)} chunks in MongoDB collection '{collection_name}'...")
        collection = self.db[collection_name]
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.embeddings,
            index_name="ncert_index"
        )
        vector_store.add_documents(documents)
        print("[VectorDBManager] Storage complete.")

    def list_collections(self):
        return [name for name in self.db.list_collection_names() if not name.startswith("system.")]

    def get_topics_for_collection(self, collection_name: str):
        collection = self.db[collection_name]
        topics = collection.distinct("metadata.topic")
        return sorted(set([t for t in topics if t and isinstance(t, str)]))

class Chatbot:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        self.memory = []
        self.vector_db = VectorDBManager()

    def create_retriever(self, collection_name: str):
        print(f"[Chatbot] Creating retriever for collection '{collection_name}'...")
        collection = self.vector_db.db[collection_name]
        return MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.vector_db.embeddings,
            index_name="ncert_index"
        ).as_retriever(search_kwargs={"k": 10})

    def generate_response(self, query: str, collection_name: str):
        print(f"[Chatbot] Vectorizing user query: '{query}'")
        retriever = self.create_retriever(collection_name)
        print("[Chatbot] Retrieving relevant chunks from vector DB...")
        docs = retriever.invoke(query)

        if not docs:
            print("\n[!] No relevant context found for your query in the indexed PDF.")
            print("[!] The LLM will answer from its own knowledge, not from the PDF.")
            return "The answer is not available in the provided material."

        print("\n--- Context Chunks Used ---")
        page_numbers = set()
        sources = set()
        for idx, doc in enumerate(docs):
            page = doc.metadata.get("page")
            source = doc.metadata.get("source")
            topic = doc.metadata.get("topic", "")
            print(f"Chunk {idx+1}:")
            print(f"Page: {page}, Source: {source}")
            print(f"Topic: {topic}")
            print(f"Content (first 500 chars):\n{doc.page_content[:500]}")
            print("-" * 40)
            if page:
                page_numbers.add(page)
            if source:
                sources.add(source)

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

        context = "\n\n".join([doc.page_content for doc in docs])

        print("\n--- Vector Embeddings Used (Document Metadata) ---")
        for doc in docs:
            print(doc.metadata)

        print("\n--- Context Provided to LLM ---\n")
        print(context[:2000])

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strict NCERT textbook assistant. Answer ONLY using the provided context. If the answer is not in the context, say 'The answer is not available in the provided material.'.
Context: {context}
Conversation History: {history}"""),
            ("user", "{question}")
        ])
        print("[Chatbot] Building answer using LLM...")
        chain = (
            {"context": lambda x: context, "question": RunnablePassthrough(), "history": lambda x: self.memory}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        self.memory.extend([
            HumanMessage(content=query),
            AIMessage(content=response)
        ])
        print("[Chatbot] Answer built.")
        return response

def main():
    pdf_processor = PDFProcessor()
    db_manager = VectorDBManager()
    chatbot = Chatbot()
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
