import os
import base64
import tempfile
from typing import List, Dict
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()

class PDFProcessor:
    def __init__(self):
        self.groq_vision = ChatGroq(
            temperature=0.2,
            model="llava-1.5-7b-hf"
        )

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text and images from PDF, process with Groq vision"""
        documents = []

        # Extract text
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        # Extract images and convert to text
        images = convert_from_path(pdf_path)
        for i, (page, image) in enumerate(zip(pages, images)):
            # Convert image to base64
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as buffered:
                image.save(buffered, format="JPEG")
                buffered.seek(0)
                base64_image = base64.b64encode(buffered.read()).decode('utf-8')

            # Get image description from Groq
            vision_response = self.groq_vision.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": "Describe this image in detail for OCR purposes"},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ])
            ])

            # Combine text and image description
            combined_content = f"{page.page_content}\n[Image Description: {vision_response.content}]"
            documents.append({
                "page_content": combined_content,
                "metadata": {
                    "source": pdf_path,
                    "page": i+1,
                    "images_processed": True
                }
            })

        return documents

class VectorDBManager:
    def __init__(self):
        self.client = MongoClient(os.getenv("MONGODB_URI"))
        self.db = self.client["ncert_chatbot"]
        # Use Hugging Face all-MiniLM-L6-v2 embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def store_documents(self, documents: List[Dict], collection_name: str):
        """Store documents in MongoDB with vector embeddings"""
        collection = self.db[collection_name]
        vector_store = MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.embeddings,
            index_name="ncert_index"
        )
        vector_store.add_documents(documents)

class Chatbot:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7,
            model="meta-llama/llama-4-scout-17b-16e-instruct"
        )
        self.memory = []
        self.vector_db = VectorDBManager()

    def create_retriever(self, collection_name: str):
        """Create vector store retriever"""
        collection = self.vector_db.db[collection_name]
        return MongoDBAtlasVectorSearch(
            collection=collection,
            embedding=self.vector_db.embeddings,
            index_name="ncert_index"
        ).as_retriever(search_kwargs={"k": 3})

    def generate_response(self, query: str, collection_name: str):
        """Generate response using RAG and conversation history"""
        retriever = self.create_retriever(collection_name)

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful NCERT textbook assistant. Use the context to answer questions.
            Context: {context}
            Conversation History: {history}"""),
            ("user", "{question}")
        ])

        chain = (
            {"context": retriever, "question": RunnablePassthrough(), "history": lambda x: self.memory}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        self.memory.extend([
            HumanMessage(content=query),
            AIMessage(content=response)
        ])
        return response

def main():
    pdf_processor = PDFProcessor()
    db_manager = VectorDBManager()
    chatbot = Chatbot()

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
            documents = pdf_processor.process_pdf(pdf_path)
            db_manager.store_documents(documents, collection_name)
            print(f"Processed {len(documents)} pages into collection '{collection_name}'")

        elif choice == "2":
            collection_name = input("Enter PDF collection name: ")
            query = input("Your question: ")
            response = chatbot.generate_response(query, collection_name)
            print("\nAssistant:", response)

        elif choice == "3":
            print("Exiting...")
            break

        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
