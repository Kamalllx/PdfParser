import os
import re
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pymongo import MongoClient

def extract_text_and_images(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        image_descriptions = []
        for page in doc:
            text += page.get_text()
            for img in page.get_images(full=True):
                image_descriptions.append("Image present (description not available)")
        doc.close()
        return text, image_descriptions
    except Exception as e:
        raise Exception(f"Error parsing PDF: {e}")

def process_pdf(pdf_path):
    text, image_descriptions = extract_text_and_images(pdf_path)
    combined_text = text + "\n\nImage Descriptions:\n" + "\n".join(image_descriptions)
    return combined_text

def setup_vector_store(text, collection_name):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = MongoClient("your_mongodb_uri_here")  # Replace with your MongoDB URI
    db = client["pdf_chatbot_db"]
    collection = db[collection_name]
    vector_store = MongoDBAtlasVectorSearch.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection=collection,
        index_name="vector_index",
    )
    return vector_store

def setup_chain(vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="mixtral-8x7b-32768"
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    return chain

def main():
    print("Welcome to the PDF Chatbot")
    pdf_path = input("Please enter the path to the PDF file: ")
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found.")
        return
    
    try:
        print("Processing PDF...")
        combined_text = process_pdf(pdf_path)
        pdf_name = re.sub(r'\W+', '_', os.path.basename(pdf_path).split(".")[0])
        collection_name = f"pdf_{pdf_name}"
        vector_store = setup_vector_store(combined_text, collection_name)
        chain = setup_chain(vector_store)
        print("PDF processed successfully. You can now ask questions about the content.")
        print("Type 'exit' or 'quit' to stop.")
        
        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye.")
                break
            try:
                response = chain({"question": user_input})
                print("Bot:", response["answer"])
            except Exception as e:
                print(f"Error generating response: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()