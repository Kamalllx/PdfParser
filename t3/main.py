import os
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
from script import PDFProcessor, NormalDBManager, GroqChatbot, load_collections, save_collections

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = '.'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('page_images', exist_ok=True)
os.makedirs('extracted_page_images', exist_ok=True)
os.makedirs('extracted_pages', exist_ok=True)

app = Flask(__name__, static_folder=STATIC_FOLDER)
CORS(app)

pdf_processor = PDFProcessor()
db_manager = NormalDBManager()
chatbot = GroqChatbot(db_manager)

collections_meta = load_collections()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

@app.route('/upload', methods=['POST'])
def upload_pdf():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        filename = secure_filename(file.filename)
        pdf_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(pdf_path)
        collection_name = os.path.splitext(filename)[0]
        documents, chunk_topics = pdf_processor.process_pdf(pdf_path)
        db_manager.store_documents(documents, collection_name)
        collections_meta[collection_name] = {
            "pdf_path": pdf_path,
            "topics": sorted(list(set(chunk_topics)))
        }
        save_collections(collections_meta)
        return jsonify({'collection': collection_name, 'topics': chunk_topics})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/collections', methods=['GET'])
def get_collections():
    collections_meta = load_collections()
    return jsonify(list(collections_meta.keys()))

@app.route('/collection/<name>/topics', methods=['GET'])
def get_topics(name):
    collections_meta = load_collections()
    topics = collections_meta.get(name, {}).get('topics', [])
    return jsonify(topics)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        collection = data['collection']
        question = data['question']
        answer = chatbot.generate_response(question, collection)
        last = chatbot.memory[-1] if chatbot.memory else {}
        context = last.get('context', '')
        relevant_chunks = db_manager.get_chunks(collection)
        used_pages = set()
        used_images = set()
        for chunk in relevant_chunks:
            if chunk['chunk'] in context:
                used_pages.add(chunk['page'])
                used_images.add(chunk['img_path'])
        image_urls = [f"/static/{img_path.replace(os.sep, '/')}" for img_path in used_images]
        return jsonify({
            'answer': answer,
            'context': context,
            'images': image_urls,
            'pages': sorted(list(used_pages))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
