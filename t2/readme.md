# NCERT Study Assistant

A CLI-based chatbot that helps students understand NCERT textbooks for classes 11 and 12, particularly in Physics and Chemistry. This application processes PDF files, extracts text and images, and uses Groq's language and vision models to provide intelligent responses to student questions.

## Features

- **PDF Processing**: Extract text and diagrams from NCERT PDFs
- **Vision Understanding**: Analyze and explain diagrams using Groq's vision model
- **Vector Storage**: Store and retrieve information efficiently using Pinecone vector database
- **Contextual Responses**: Maintain conversation history for contextually relevant answers
- **CLI Interface**: Simple terminal-based interface for ease of use

## Requirements

- Python 3.8+
- Groq API key
- Pinecone API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ncert-study-assistant.git
   cd ncert-study-assistant
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file with your API keys.

## Usage

### Processing PDFs

To process NCERT PDFs and add them to the vector database:

```bash
python main.py --pdf /path/to/your/pdfs
```

You can specify a single PDF file or a directory containing multiple PDFs.

### Starting the Chatbot

To start the chatbot with already processed PDFs:

```bash
python main.py
```

### Options

- `--pdf`, `-p`: Path to PDF file or directory
- `--index_name`, `-i`: Name of the Pinecone index (default: ncert-assistant)
- `--dimension`, `-d`: Dimension of vector embeddings (default: 1536)
- `--model`, `-m`: Groq model to use (default: llama3-70b-8192)
- `--vision_model`, `-v`: Groq vision model (default: llava-13b)
- `--chunk_size`, `-c`: Text chunk size (default: 1000)
- `--chunk_overlap`, `-o`: Text chunk overlap (default: 200)

## Example

```bash
# Process PDFs and start chatbot
python main.py --pdf ./ncert_books/physics_class12

# Start chatbot with existing data
python main.py

# Use custom model and index
python main.py --model llama3-8b-8192 --index_name physics-assistant
```

## How It Works

1. **PDF Processing**: The application extracts text and images from PDFs.
2. **Text Processing**: Text is split into manageable chunks for efficient storage.
3. **Image Understanding**: Images and diagrams are processed with Groq's vision model.
4. **Vector Storage**: All information is stored in Pinecone with embeddings.
5. **Query Processing**: User queries are matched with relevant information.
6. **Response Generation**: The LLM generates helpful responses using retrieved context.

## File Structure

- `main.py`: Entry point and CLI interface
- `pdf_processor.py`: Handles PDF extraction and processing
- `vector_store.py`: Manages Pinecone vector database
- `chatbot.py`: Core chatbot logic using Groq
- `conversation_manager.py`: Manages conversation history

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NCERT for their excellent educational materials
- Groq for their powerful LLM and vision models
- Pinecone for vector database capabilities
- LangChain for simplifying LLM application development