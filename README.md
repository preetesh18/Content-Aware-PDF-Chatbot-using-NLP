# Content-Aware-PDF-Chatbot-using-NLP
# 📑 AI Chatbot for Content-Aware Q&A using User Files
This project implements an intelligent, file-aware Q&A chatbot built with **Streamlit**, **LangChain**, **Ollama**, and **ChromaDB**. The chatbot can process PDF, Word, or Excel files uploaded by the user, extract and vectorize the content, and answer queries based solely on the uploaded data using a local LLM.

---

## 🧠 Key Features

- **Upload and Parse Documents**: Supports PDF, DOCX, DOC, XLSX, and XLS formats
- **Context-Aware Q&A**: Answers are strictly generated using document content only
- **Embedding + Retrieval**: Uses `OllamaEmbeddingFunction` and `ChromaDB` for long-term memory
- **Relevance Re-ranking**: Applies CrossEncoder ranking for better answer quality
- **LLM-Driven Responses**: Calls LLaMA 3.2 via local Ollama server for streaming answers
- **Auto Logging**: Embedded in interactive Streamlit UI, no CLI use required

---

## 🚀 How It Works

1. **Upload File**: Document is parsed into chunks using LangChain loaders and text splitters.
2. **Vector Storage**: Text chunks are stored in a persistent ChromaDB collection with Ollama embeddings.
3. **Query Input**: User types a question via the Streamlit UI.
4. **Search & Rerank**: Finds top relevant chunks using semantic search, reranked using CrossEncoder.
5. **Generate Answer**: Passes selected content and question to LLaMA 3.2 via Ollama for the final answer.

---

## 🧰 Tech Stack

| Component         | Tool/Library                           |
|------------------|-----------------------------------------|
| LLM               | [Ollama](https://ollama.com/) with `llama3.2` |
| Vector DB         | [ChromaDB](https://www.trychroma.com/) |
| Embedding Model   | `nomic-embed-text` via Ollama client   |
| Reranker          | `cross-encoder/ms-marco-MiniLM-L-6-v2` (HuggingFace) |
| Document Parsing  | LangChain loaders                      |
| UI Framework      | [Streamlit](https://streamlit.io/)     |

---

## 📂 File Upload Support

- ✅ PDF (`.pdf`)
- ✅ Word Docs (`.doc`, `.docx`)
- ✅ Excel Spreadsheets (`.xls`, `.xlsx`)

---

## 🛠 Installation

### 1. Clone the repository

git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name



### 2. Install Dependencies

Make sure Python 3.8 or newer is installed.

pip install -r requirements.txt


> **Note:** Start your local Ollama server and ensure the `llama3.2` and `nomic-embed-text` models are available.

### 3. Run the App

streamlit run app.py



---

## 💡 Sample Usage

1. Upload a document (e.g., a PDF about Indian philosophy).
2. Type a question like:  
   `What does Chapter 2 of the Bhagavad Gita say about duty?`
3. See the answer streamed in real-time, with context derived only from the file.
