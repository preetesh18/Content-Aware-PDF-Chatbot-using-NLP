import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredFileLoader
from langchain_core import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
import ollama
import requests
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import pandas as pd  # For Excel file handling

system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""


def process_document(uploaded_file: UploadedFile) -> list[documents.Document]:
    """Processes an uploaded file (PDF, Excel, Word) into text chunks."""
    temp_file = tempfile.NamedTemporaryFile("wb", suffix=f".{uploaded_file.name.split('.')[-1]}", delete=False)
    try:
        temp_file.write(uploaded_file.read())
        temp_file.close()

        # Load the file based on its type
        file_extension = uploaded_file.name.split(".")[-1].lower()
        if file_extension == "pdf":
            loader = PyMuPDFLoader(temp_file.name)
            docs = loader.load()
        elif file_extension in ["docx", "doc"]:
            loader = UnstructuredFileLoader(temp_file.name)
            docs = loader.load()
        elif file_extension in ["xlsx", "xls"]:
            # Handle Excel files specifically
            df = pd.read_excel(temp_file.name)
            # Convert the DataFrame to a string representation
            excel_content = df.to_string(index=False)
            docs = [documents.Document(page_content=excel_content, metadata={"source": uploaded_file.name})]
        else:
            raise ValueError("Unsupported file type. Please upload a PDF, Word, or Excel file.")

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
        return text_splitter.split_documents(docs)
    finally:
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

def get_vector_collection() -> chromadb.Collection:
    """Gets or creates a ChromaDB collection for vector storage."""
    ollama_ef = OllamaEmbeddingFunction(url="http://localhost:11434/api/embeddings", model_name="nomic-embed-text:latest")
    chroma_client = chromadb.PersistentClient(path="./demo-rag-chroma")
    return chroma_client.get_or_create_collection(name="rag_app", embedding_function=ollama_ef)

def add_to_vector_collection(all_splits: list[documents.Document], file_name: str):
    """Adds document splits to the vector collection."""
    collection = get_vector_collection()
    documents, metadatas, ids = [], [], []
    for idx, split in enumerate(all_splits):
        documents.append(split.page_content)
        metadatas.append(split.metadata)
        ids.append(f"{file_name}_{idx}")
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    st.success("Data added to the vector store!")

def query_collection(prompt: str, n_results: int = 10):
    """Queries the vector collection for relevant documents."""
    collection = get_vector_collection()
    return collection.query(query_texts=[prompt], n_results=n_results)

def call_llm(context: str, prompt: str):
    """Calls the LLM to generate a concise response."""
    try:
        response = ollama.chat(
            model="llama3.2:latest",
            stream=True,
            options={"temperature": 0.5},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}, Question: {prompt}"},
            ],
        )
        for chunk in response:
            if not chunk["done"]:
                yield chunk["message"]["content"]
    except Exception as e:
        yield f"Error generating response: {str(e)}"

def re_rank_cross_encoders(query: str, documents: list[str]) -> tuple[str, list[int]]:
    """Re-ranks documents using a cross-encoder for better relevance."""
    encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = [encoder_model.predict([(query, doc)])[0] for doc in documents]
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    relevant_text_ids = ranked_indices[:3]
    relevant_text = " ".join([documents[i] for i in relevant_text_ids])
    return relevant_text, relevant_text_ids

if __name__ == "__main__":
    st.set_page_config(page_title="AI Chatbot for Content-Awareness", page_icon="🤖", layout="wide")
    with st.sidebar:
        uploaded_file = st.file_uploader("📑 Upload files for QnA PDF ", type=["pdf", "docx", "doc", "xlsx", "xls"])
        if st.button("⚡️ Process") and uploaded_file:
            all_splits = process_document(uploaded_file)
            add_to_vector_collection(all_splits, uploaded_file.name.translate(str.maketrans({"-": "_", ".": "_", " ": "_"})))

    st.header("AI CHATBOT for content-awareness from the refernece Bhagavad Gita BUILT BY PREETESH KUMAR SINGHA")
    prompt = st.text_area("Ask a question related to your document:")
    if st.button("🔥 Ask") and prompt:
        results = query_collection(prompt)
        context = results.get("documents")[0]
        relevant_text, relevant_text_ids = re_rank_cross_encoders(prompt, context)
        response = call_llm(context=relevant_text, prompt=prompt)
        st.write_stream(response)

        with st.expander("See retrieved documents"):
            st.write(results)
        with st.expander("See most relevant document ids"):
            st.write(relevant_text_ids)