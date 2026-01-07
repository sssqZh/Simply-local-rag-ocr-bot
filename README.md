# 📚 Local-Doc-Chat-OCR: RAG Assistant with Vision

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20V3-purple)](https://www.deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[**English**](README.md) | [**中文说明**](README_CN.md)

<img width="100%" alt="Image" src="https://github.com/user-attachments/assets/4ec13e1a-8b76-40b2-beea-12d2ee53771a" />

A local **RAG (Retrieval-Augmented Generation)** Q&A system built with **Streamlit**. 


Unlike traditional RAG tools, this project integrates **OCR (Optical Character Recognition)** capabilities, allowing you to chat not only with text documents but also with **scanned PDFs** and **images**.

Powered by **DeepSeek V3** (for high-performance reasoning) and local **Ollama** (for privacy-preserving embedding).

## ⚡️ Update：
- **For better Chinese understanding**。
  - Prerequisites：
  - Please download Ollama。
**And run this in terminal**：

```
ollama pull bge-m3
```


## ✨ Core Features

- **📄 Universal Document Support**:
  - **PDF**: Handles both standard text PDFs and **Scanned/Image-based PDFs** (Auto-triggers OCR).
  - **Markdown/TXT**: Supports common text formats.
- **👁️ Built-in OCR Engine**:
  - Integrated `RapidOCR` + `PyMuPDF` for local text extraction. No need for third-party OCR APIs.
- **🧠 Hybrid AI Architecture**:
  - **LLM**: DeepSeek API (OpenAI SDK Compatible).
  - **Embedding**: Local Ollama (`all-minilm`), zero-cost & privacy-first.
  - **Vector DB**: ChromaDB for local persistence.
- **💬 Streaming Interaction**:
  - Real-time typewriter effect responses.

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | Streamlit | Lightweight Python Web Framework |
| **LLM** | DeepSeek API | High performance, low cost reasoning model |
| **Embedding** | Ollama | Running `all-minilm` locally |
| **Vector DB** | ChromaDB | Local vector storage |
| **OCR** | RapidOCR | ONNX-based offline OCR engine |
| **ETL** | PyMuPDF (fitz) | PDF parsing and image extraction |

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have [Python 3.8+](https://www.python.org/) and [Ollama](https://ollama.ai/) installed.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/local-rag-ocr-bot.git
cd local-rag-ocr-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
*Note: The OCR libraries are relatively large, so the download might take a moment.*

### 3. Prepare Model (Ollama)

Pull the embedding model in your terminal:

```bash
ollama pull all-minilm
```
*Make sure the Ollama service is running in the background.*

### 4. Configure Environment

Copy the example configuration file:

```bash
# Windows
copy .env.example .env
# Mac/Linux
cp .env.example .env
```

Open `.env` and fill in your DeepSeek API Key:

```ini
# Your DeepSeek API Key
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Keep others as default
DEEPSEEK_BASE_URL=https://api.deepseek.com
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=all-minilm
CHROMA_DB_PATH=./chroma_db
```

### 5. Run App

```bash
streamlit run app.py
```

The browser will automatically open at `http://localhost:8501`.

## 📂 Project Structure

```text
.
├── app.py                  # Main Streamlit application
├── rag_engine.py           # Core logic (OCR, Vectorization, RAG)
├── requirements.txt        # Python dependencies
├── .env.example            # Env template (Safe to commit)
├── .gitignore              # Git ignore rules
├── README.md               # English Documentation
└── README_CN.md            # Chinese Documentation
```

## ⚠️ Notes

1.  **OCR Speed**: If you upload a scanned PDF, the system performs page-by-page recognition. Depending on your CPU, this may take longer than processing standard text. Please watch the terminal for progress.
2.  **DeepSeek Quota**: Ensure your API Key has sufficient balance.
3.  **Reset Data**: To clear the knowledge base, click the "Clear Knowledge Base" button in the sidebar or manually delete the local `chroma_db` folder.

## 🙌 Acknowledgments

Special thanks to the following tools that made this project possible:

- **[Cursor](https://cursor.sh/)**: For the incredible AI-assisted coding experience.
- **[Google Gemini](https://deepmind.google/technologies/gemini/)**: For providing architectural advice and debugging help.
- **[DeepSeek](https://www.deepseek.com/)**: For the powerful reasoning API.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
Feel free to Fork and Star!
