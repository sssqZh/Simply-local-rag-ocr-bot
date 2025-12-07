# 📚 本地 RAG 智能知识库助手 (支持 OCR & DeepSeek)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek%20V3-purple)](https://www.deepseek.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[**English**](README.md) | [**中文说明**](README_CN.md)

<img width="100%" alt="Image" src="https://github.com/user-attachments/assets/4ec13e1a-8b76-40b2-beea-12d2ee53771a" />

这是一个基于 **Streamlit** 构建的本地 RAG（检索增强生成）问答系统。它不仅支持普通的文本和文档，还集成了 **OCR (光学字符识别)** 技术，能够处理**扫描版 PDF** 和图片型文档。

后端模型采用高性价比的 **DeepSeek V3**，配合本地运行的 **Ollama** 进行隐私安全的向量嵌入。

## ✨ 核心功能

- **📄 全能文档支持**：
  - **PDF**: 支持普通文本 PDF 及 **扫描件/纯图片 PDF** (自动触发 OCR)。
  - **Markdown/TXT**: 支持常见文本格式。
- **👁️ 内置 OCR 引擎**：
  - 集成 `RapidOCR` + `PyMuPDF`，在本地直接识别文档文字，无需上传到第三方 OCR 平台。
- **🧠 混合 AI 架构**：
  - **LLM**: DeepSeek API (OpenAI SDK 兼容)。
  - **Embedding**: 本地运行 Ollama (`all-minilm`)，零成本且保护隐私。
  - **Vector DB**: ChromaDB 本地持久化存储，重启不丢失数据。
- **💬 流式交互**：
  - 类似 ChatGPT 的打字机效果，实时生成回答。

## 🛠️ 技术栈

| 组件 | 技术选型 | 说明 |
| :--- | :--- | :--- |
| **前端** | Streamlit | 极简 Python Web 框架 |
| **大模型** | DeepSeek API | 高性能、低成本的推理模型 |
| **Embedding** | Ollama | 本地运行 `all-minilm` 模型 |
| **向量库** | ChromaDB | 轻量级本地向量数据库 |
| **OCR** | RapidOCR | 基于 ONNX 的离线 OCR 引擎 |
| **文档处理** | PyMuPDF (fitz) | 高效 PDF 解析与图片提取 |

## 🚀 快速开始

### 1. 环境准备

确保已安装 [Python 3.8+](https://www.python.org/) 和 [Ollama](https://ollama.ai/)。

```bash
# 克隆项目
git clone https://github.com/你的用户名/local-rag-ocr-bot.git
cd local-rag-ocr-bot
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```
*注意：OCR 库依赖较大，下载可能需要一点时间。*

### 3. 模型准备 (Ollama)

在终端运行以下命令，拉取嵌入模型：

```bash
ollama pull all-minilm
```
*请确保 Ollama 服务在后台运行。*

### 4. 配置环境变量

复制配置模板：

```bash
# Windows
copy .env.example .env
# Mac/Linux
cp .env.example .env
```

打开 `.env` 文件，填入你的配置：

```ini
# 你的 DeepSeek API Key
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# 其他配置保持默认即可
DEEPSEEK_BASE_URL=https://api.deepseek.com
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=all-minilm
CHROMA_DB_PATH=./chroma_db
```

### 5. 运行应用

```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`。

## 📂 项目结构

```text
.
├── app.py                  # Streamlit 前端主程序
├── rag_engine.py           # 核心逻辑 (OCR处理、向量化、RAG检索)
├── requirements.txt        # 项目依赖列表
├── .env.example            # 环境变量模板 (安全)
├── .gitignore              # Git 忽略配置
└── README.md               # 项目说明文档
```

## ⚠️ 使用注意事项

1.  **OCR 识别速度**：如果你上传的是扫描版 PDF，系统会自动进行逐页识别。取决于你的电脑性能，这可能比普通文档处理慢一些，请留意终端的进度提示。
2.  **DeepSeek 额度**：请确保你的 API Key 有充足的余额。
3.  **数据重置**：如果想清空知识库，只需点击侧边栏的“清空知识库”按钮，或者手动删除本地的 `chroma_db` 文件夹。

## 🙌 致谢

特别感谢以下工具对本项目开发的巨大帮助：

- **[Cursor](https://cursor.sh/)**: 提供了极致的 AI 辅助编程体验，极大地加速了开发进程。
- **[Google Gemini](https://deepmind.google/technologies/gemini/)**: 在架构设计思路与代码调试方面提供了宝贵的建议。
- **[DeepSeek](https://www.deepseek.com/)**: 提供了强大的推理 API 支持。

## 📄 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源许可证。
仅供学习和研究使用，欢迎 Fork 和 Star！
```
