"""
RAG (Retrieval-Augmented Generation) 引擎模块

该模块实现了完整的 RAG 系统，包括：
- 文档加载和分块
- 向量化存储（ChromaDB）
- 检索增强生成
- 流式响应支持
"""
import fitz  # PyMuPDF
from rapidocr_onnxruntime import RapidOCR
import os
import re
from typing import List, Optional, Iterator, Dict, Any
from pathlib import Path
from io import BytesIO

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv
import os

# 原有的 import ...
from io import BytesIO
import chromadb


#1. 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 拼接出 .env 的绝对路径
env_path = os.path.join(current_dir, '.env')

# 3. 打印调试信息（让你看着放心）
print(f"正在加载配置文件: {env_path}")

# 4. 强制加载
load_dotenv(dotenv_path=env_path, override=True)


class DocumentProcessor:
    """文档处理器类，负责加载和预处理各种格式的文档"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        # 初始化 OCR 实例 (只初始化一次，避免重复加载模型)
        try:
            self.ocr = RapidOCR()
            self.ocr_available = True
            print("✅ OCR 模块初始化成功 (RapidOCR)")
        except Exception as e:
            print(f"⚠️ OCR 初始化失败: {e}")
            self.ocr_available = False

    def load_pdf(self, file_content: bytes) -> str:
        """
        加载 PDF 文件内容 (支持扫描件 OCR)
        """
        text = ""
        
        # 1. 尝试使用 pypdf 提取文本 (速度快，针对非扫描件)
        try:
            from pypdf import PdfReader
            pdf_reader = PdfReader(BytesIO(file_content))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"⚠️ pypdf 读取出错: {e}，尝试切换到 OCR...")

        # 2. 判断提取结果。如果内容为空或极少(少于50个字)，判定为扫描件，启用 OCR
        if len(text.strip()) < 50:
            if self.ocr_available:
                print("🔍 检测到扫描版 PDF 或文本极少，正在进行 OCR 识别 (速度较慢，请耐心等待)...")
                text = self._ocr_pdf(file_content)
            else:
                text = "无法提取文本，且 OCR 模块未启用。"
        
        return text

    def _ocr_pdf(self, file_content: bytes) -> str:
        """
        使用 PyMuPDF + RapidOCR 进行识别
        """
        ocr_text = ""
        try:
            # 使用 fitz (PyMuPDF) 打开 PDF
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                total_pages = len(doc)
                for i, page in enumerate(doc):
                    # 将页面转换为图片 (dpi=150 兼顾速度和精度)
                    pix = page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    
                    # 调用 RapidOCR 识别
                    result, _ = self.ocr(img_bytes)
                    
                    if result:
                        # result 格式: [[box, text, score], ...]
                        page_content = "\n".join([line[1] for line in result])
                        ocr_text += page_content + "\n"
                    
                    # 打印进度 (因为 OCR 比较慢)
                    print(f"   -> 正在识别第 {i+1}/{total_pages} 页...")
                    
        except Exception as e:
            print(f"❌ OCR 识别过程中出错: {e}")
            return ""
            
        return ocr_text

    def load_markdown(self, file_content: bytes) -> str:
        # ... (保持不变) ...
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            return file_content.decode('gbk', errors='ignore')

    def load_txt(self, file_content: bytes) -> str:
         # ... (保持不变) ...
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            return file_content.decode('gbk', errors='ignore')

    def process_file(self, file_content: bytes, filename: str) -> List[Document]:
        # ... (保持不变，但为了确保安全，我把你的原始逻辑复制在这里) ...
        file_ext = Path(filename).suffix.lower()

        if file_ext == '.pdf':
            text = self.load_pdf(file_content)
        elif file_ext in ['.md', '.markdown']:
            text = self.load_markdown(file_content)
        elif file_ext in ['.txt', '.text']:
            text = self.load_txt(file_content)
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}")

        text = self._clean_text(text)
        
        # 再次检查：如果经过 OCR 还是空的
        if not text.strip():
            print(f"⚠️ 文件 {filename} 处理后内容依然为空。")
            return []

        doc = Document(
            page_content=text,
            metadata={"source": filename, "file_type": file_ext}
        )

        chunks = self.text_splitter.split_documents([doc])
        return chunks

    def _clean_text(self, text: str) -> str:
        # ... (保持不变) ...
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：、""''（）【】]', ' ', text)
        return text.strip()

class VectorStore:
    """向量存储类，负责管理 ChromaDB 向量数据库"""

    def __init__(self, db_path: str, collection_name: str):
        """
        初始化向量存储

        Args:
            db_path: ChromaDB 数据库路径
            collection_name: 集合名称
        """
        self.db_path = db_path
        self.collection_name = collection_name

        # 初始化 Ollama Embeddings
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_model
        )

        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)

        # 初始化 LangChain Chroma
        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加文档到向量数据库

        Args:
            documents: 文档列表

        Returns:
            添加的文档 ID 列表
        """
        return self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            相关文档列表
        """
        return self.vectorstore.similarity_search(query, k=k)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """
        带分数的相似度搜索

        Args:
            query: 查询文本
            k: 返回的文档数量

        Returns:
            (文档, 分数) 元组列表
        """
        return self.vectorstore.similarity_search_with_score(query, k=k)

    def delete_collection(self):
        """删除集合（清空数据库）"""
        try:
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(name=self.collection_name)
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
        except Exception as e:
            print(f"删除集合失败: {str(e)}")


class RAGEngine:
    """RAG 引擎主类，整合文档处理、向量存储和 LLM 调用"""

    def __init__(self):
        """初始化 RAG 引擎"""
        # 加载配置
        self.db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")
        chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))

        # 初始化组件
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.vectorstore = VectorStore(
            db_path=self.db_path,
            collection_name=self.collection_name
        )

        # 初始化 OpenAI 客户端（用于调用 DeepSeek API）
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

         # 直接填入你的 Key (注意保留引号)
        #api_key = "" 
        #base_url = "https://api.deepseek.com"

        if not api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY")

        self.llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )



        if not api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY 环境变量，请在 .env 文件中配置")

        self.llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []

    def add_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """
        添加文档到知识库

        Args:
            file_content: 文件字节内容
            filename: 文件名

        Returns:
            处理结果字典
        """
        try:
            # 处理文档
            documents = self.doc_processor.process_file(file_content, filename)

            # 添加到向量数据库
            doc_ids = self.vectorstore.add_documents(documents)

            return {
                "success": True,
                "message": f"成功添加文档: {filename}",
                "chunks_count": len(documents),
                "doc_ids": doc_ids
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"添加文档失败: {str(e)}",
                "chunks_count": 0,
                "doc_ids": []
            }

    def _build_prompt(self, query: str, context_docs: List[Document]) -> str:
        """
        构建 RAG 提示词 (已修改为混合模式)
        """
        # 如果没有文档，上下文就是空的
        if not context_docs:
            context = "（当前没有相关的知识库内容）"
        else:
            context = "\n\n".join([
                f"[参考片段 {i+1}]\n{doc.page_content}"
                for i, doc in enumerate(context_docs)
            ])

        # 修改提示词，允许模型使用通用知识
        prompt = f"""你是一个智能助手。请参考下面的【知识库片段】来回答用户的【问题】。

【知识库片段】：
{context}

【用户问题】：{query}

回答要求：
1. 如果【知识库片段】中有答案，请优先基于知识库回答。
2. 如果【知识库片段】与问题无关或没有内容，请忽略知识库，直接使用你自己的通用知识来回答用户的问题。
3. 回答要自然、流畅。
"""
        return prompt

    
    def query(self, query: str, stream: bool = False) -> Iterator[str]:
        """
        查询知识库并生成回答（流式）- 已修改为支持通用闲聊
        """
        # 1. 尝试检索相关文档
        # 注意：如果数据库是空的，这里会返回空列表，不会报错
        try:
            context_docs = self.vectorstore.similarity_search(query, k=4)
        except Exception:
            # 如果数据库还没初始化或出错，就当做没文档
            context_docs = []

        # 删除原本的 "if not context_docs: return" 拦截代码
        # 让代码继续往下走，去调用 DeepSeek

        # 2. 构建提示词 (会自动处理 context_docs 为空的情况)
        prompt = self._build_prompt(query, context_docs)

        # 3. 添加系统消息和用户消息
        # 可以在 system 里稍微强化一下人设
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手。既能回答知识库的问题，也能进行日常对话。"},
            {"role": "user", "content": prompt}
        ]

        # 4. 调用 DeepSeek API (保持原样)
        try:
            if stream:
                response = self.llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=True,
                    temperature=0.7
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                response = self.llm_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    stream=False,
                    temperature=0.7
                )
                yield response.choices[0].message.content

        except Exception as e:
            yield f"生成回答时出错: {str(e)}"

    def clear_knowledge_base(self):
        """清空知识库"""
        self.vectorstore.delete_collection()
        self.conversation_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息字典
        """
        try:
            count = self.vectorstore.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }
        except:
            return {
                "total_chunks": 0,
                "collection_name": self.collection_name,
                "db_path": self.db_path
            }

