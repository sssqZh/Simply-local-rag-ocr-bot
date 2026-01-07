"""
RAG (Retrieval-Augmented Generation) 引擎模块 - 优化版
修改记录：
1. 强制将 Embedding 模型指定为 'bge-m3' (解决中文检索问题)。
2. 清理了重复的 API Key 初始化逻辑。
3. 增加了调试打印，方便查看当前使用的模型。
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

# --- 1. 环境配置加载 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, '.env')
print(f"📂 正在加载配置文件: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)


class DocumentProcessor:
    """文档处理器类：负责加载、OCR识别和分块"""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # 修改建议：中文文档 chunk_size 稍微调小一点，overlap 适中，有助于提高检索精度
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            # 针对中文优化分隔符优先级
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        # 初始化 OCR
        try:
            self.ocr = RapidOCR()
            self.ocr_available = True
            print("✅ OCR 模块初始化成功 (RapidOCR)")
        except Exception as e:
            print(f"⚠️ OCR 初始化失败: {e}")
            self.ocr_available = False

    def load_pdf(self, file_content: bytes) -> str:
        text = ""
        # 1. 尝试直接提取
        try:
            from pypdf import PdfReader
            pdf_reader = PdfReader(BytesIO(file_content))
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            print(f"⚠️ pypdf 读取出错: {e}，尝试切换到 OCR...")

        # 2. 如果提取内容极少，判定为扫描件，启用 OCR
        if len(text.strip()) < 50:
            if self.ocr_available:
                print("🔍 检测到扫描版 PDF，正在进行 OCR 识别 (速度较慢，请耐心)...")
                text = self._ocr_pdf(file_content)
            else:
                text = "无法提取文本，且 OCR 模块未启用。"
        
        return text

    def _ocr_pdf(self, file_content: bytes) -> str:
        ocr_text = ""
        try:
            with fitz.open(stream=file_content, filetype="pdf") as doc:
                total_pages = len(doc)
                for i, page in enumerate(doc):
                    pix = page.get_pixmap(dpi=150) # 150 dpi 兼顾速度
                    img_bytes = pix.tobytes("png")
                    result, _ = self.ocr(img_bytes)
                    if result:
                        page_content = "\n".join([line[1] for line in result])
                        ocr_text += page_content + "\n"
                    if (i + 1) % 5 == 0:
                        print(f"   -> OCR 进度: {i+1}/{total_pages} 页...")
        except Exception as e:
            print(f"❌ OCR 出错: {e}")
            return ""
        return ocr_text

    def load_markdown(self, file_content: bytes) -> str:
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            return file_content.decode('gbk', errors='ignore')

    def load_txt(self, file_content: bytes) -> str:
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            return file_content.decode('gbk', errors='ignore')

    def process_file(self, file_content: bytes, filename: str) -> List[Document]:
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
        if not text.strip():
            return []

        doc = Document(
            page_content=text,
            metadata={"source": filename, "file_type": file_ext}
        )
        return self.text_splitter.split_documents([doc])

    def _clean_text(self, text: str) -> str:
        # 简单的清洗，保留中文标点
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


class VectorStore:
    """向量存储类：管理 ChromaDB"""

    def __init__(self, db_path: str, collection_name: str):
        self.db_path = db_path
        self.collection_name = collection_name

        # --- 核心修改：强制使用 bge-m3 模型 ---
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # ⚠️ 强制指定为 bge-m3，这是解决“死锁”搜不到的关键
        # 如果环境变量没设，就默认 bge-m3
        ollama_model = os.getenv("OLLAMA_MODEL", "bge-m3") 

        print(f"🧠 正在初始化 Embedding 模型: {ollama_model} (地址: {ollama_base_url})")

        try:
            self.embeddings = OllamaEmbeddings(
                base_url=ollama_base_url,
                model=ollama_model
            )
        except Exception as e:
            print(f"❌ Embedding 模型初始化失败: {e}")
            raise e

        # 初始化 ChromaDB
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )

        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(name=collection_name)

        self.vectorstore = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List[Document]) -> List[str]:
        return self.vectorstore.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            # 重新创建
            self.collection = self.client.create_collection(name=self.collection_name)
            # 重新绑定 LangChain 接口
            self.vectorstore = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            print("🗑️ 知识库已清空")
        except Exception as e:
            print(f"删除集合失败: {str(e)}")


class RAGEngine:
    """RAG 引擎主类"""

    def __init__(self):
        # 1. 向量库配置
        self.db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "knowledge_base")
        chunk_size = int(os.getenv("MAX_CHUNK_SIZE", "800"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

        # 2. 初始化文档处理和向量库
        self.doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vectorstore = VectorStore(db_path=self.db_path, collection_name=self.collection_name)

        # 3. 初始化 DeepSeek API (清理了重复代码)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        if not api_key:
            print("❌ 错误: 未找到 DEEPSEEK_API_KEY，请检查 .env 文件")
            raise ValueError("API Key Missing")

        print(f"🤖 正在连接 DeepSeek API...")
        self.llm_client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        self.conversation_history: List[Dict[str, str]] = []

    def add_document(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        """添加文档"""
        try:
            documents = self.doc_processor.process_file(file_content, filename)
            if not documents:
                return {"success": False, "message": "文档内容为空或无法识别", "chunks_count": 0}
            
            doc_ids = self.vectorstore.add_documents(documents)
            return {
                "success": True,
                "message": f"成功添加: {filename}",
                "chunks_count": len(documents),
                "doc_ids": doc_ids
            }
        except Exception as e:
            return {"success": False, "message": f"添加失败: {str(e)}", "chunks_count": 0}

    def _build_prompt(self, query: str, context_docs: List[Document]) -> str:
        """构建提示词"""
        if not context_docs:
            context = "（没有检索到相关背景信息）"
        else:
            context = "\n\n".join([f"[参考片段 {i+1}]\n{doc.page_content}" for i, doc in enumerate(context_docs)])

        prompt = f"""你是一个专业的工程知识助手。请基于下面的【参考资料】回答用户的【问题】。

【参考资料】：
{context}

【问题】：{query}

要求：
1. 如果参考资料中有答案，请详细引用资料回答。
2. 如果参考资料与问题无关，请忽略资料，利用你的通用知识回答。
3. 回答要条理清晰，适合工程管理人员阅读。
"""
        return prompt

    def query(self, query: str, stream: bool = False) -> Iterator[str]:
        """查询入口"""
        # 1. 检索 (Top-K 设为 4 或 5，给 DeepSeek 更多上下文)
        try:
            print(f"🔍 正在检索: {query}")
            context_docs = self.vectorstore.similarity_search(query, k=5)
            # 调试：打印检索到的片段前50个字，看看准不准
            for i, doc in enumerate(context_docs):
                print(f"   [片段{i+1}] {doc.page_content[:50].replace(chr(10), ' ')}...")
        except Exception as e:
            print(f"⚠️ 检索出错 (可能是库为空): {e}")
            context_docs = []

        # 2. 构建提示词
        prompt = self._build_prompt(query, context_docs)

        # 3. 消息历史 (System Prompt + User Prompt)
        messages = [
            {"role": "system", "content": "你是一个乐于助人的智能助手。"},
            {"role": "user", "content": prompt}
        ]

        # 4. 调用 DeepSeek
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=stream,
                temperature=0.3 # 降低温度，让回答更严谨
            )
            
            if stream:
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                yield response.choices[0].message.content

        except Exception as e:
            yield f"API 调用出错: {str(e)}"

    def clear_knowledge_base(self):
        self.vectorstore.delete_collection()
        self.conversation_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息
        """
        try:
            # 尝试获取真实的 chunk 数量
            count = self.vectorstore.collection.count()
        except:
            count = 0
            
        # 必须返回 collection_name 和 db_path，防止 app.py 报错
        return {
            "total_chunks": count,
            "collection_name": self.collection_name, 
            "db_path": self.db_path,
            "model": "bge-m3 + deepseek-chat"
        }