from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List
from .config import cfg

class DBManager:
    _instance = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
            # 这里的 uri 已经在 config 中处理过
            db_uri = cfg['db']['uri']
            cls._instance._db = SQLDatabase.from_uri(db_uri)
        return cls._instance

    @property
    def db(self):
        return self._db

    def refresh_db_connection(self):
        db_uri = cfg['db']['uri']
        self._db = SQLDatabase.from_uri(db_uri)
        print("🔄 [DBManager] 数据库连接已刷新。")

    def get_table_info(self, table_names: List[str] = None) -> str:
        all_tables = self._db.get_usable_table_names()
        if not table_names:
            return self._db.get_table_info(all_tables)
        valid_tables = [t for t in table_names if t in all_tables]
        return self._db.get_table_info(valid_tables)

db_manager = DBManager()

class SchemaRetriever:
    _instance = None
    _vector_store = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SchemaRetriever, cls).__new__(cls)
            cls._instance._initialize_index()
        return cls._instance
    
    def _initialize_index(self):
        print("📥 [System] 正在构建 RAG 索引...")
        try:
            table_names = db_manager.db.get_usable_table_names()
            if not table_names:
                print("   ⚠️ 警告: 数据库为空，跳过索引。")
                return

            docs = []
            for t in table_names:
                ddl = db_manager.db.get_table_info([t])
                docs.append(Document(page_content=f"Table: {t}\nSchema: {ddl}", metadata={"table_name": t}))
            
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self._vector_store = FAISS.from_documents(docs, embeddings)
            print(f"   ✅ RAG 索引构建成功 ({len(docs)} 表)。")
        except Exception as e:
            print(f"   ❌ RAG 索引失败: {e}")

    def retrieve_relevant_schemas(self, query: str, top_k: int = 3) -> str:
        if not self._vector_store:
            return db_manager.get_table_info()
        docs = self._vector_store.similarity_search(query, k=top_k)
        retrieved = list(set([d.metadata['table_name'] for d in docs]))
        return db_manager.get_table_info(retrieved)

schema_retriever = SchemaRetriever()