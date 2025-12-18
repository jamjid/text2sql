# ============================================
# NL2SQL Enterprise Agent (RAG + ReAct + Time-Aware)
# ============================================
import os
import yaml
import json
import datetime
import logging
from typing import TypedDict, Annotated, List, Literal, Optional

# --- LangChain / LangGraph Imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from langchain_community.vectorstores import FAISS

# ==========================================
# 1. 配置加载器
# ==========================================
class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self, path="dev.yaml"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CRITICAL: 配置文件 {path} 不存在！")
        
        with open(path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        log_path = self._config['logging']['file_path']
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    @property
    def config(self):
        return self._config

cfg = ConfigManager().config

# ==========================================
# 2. 数据库与 RAG 检索引擎 (核心升级)
# ==========================================
class DBManager:
    _instance = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DBManager, cls).__new__(cls)
            db_uri = cfg['db']['uri']
            cls._instance._db = SQLDatabase.from_uri(db_uri)
        return cls._instance

    @property
    def db(self):
        return self._db

    def get_table_info(self, table_names: List[str] = None) -> str:
        all_tables = self._db.get_usable_table_names()
        if not table_names:
            return self._db.get_table_info(all_tables)
        
        # 严格过滤，防止 LLM 幻觉出的表名导致报错
        valid_tables = [t for t in table_names if t in all_tables]
        return self._db.get_table_info(valid_tables)

db_manager = DBManager()

# --- [新增] Schema 检索器 (RAG) ---
class SchemaRetriever:
    _instance = None
    _vector_store = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SchemaRetriever, cls).__new__(cls)
            cls._instance._initialize_index()
        return cls._instance
    
    def _initialize_index(self):
        """构建向量索引：将所有表名和结构向量化"""
        print("📥 [System] 正在构建 Schema 向量索引 (RAG)...")
        table_names = db_manager.db.get_usable_table_names()
        docs = []
        
        # 这里为了演示，我们只索引表名和基础 DDL。
        # 生产环境建议索引表的 COMMENT 注释，以支持模糊语义搜索。
        for t in table_names:
            # 获取该表的 DDL 作为内容
            ddl = db_manager.db.get_table_info([t])
            # Metadata 记录表名
            docs.append(Document(page_content=f"Table Name: {t}\nSchema: {ddl}", metadata={"table_name": t}))
            
        if docs:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self._vector_store = FAISS.from_documents(docs, embeddings)
            print(f"   ✅ 已索引 {len(docs)} 张表。")
        else:
            print("   ⚠️ 警告: 数据库为空，跳过索引构建。")

    def retrieve_relevant_schemas(self, query: str, top_k: int = 3) -> str:
        """根据用户问题，检索最相关的表结构"""
        if not self._vector_store:
            return db_manager.get_table_info() # 兜底：返回所有
            
        print(f"   🔍 [RAG] 正在检索与 '{query}' 相关的表...")
        docs = self._vector_store.similarity_search(query, k=top_k)
        
        retrieved_tables = [d.metadata['table_name'] for d in docs]
        # 去重
        retrieved_tables = list(set(retrieved_tables))
        print(f"   🎯 [RAG] 命中表: {retrieved_tables}")
        
        return db_manager.get_table_info(retrieved_tables)

# 初始化 RAG 引擎 (启动时加载)
schema_retriever = SchemaRetriever()

# ==========================================
# 3. 状态定义
# ==========================================
class IntentResult(BaseModel):
    query_type: Literal["statistic", "query", "sort", "unknown"] = Field(...)
    # 注意：有了 RAG，Intent 阶段提取表名的压力变小了，但保留它作为辅助校验依然很好
    keywords: List[str] = Field(default=[])
    complexity: Literal["simple", "complex"] = Field(...)

class AgentState(TypedDict):
    user_input: str
    intent: Optional[dict]
    schema_context: Optional[str]
    generated_sql: Optional[str]
    query_result: Optional[str]
    final_answer: Optional[str]
    error: Optional[str]
    retry_count: int
    approval_status: Optional[str]

# ==========================================
# 4. 路由逻辑
# ==========================================
MAX_RETRIES = 3

def check_is_risky(sql: str) -> bool:
    """
    检查 SQL 是否包含高危操作 或 敏感数据访问
    """
    if not sql: return False
    
    sql_upper = sql.upper()
    
    # 1. [数据破坏风险] DML/DDL 关键词
    # 文中提到: UPDATE, INSERT, DELETE, DROP TABLE
    destructive_keywords = ["DELETE", "UPDATE", "DROP", "ALTER", "TRUNCATE", "INSERT", "GRANT", "REVOKE"]
    for kw in destructive_keywords:
        if kw in sql_upper:
            print(f"   🛡️ [Security] 拦截破坏性操作: {kw}")
            return True
            
    # 2. [数据泄露风险] 敏感字段关键词
    # 文中提到: "查询到它本不应访问的敏感数据（如用户密码）"
    sensitive_keywords = ["PASSWORD", "PASSWD", "SECRET", "HASH", "TOKEN", "API_KEY", "SALARY", "CREDIT_CARD"]
    for kw in sensitive_keywords:
        if kw in sql_upper:
            print(f"   🛡️ [Security] 拦截敏感数据访问: {kw}")
            return True
            
    return False

def check_safety_router(state: AgentState) -> Literal["approve", "execute"]:
    if check_is_risky(state.get("generated_sql", "")):
        print(f"   🛡️ [Router] 风险操作拦截 -> 人工审批")
        return "approve"
    return "execute"

def post_approval_router(state: AgentState) -> Literal["execute", "reject"]:
    if state.get("approval_status") == "approved":
        return "execute"
    return "reject"

def should_continue(state: AgentState) -> Literal["retry", "synthesize"]:
    error = state.get("error")
    retries = state.get("retry_count", 0)
    if error:
        if retries < MAX_RETRIES:
            print(f"   🔄 [Router] 触发 ReAct 修正 (剩余次数: {MAX_RETRIES - retries - 1})")
            return "retry"
        else:
            print(f"   🛑 [Router] 超过重试上限 -> 停止")
            return "synthesize"
    return "synthesize"

# ==========================================
# 5. 节点实现
# ==========================================

def parse_intent_node(state: AgentState):
    print(f"\n🚀 [Node: Intent] 分析: {state['user_input']}")
    llm = ChatOpenAI(model=cfg['llm']['model_name'], temperature=0)
    structured_llm = llm.with_structured_output(IntentResult)
    
    # 增加 rewrite 指令，做轻量级的问题标准化
    system_prompt = """你是一个数据库专家。分析用户意图。
    如果用户输入模糊（如“查下那个啥”），请尽力推断。"""
    
    try:
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        result = (prompt | structured_llm).invoke({"input": state['user_input']})
        print(f"   ✅ 意图: {result.query_type}")
        return {"intent": result.dict()}
    except Exception as e:
        return {"error": f"Intent Error: {e}"}

def generate_sql_node(state: AgentState):
    print(f"\n⚙️ [Node: Generate SQL] ...")
    current_retries = state.get("retry_count", 0)
    user_input = state['user_input']
    
    # --- [Time Aware] 时间注入 ---
    now = datetime.datetime.now()
    current_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    weekday_str = now.strftime("%A")
    
    # --- [RAG] 动态 Schema 检索 ---
    if not state.get("schema_context"):
        schema_context = schema_retriever.retrieve_relevant_schemas(user_input, top_k=3)
    else:
        schema_context = state["schema_context"]

    llm = ChatOpenAI(model=cfg['llm']['model_name'], temperature=0)
    
    # --- [Optimization] 引入 Few-Shot 示例 (源自文档建议) ---
    few_shot_examples = """
    【参考示例】
    问题: "显示所有客户及其订单数量。"
    SQL: SELECT c.name, COUNT(o.order_id) FROM customers c LEFT JOIN orders o ON c.id = o.customer_id GROUP BY c.name;
    
    问题: "哪个产品的单笔订单金额最高？"
    SQL: SELECT product, amount FROM orders ORDER BY amount DESC LIMIT 1;
    """
    
    system_prompt = f"""你是一个 SQL 生成专家。
    
    【环境信息】
    当前时间: {current_time_str} ({weekday_str})
    数据库: SQLite
    
    【相关表结构】
    {schema_context}
    
    {few_shot_examples}

    【任务】
    请根据Schema编写SQL。只输出 SQL 语句，无 Markdown。
    注意：
    1. 涉及日期查询时，请参考【当前时间】。
    2. 严格遵循示例中的 JOIN 和聚合逻辑。
    """
    
    user_prompt = f"用户问题: {user_input}"
    
    # ReAct 错误修正上下文
    last_error = state.get("error")
    if last_error and current_retries > 0:
        print(f"   ⚠️ [Self-Correction] 注入上轮错误信息...")
        user_prompt += f"\n\n上一轮 SQL: {state.get('generated_sql')}\n报错信息: {last_error}\n请修正 SQL。"
        
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)])
    
    try:
        response = (prompt | llm).invoke({})
        sql = response.content.strip().replace("```sql", "").replace("```", "")
        print(f"   💻 SQL: {sql}")
        return {"generated_sql": sql, "schema_context": schema_context}
    except Exception as e:
        return {"error": f"Gen Error: {e}"}

def human_approval_node(state: AgentState):
    print(f"\n✋ [Node: Approval] ⚠️ 高危 SQL 拦截: {state.get('generated_sql')}")
    try:
        decision = input("   👮‍♂️ 允许执行吗? (yes/no): ").strip().lower()
    except: decision = "no"
    
    if decision == "yes":
        return {"approval_status": "approved"}
    return {"approval_status": "rejected", "error": "User rejected execution."}

def execute_sql_node(state: AgentState):
    print(f"\n⚡ [Node: Execute] ...")
    sql = state.get("generated_sql")
    if not sql: return {"error": "No SQL"}
    
    try:
        result = db_manager.db.run(sql)
        print(f"   ✅ 结果: {str(result)[:100]}...") # 只打印前100字符
        return {"query_result": str(result), "error": None} # 成功必须清除 error
    except Exception as e:
        print(f"   ❌ 失败: {e}")
        return {"error": str(e)}

def prepare_retry_node(state: AgentState):
    return {"retry_count": state.get("retry_count", 0) + 1}

def synthesize_answer_node(state: AgentState):
    print(f"\n🗣️ [Node: Synthesize] ...")
    error = state.get("error")
    if error:
        return {"final_answer": f"抱歉，遇到问题: {error}"}
    
    llm = ChatOpenAI(model=cfg['llm']['model_name'], temperature=0.5)
    system_prompt = "你是一个数据分析师。根据数据回答用户问题。保留两位小数。"
    user_prompt = f"问题: {state['user_input']}\nSQL: {state.get('generated_sql')}\n数据: {state.get('query_result')}"
    
    try:
        res = (ChatPromptTemplate.from_messages([("system", system_prompt), ("human", user_prompt)]) | llm).invoke({})
        print(f"   🤖 回答: {res.content}")
        return {"final_answer": res.content}
    except Exception as e:
        return {"final_answer": "合成失败", "error": str(e)}

def log_node(state: AgentState):
    log_file = cfg['logging']['file_path']
    entry = {
        "ts": datetime.datetime.now().isoformat(),
        "query": state["user_input"],
        "sql": state.get("generated_sql"),
        "error": state.get("error")
    }
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except: pass
    return {}

# ==========================================
# 6. 构建图
# ==========================================
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("parse_intent", parse_intent_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("synthesize_answer", synthesize_answer_node)
    workflow.add_node("audit_log", log_node)

    workflow.set_entry_point("parse_intent")
    
    workflow.add_edge("parse_intent", "generate_sql")
    
    workflow.add_conditional_edges(
        "generate_sql", 
        check_safety_router, 
        {"approve": "human_approval", "execute": "execute_sql"}
    )
    
    workflow.add_conditional_edges(
        "human_approval",
        post_approval_router,
        {"execute": "execute_sql", "reject": "synthesize_answer"}
    )

    # ReAct 核心闭环
    workflow.add_conditional_edges(
        "execute_sql",
        should_continue,
        {"retry": "prepare_retry", "synthesize": "synthesize_answer"}
    )
    
    workflow.add_edge("prepare_retry", "generate_sql")
    workflow.add_edge("synthesize_answer", "audit_log")
    workflow.add_edge("audit_log", END)

    return workflow.compile()

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️ 请设置 OPENAI_API_KEY")
    
    app = build_graph()
    
    # 测试 1 模糊表名 (测试 RAG)
    # 假设有一个表叫 'orders'，用户只说 '买卖记录'，RAG 应能通过注释关联(需完善DDL注释)
    # 这里测试 RAG 的表名过滤功能
    print("-" * 50)
    app.invoke({"user_input": "Alice 最近有没有买过 Laptop？"})
    
    # 测试 2 时间感知
    print("\n" + "-" * 50)
    app.invoke({"user_input": "上个月的所有订单总额是多少？"})
    
    print("\n✅ 系统测试完成。")
   
    # 测试3 常规查询
    print(f"🏁 开始测试常规查询...")
    app.invoke({"user_input": "统计 New York 用户的订单总额"})
    
    # 测试 4 高危拦截
    print("\n" + "-" * 50)
    print("🧨 开始测试高危拦截 (请输入 no 拒绝)...")
    app.invoke({"user_input": "把 Alice 的订单金额全部改成 0"})