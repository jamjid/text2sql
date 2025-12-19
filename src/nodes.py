import datetime
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage

from .config import cfg
from .state import AgentState, IntentResult, SQLOutput
from .database import db_manager, schema_retriever

# ----------------- 节点逻辑 -----------------

def parse_intent_node(state: AgentState):
    print(f"\n🚀 [节点: 意图识别] 分析中: {state['user_input']}")
    llm = ChatOpenAI(model=cfg['llm']['model_name'], temperature=0)
    structured_llm = llm.with_structured_output(IntentResult)
    
    # 🚀 提示词优化：明确要求中文交互
    system_prompt = """你是一个专业的数据库专家。请分析用户的意图。
    
    【关键规则】
    1. 如果用户的问题非常模糊（例如“最好的产品”但未定义是销量最高还是评分最高），请将 needs_clarification 设为 True。
    2. 在 clarification_question 中用自然的中文生成反问句，例如：“您指的是销量最好还是评分最高？”。
    3. 如果可以通过常识推断（例如“卖得最好的”隐含指销量），则不需要澄清。
    """
    
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt), 
            ("human", "{input}")
        ])
        result = (prompt | structured_llm).invoke({"input": state['user_input']})
        
        clarify_str = "是" if result.needs_clarification else "否"
        print(f"   ✅ 识别结果: {result.query_type} | 需要澄清: {clarify_str}")
        return {"intent": result.model_dump()}
    except Exception as e:
        return {"error": f"意图识别错误: {e}"}

def handle_clarification_node(state: AgentState):
    intent = state.get("intent", {})
    question = intent.get("clarification_question", "能否请您详细说明一下您的具体需求？")
    print(f"   ❓ [请求澄清] 追问用户: {question}")
    return {"final_answer": question}

# ... (generate_sql_node 逻辑复用之前的，但确保提示词是中文) ...

def validate_sql_node(state: AgentState):
    print(f"\n🔍 [节点: SQL预校验] 正在检查语法...")
    sql = state.get("generated_sql")
    if not sql: return {"error": "生成的 SQL 为空"}

    # 使用 EXPLAIN QUERY PLAN 进行无副作用的语法检查
    try:
        explain_sql = f"EXPLAIN QUERY PLAN {sql}"
        db_manager.db.run(explain_sql)
        print(f"   ✅ 语法校验通过")
        return {"error": None} 
    except Exception as e:
        print(f"   ❌ 语法校验失败: {e}")
        return {"error": f"SQL语法错误: {e}"}

# 注意：generate_sql_node, execute_sql_node 等其他节点复用之前的逻辑即可，
# 只要确保 print 内容你自己能看懂即可。核心是 Prompt 是中文。