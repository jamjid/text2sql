from typing import TypedDict, Optional, List, Literal
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

# 🚀 [汉化] 意图解析模型
class IntentResult(BaseModel):
    query_type: Literal["statistic", "query", "sort", "unknown"] = Field(
        ..., 
        description="用户问题的类型：统计(statistic)、查询(query)、排序(sort) 或 未知(unknown)"
    )
    complexity: Literal["simple", "complex"] = Field(
        ..., 
        description="SQL 查询的复杂度：简单(simple) 或 复杂(complex)"
    )
    # 新增: 澄清字段 (汉化描述)
    needs_clarification: bool = Field(
        description="如果用户问题模糊不清且需要澄清，则为 True；否则为 False。"
    )
    clarification_question: Optional[str] = Field(
        description="如果需要澄清，此处填写向用户反问的具体问题（请用中文）。"
    )

# 🚀 [汉化] SQL 输出模型
class SQLOutput(BaseModel):
    sql_query: str = Field(..., description="生成的最终可执行 SQL 语句。")
    chain_of_thought: str = Field(..., description="生成 SQL 的思考过程和逻辑推演（请用中文描述）。")

class AgentState(TypedDict):
    user_input: str
    chat_history: List[BaseMessage]
    
    intent: Optional[dict]
    schema_context: Optional[str]
    generated_sql: Optional[str]
    query_result: Optional[str]
    final_answer: Optional[str]
    
    error: Optional[str]
    retry_count: int
    approval_status: Optional[str]