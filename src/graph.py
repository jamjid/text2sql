from langgraph.graph import StateGraph, END
from typing import Literal

from .state import AgentState
from .utils import check_is_risky
from .nodes import (
    parse_intent_node, handle_clarification_node,
    generate_sql_node, validate_sql_node,
    human_approval_node, execute_sql_node, 
    prepare_retry_node, synthesize_answer_node, log_node
)

MAX_RETRIES = 3

# --- 路由逻辑 ---

def intent_router(state: AgentState) -> Literal["clarify", "generate"]:
    intent = state.get("intent", {})
    if intent.get("needs_clarification"):
        return "clarify"
    return "generate"

def validation_router(state: AgentState) -> Literal["approve", "retry"]:
    if state.get("error"):
        print(f"   🔄 [路由] 校验未通过 -> 触发自动修正")
        return "retry"
    
    if check_is_risky(state.get("generated_sql", "")):
        return "approve"
    return "execute"

def post_approval_router(state: AgentState) -> Literal["execute", "reject"]:
    if state.get("approval_status") == "approved":
        return "execute"
    return "reject"

def post_execute_router(state: AgentState) -> Literal["synthesize", "retry"]:
    error = state.get("error")
    retries = state.get("retry_count", 0)
    if error and retries < MAX_RETRIES:
        print(f"   🔄 [路由] 执行报错 -> 触发 ReAct 重试 (剩余: {MAX_RETRIES - retries - 1})")
        return "retry"
    return "synthesize"

# --- 构建图 ---
def build_graph():
    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("parse_intent", parse_intent_node)
    workflow.add_node("handle_clarification", handle_clarification_node)
    workflow.add_node("generate_sql", generate_sql_node)
    workflow.add_node("validate_sql", validate_sql_node)
    workflow.add_node("human_approval", human_approval_node)
    workflow.add_node("execute_sql", execute_sql_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("synthesize_answer", synthesize_answer_node)
    workflow.add_node("audit_log", log_node)

    # 设置入口
    workflow.set_entry_point("parse_intent")
    
    # 意图路由
    workflow.add_conditional_edges(
        "parse_intent",
        intent_router,
        {
            "clarify": "handle_clarification",
            "generate": "generate_sql"
        }
    )
    workflow.add_edge("handle_clarification", "audit_log")

    # SQL 生成与校验
    workflow.add_edge("generate_sql", "validate_sql")
    workflow.add_conditional_edges(
        "validate_sql",
        validation_router,
        {
            "retry": "prepare_retry",
            "approve": "human_approval",
            "execute": "execute_sql"
        }
    )

    workflow.add_edge("prepare_retry", "generate_sql")
    
    # 审批路由
    workflow.add_conditional_edges(
        "human_approval",
        post_approval_router,
        {"execute": "execute_sql", "reject": "synthesize_answer"}
    )

    # 执行路由
    workflow.add_conditional_edges(
        "execute_sql",
        post_execute_router,
        {"retry": "prepare_retry", "synthesize": "synthesize_answer"}
    )
    
    workflow.add_edge("synthesize_answer", "audit_log")
    workflow.add_edge("audit_log", END)

    return workflow.compile()