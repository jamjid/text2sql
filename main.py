import os
from langchain_core.messages import HumanMessage, AIMessage

# 导入模块 (确保这些文件都在 src 目录下)
from src.utils import auto_initialize_database
from src.database import db_manager, schema_retriever
from src.graph import build_graph

def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️ 请设置 OPENAI_API_KEY")
        # return 

    # 1. 初始化数据库与索引
    auto_initialize_database()
    db_manager.refresh_db_connection()
    schema_retriever._initialize_index()
    
    # 2. 构建图
    app = build_graph()
    
    # 3. 启动交互循环
    print("\n" + "="*50)
    print("🤖 Enterprise Text2SQL Agent (v2.1 Modular)")
    print("支持多轮对话、RAG 增强、自愈修正、意图澄清")
    print("="*50)
    
    chat_history = [] # 本地会话记录
    
    while True:
        try:
            q = input("\nuser > ").strip()
            if q.lower() in ["exit", "quit", "q"]:
                break
            if not q: continue
            
            # 构造输入状态
            inputs = {
                "user_input": q,
                "chat_history": chat_history, # 注入历史
                "retry_count": 0
            }
            
            # 执行图
            for event in app.stream(inputs):
                pass
                
            # 获取最终结果
            result = app.invoke(inputs)
            final_answer = result.get("final_answer", "No answer")
            
            # 更新历史
            chat_history.append(HumanMessage(content=q))
            chat_history.append(AIMessage(content=final_answer))
            
            # 限制历史长度 (滑动窗口)
            if len(chat_history) > 10:
                chat_history = chat_history[-10:]
                
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"❌ System Error: {e}")

if __name__ == "__main__":
    main()