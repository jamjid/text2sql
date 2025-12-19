import os
import sqlite3

def check_is_risky(sql: str) -> bool:
    """
    [安全组件] 检查 SQL 是否包含高危操作
    """
    if not sql: return False
    sql_upper = sql.upper()
    
    # 1. 破坏性操作拦截
    risky_keywords = ["DELETE", "UPDATE", "DROP", "ALTER", "TRUNCATE", "INSERT", "GRANT", "REVOKE"]
    for kw in risky_keywords:
        if kw in sql_upper:
            print(f"   🛡️ [安全拦截] 检测到高危指令: {kw}")
            return True
            
    # 2. 敏感数据拦截
    sensitive_keywords = ["PASSWORD", "PASSWD", "SECRET", "HASH", "TOKEN", "API_KEY", "CREDIT_CARD"]
    for kw in sensitive_keywords:
        if kw in sql_upper:
            print(f"   🛡️ [安全拦截] 检测到敏感数据访问: {kw}")
            return True     
    return False

def auto_initialize_database(db_path="data/ecommerce.db"):
    """
    [初始化组件] 自动检测并生成测试数据库
    """
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    if os.path.exists(db_path):
        # 简单检查大小，如果文件有内容则跳过初始化
        if os.path.getsize(db_path) > 0:
            print(f"📦 [系统] 检测到现有数据库 {db_path}，跳过初始化。")
            return

    print(f"📦 [系统] 正在初始化测试数据库 {db_path} ...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 定义初始化 SQL：包含两张表 (customers, orders) 和测试数据
    init_script = """
    -- 创建用户表
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY, 
        name VARCHAR(50), 
        age INTEGER, 
        city VARCHAR(50)
    );
    
    -- 创建订单表
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY, 
        customer_id INTEGER, 
        product VARCHAR(50), 
        amount DECIMAL(10, 2), 
        order_date DATE, 
        FOREIGN KEY(customer_id) REFERENCES customers(id)
    );
    
    -- 写入测试数据 (用户)
    INSERT INTO customers (id, name, age, city) VALUES 
        (1, 'Alice', 30, 'New York'), 
        (2, 'Bob', 25, 'Los Angeles'), 
        (3, 'Charlie', 35, 'Chicago'), 
        (4, 'Diana', 28, 'New York');
        
    -- 写入测试数据 (订单)
    INSERT INTO orders (order_id, customer_id, product, amount, order_date) VALUES 
        (101, 1, 'Laptop', 1200.00, '2023-10-01'), 
        (102, 1, 'Mouse', 25.00, '2023-10-02'), 
        (103, 2, 'Smartphone', 800.00, '2023-10-03'), 
        (104, 1, 'Keyboard', 100.00, '2023-10-05'), 
        (105, 3, 'Headphones', 150.00, '2023-10-06'), 
        (106, 4, 'Monitor', 300.00, '2023-10-07');
    """
    try:
        cursor.executescript(init_script)
        conn.commit()
        print(f"   ✅ 测试数据写入完成 (包含 Users, Orders 表)。")
    except Exception as e:
        print(f"   ❌ 数据库初始化失败: {e}")
    finally:
        conn.close()