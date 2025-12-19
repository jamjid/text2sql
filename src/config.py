import os
import yaml

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
            print(f"⚠️ {path} 不存在，正在生成默认配置...")
            default_config = {
                "app": {"name": "Text2SQL", "env": "dev"},
                "llm": {"model_name": "gpt-4o-mini", "temperature": 0}, # 建议 SQL 生成设为 0
                "logging": {"file_path": "data/query_audit.jsonl"},
                "db": {"uri": "sqlite:///data/ecommerce.db"}
            }
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(default_config, f)
            self._config = default_config
        else:
            with open(path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        
        # 确保日志和数据目录存在
        os.makedirs(os.path.dirname(self._config['logging']['file_path']), exist_ok=True)
        os.makedirs(os.path.dirname(self._config['db']['uri'].replace("sqlite:///", "")), exist_ok=True)

    @property
    def config(self):
        return self._config

cfg = ConfigManager().config