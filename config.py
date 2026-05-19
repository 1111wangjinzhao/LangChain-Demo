"""
config.py — 全局配置中心

生产实践要点：
  - 所有敏感信息（API Key）通过环境变量注入，不硬编码在代码中
  - 本地开发：在项目根目录创建 .env 文件（参考 .env.example），
    或在终端中 export DEEPSEEK_API_KEY=your_key
  - 生产部署：通过 CI/CD 环境变量或 Secrets Manager 注入
"""
import os
from pathlib import Path

# 自动加载 .env 文件（如果存在），方便本地开发
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv 未安装时跳过，直接依赖系统环境变量

# ============================================================
# 项目根目录
# ============================================================
BASE_DIR = Path(__file__).parent

# ============================================================
# LLM 配置 (DeepSeek，OpenAI 兼容接口)
# 生产环境：从环境变量读取，例如 os.environ["DEEPSEEK_API_KEY"]
# ============================================================
LLM_CONFIG = {
    "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    "api_key": os.environ["DEEPSEEK_API_KEY"],  # 必须通过环境变量提供，不提供则启动时报错
    "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    "temperature": 0,
}

# ============================================================
# 短期记忆 (Session Memory) 配置
# 使用 SQLite 作为 Checkpointer 的存储后端
#
# 【架构说明】短期记忆 vs 长期记忆
#   短期记忆：checkpointer 负责，以 thread_id 为键，存储当前会话的完整 State 快照
#             用于：断点续传、Human-in-the-loop 挂起恢复、同一会话多轮对话
#   长期记忆：独立数据库，以 user_id 为键，跨 session 持久存储用户档案和历史
#
# 生产替换：将 SQLite 换成 PostgreSQL 只需修改此处的连接串，其余代码不变
#   from langgraph.checkpoint.postgres import PostgresSaver
#   CHECKPOINT_DB_URI = "postgresql://user:pass@host:5432/dbname"
# ============================================================
SHORT_TERM_MEMORY = {
    "db_path": str(BASE_DIR / "data" / "checkpoints.db"),
}

# ============================================================
# 长期记忆 (Long-term Memory) 配置
# 独立 SQLite 数据库，存储用户档案 (user_profile) 和历史工单
# ============================================================
LONG_TERM_MEMORY = {
    "db_path": str(BASE_DIR / "data" / "long_term_memory.db"),
}

# ============================================================
# 意图分类标签
# 集中定义，避免各处硬编码字符串
# ============================================================
INTENT_LABELS = {
    "REFUND": "refund",       # 退款/售后
    "TECH": "tech",           # 技术支持
    "COMPLAINT": "complaint", # 投诉
    "GENERAL": "general",     # 一般咨询
}
