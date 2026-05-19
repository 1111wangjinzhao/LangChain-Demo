"""
memory/short_term.py — 短期记忆（会话级持久化）

【什么是短期记忆？】
  短期记忆 = 单次"会话/工单"的完整对话状态。
  它的生命周期是：工单创建 → 工单关闭。
  核心职责：
    1. 支持多轮对话（同一 thread_id 内，每次 invoke 都能看到之前的消息）
    2. 支持 Human-in-the-loop（图执行到中断点时，State 被序列化存档，
       人工审批后调用 app.invoke(None, config) 从断点恢复）
    3. 支持断点续传（进程重启后，State 依然在数据库中）

【Checkpointer 工作原理】
  LangGraph 把 Checkpointer 理解为"存档点"：
    - 每执行完一个节点，自动保存一次 State 快照到数据库
    - 键值：(thread_id, checkpoint_id)
    - 恢复时：通过 thread_id 找到最新的 checkpoint，还原 State

【与 MemorySaver 的区别】
  MemorySaver：存在内存字典中，进程退出即消失
  SqliteSaver：存在磁盘 SQLite 文件，进程重启依然存在
  PostgresSaver：生产级，支持多实例并发、连接池（换一行导入即可）

【如何换成 PostgreSQL（生产环境）】
  from langgraph.checkpoint.postgres import PostgresSaver
  import psycopg
  conn = psycopg.connect("postgresql://user:pass@host:5432/dbname")
  checkpointer = PostgresSaver(conn)
  checkpointer.setup()  # 建表，只需执行一次
"""
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from config import SHORT_TERM_MEMORY

# 确保数据目录存在
os.makedirs(os.path.dirname(SHORT_TERM_MEMORY["db_path"]), exist_ok=True)


def get_checkpointer() -> SqliteSaver:
    """
    获取 SQLite Checkpointer 实例。

    【使用方式】
      SqliteSaver 是一个上下文管理器，推荐用 with 语句管理连接生命周期：

        with get_checkpointer() as cp:
            app = graph.compile(checkpointer=cp)
            app.invoke(...)

      或者在应用启动时创建，在应用关闭时手动调用 .close()。

    【SQLite 的局限性】
      - 不支持多进程/多线程并发写入（单文件锁）
      - 适合：本地开发、单实例小流量服务
      - 不适合：高并发生产环境（请换 PostgresSaver）
    """
    return SqliteSaver.from_conn_string(SHORT_TERM_MEMORY["db_path"])
