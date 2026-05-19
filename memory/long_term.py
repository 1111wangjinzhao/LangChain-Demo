"""
memory/long_term.py — 长期记忆（用户档案 + 历史工单）

【什么是长期记忆？】
  长期记忆 = 跨 session、跨工单的用户画像和历史信息。
  它的生命周期：用户注册 → 永久保存（或按业务规则清理）。
  核心职责：
    1. 让 AI 知道"这个用户是谁"（姓名、偏好、VIP等级）
    2. 让 AI 知道"这个用户的历史问题"（避免重复询问）
    3. 支持个性化服务（老客户优先处理、针对性推荐）

【与短期记忆的关系】
  短期记忆（Checkpointer）：存"这次会话"的对话 State
  长期记忆（此模块）：存"这个用户"的档案和历史

【数据库设计】
  表1 user_profiles：用户基本档案
    - user_id      用户唯一标识
    - name         姓名
    - vip_level    VIP 等级 (0=普通, 1=银牌, 2=金牌)
    - total_orders 总订单数
    - notes        备注（客服手动填写的标签，如"脾气急躁"）
    - created_at   首次创建时间
    - updated_at   最后更新时间

  表2 interaction_history：历史工单记录
    - id           自增主键
    - user_id      关联用户
    - thread_id    关联短期记忆的 thread_id
    - intent       本次工单类型
    - summary      工单摘要（节省 LLM token，只存核心信息）
    - resolved     是否已解决
    - created_at   创建时间

【生产扩展方向】
  - 将 SQLite 换成 PostgreSQL / MySQL（修改连接逻辑即可）
  - 加入向量搜索：将历史摘要 embed 后存入 pgvector，支持语义相似检索
  - 加入 Redis 缓存层：热点用户档案缓存，降低 DB 查询压力
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional
from config import LONG_TERM_MEMORY

os.makedirs(os.path.dirname(LONG_TERM_MEMORY["db_path"]), exist_ok=True)


class LongTermMemory:
    """用户档案与历史工单的持久化操作类。"""

    def __init__(self, db_path: str = LONG_TERM_MEMORY["db_path"]):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 让查询结果支持按列名访问
        return conn

    def _init_db(self):
        """初始化数据库表结构（幂等，可重复执行）。"""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id     TEXT PRIMARY KEY,
                    name        TEXT NOT NULL DEFAULT '用户',
                    vip_level   INTEGER NOT NULL DEFAULT 0,
                    total_orders INTEGER NOT NULL DEFAULT 0,
                    notes       TEXT DEFAULT '',
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS interaction_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     TEXT NOT NULL,
                    thread_id   TEXT NOT NULL,
                    intent      TEXT NOT NULL,
                    summary     TEXT NOT NULL DEFAULT '',
                    resolved    INTEGER NOT NULL DEFAULT 0,
                    created_at  TEXT NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES user_profiles(user_id)
                );

                CREATE INDEX IF NOT EXISTS idx_history_user_id
                    ON interaction_history(user_id);
            """)

    # ----------------------------------------------------------------
    # 用户档案 CRUD
    # ----------------------------------------------------------------

    def get_or_create_user(self, user_id: str, name: str = "用户") -> dict:
        """获取用户档案，不存在则自动创建（首次建档）。"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
            ).fetchone()

            if row:
                return dict(row)

            now = datetime.now().isoformat()
            conn.execute(
                """INSERT INTO user_profiles
                   (user_id, name, vip_level, total_orders, notes, created_at, updated_at)
                   VALUES (?, ?, 0, 0, '', ?, ?)""",
                (user_id, name, now, now)
            )
            return {
                "user_id": user_id, "name": name,
                "vip_level": 0, "total_orders": 0,
                "notes": "", "created_at": now, "updated_at": now
            }

    def update_user_notes(self, user_id: str, notes: str):
        """更新客服备注（如：曾多次退款、VIP 优先处理）。"""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE user_profiles SET notes = ?, updated_at = ? WHERE user_id = ?",
                (notes, datetime.now().isoformat(), user_id)
            )

    # ----------------------------------------------------------------
    # 历史工单记录
    # ----------------------------------------------------------------

    def save_interaction(
        self,
        user_id: str,
        thread_id: str,
        intent: str,
        summary: str,
        resolved: bool = True
    ):
        """保存一条历史工单记录。在图执行结束时由 save_memory_node 调用。"""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO interaction_history
                   (user_id, thread_id, intent, summary, resolved, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, thread_id, intent, summary,
                 1 if resolved else 0, datetime.now().isoformat())
            )

    def get_recent_interactions(self, user_id: str, limit: int = 5) -> list[dict]:
        """获取用户最近 N 条历史工单，用于构建长期记忆上下文。"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT intent, summary, resolved, created_at
                   FROM interaction_history
                   WHERE user_id = ?
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (user_id, limit)
            ).fetchall()
            return [dict(row) for row in rows]

    # ----------------------------------------------------------------
    # 构建注入 LLM 的上下文字符串
    # ----------------------------------------------------------------

    def build_context_for_llm(self, user_id: str) -> str:
        """
        将用户的长期记忆格式化为自然语言字符串，注入给 LLM 作为 System 上下文。

        【为什么格式化成字符串而不是直接传字典？】
          LLM 理解自然语言比理解 JSON 结构更可靠。
          将结构化数据转换为自然语言描述，能显著提升 LLM 的上下文利用效果。
        """
        profile = self.get_or_create_user(user_id)
        recent = self.get_recent_interactions(user_id, limit=3)

        vip_map = {0: "普通用户", 1: "银牌会员", 2: "金牌会员"}
        vip_label = vip_map.get(profile.get("vip_level", 0), "普通用户")

        lines = [
            f"【用户档案】",
            f"  姓名：{profile['name']}",
            f"  会员等级：{vip_label}",
            f"  历史订单数：{profile['total_orders']} 单",
        ]
        if profile.get("notes"):
            lines.append(f"  客服备注：{profile['notes']}")

        if recent:
            lines.append("\n【近期历史工单】")
            for item in recent:
                status = "已解决" if item["resolved"] else "未解决"
                lines.append(f"  - [{item['intent']}] {item['summary']} ({status})")
        else:
            lines.append("\n【近期历史工单】暂无记录（新用户）")

        return "\n".join(lines)


# 模块级单例，全局共享一个实例
long_term_memory = LongTermMemory()
