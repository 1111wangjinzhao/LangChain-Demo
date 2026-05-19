"""
nodes/memory_nodes.py — 记忆管理节点

这两个节点是"记忆生命周期"的边界节点：
  - load_memory_node：图的入口附近，负责加载长期记忆注入上下文
  - save_memory_node：图的出口附近，负责将本次交互结果写入长期记忆

【为什么把记忆操作封装成节点？】
  - 职责分离：业务节点（tech_support, refund）只关注业务逻辑，不关心记忆存取
  - 可观测性：记忆加载/保存是独立的步骤，便于日志追踪和调试
  - 灵活性：可以在图中精确控制记忆操作的时机（比如某些路径不需要保存）
"""
from state import CustomerServiceState
from memory.long_term import long_term_memory


def load_memory_node(state: CustomerServiceState) -> dict:
    """
    图的第一个节点：加载用户长期记忆。

    从长期记忆数据库中读取用户档案和历史工单，
    格式化成自然语言字符串，写入 State 的 long_term_context 字段。
    后续所有业务节点都可以从 State 中读取这个上下文，注入给 LLM。

    【执行时机】
      每次 invoke 时的第一步。无论用户问什么，先了解这个用户是谁。
    """
    user_id = state.get("user_id", "anonymous")
    print(f"\n📚 [记忆节点] 正在加载用户 {user_id} 的长期记忆...")

    context = long_term_memory.build_context_for_llm(user_id)
    print(f"   已加载长期上下文：\n{context}\n")

    return {"long_term_context": context}


def save_memory_node(state: CustomerServiceState) -> dict:
    """
    图的最后一个节点：保存本次交互到长期记忆。

    将本次工单的核心信息（意图、处理结果摘要）写入长期记忆数据库，
    供下次对话时加载，实现"越用越懂你"的效果。

    【执行时机】
      所有业务节点处理完成后，图结束前。
    """
    user_id = state.get("user_id", "anonymous")
    intent = state.get("intent", "general")
    summary = state.get("resolution_summary", "")

    if not summary:
        # 如果业务节点没有显式设置 resolution_summary，
        # 则从最后一条 AI 消息中提取前 100 个字符作为摘要
        messages = state.get("messages", [])
        ai_messages = [m for m in messages if hasattr(m, 'type') and m.type == 'ai']
        if ai_messages:
            summary = ai_messages[-1].content[:100] + "..."

    if summary:
        print(f"\n💾 [记忆节点] 正在保存本次交互记录（用户: {user_id}, 意图: {intent}）...")
        long_term_memory.save_interaction(
            user_id=user_id,
            thread_id=state.get("messages", [{}])[0].id if state.get("messages") else "unknown",
            intent=intent,
            summary=summary,
            resolved=True
        )
        print(f"   已保存摘要：{summary[:50]}...")

    return {}
