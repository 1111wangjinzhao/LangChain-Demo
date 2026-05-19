"""
state.py — 全局状态定义 (数据总线)

【LangGraph 核心概念：State】
  State 是贯穿整个图所有节点的共享数据结构。
  每个节点接收当前 State，返回一个字典来"更新"State 中的特定字段。
  LangGraph 会将节点的返回值 merge 到 State 中，而不是整体替换。

【字段更新策略说明】
  - 普通字段（str, dict）：直接覆盖（新值替换旧值）
  - messages 字段：使用 add_messages reducer，执行追加操作（append）
    这是聊天记录的标准处理方式，保证历史消息不丢失

【混合架构：字段归属说明】
  Workflow 节点负责写入：long_term_context, resolution_summary
  Agent 节点负责写入：intent, next_agent, investigation_report, supervisor_reasoning
  Human-in-the-loop 通过 update_state 写入：human_approved, human_feedback
"""
from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class CustomerServiceState(TypedDict):
    # ----------------------------------------------------------
    # 核心对话记录
    # Annotated[list, add_messages] 是 LangGraph 提供的 reducer 机制：
    #   节点返回 {"messages": [new_msg]}，LangGraph 调用 add_messages
    #   将 new_msg 追加到列表末尾，而非覆盖。
    # ----------------------------------------------------------
    messages: Annotated[list, add_messages]

    # ----------------------------------------------------------
    # 会话元信息
    # ----------------------------------------------------------
    user_id: str          # 用户唯一标识，用于加载/保存长期记忆

    # ----------------------------------------------------------
    # [WORKFLOW] 长期记忆上下文（由 load_memory_node 填充）
    # 格式化为自然语言，注入给所有后续 LLM 节点作为背景
    # ----------------------------------------------------------
    long_term_context: Optional[str]

    # ----------------------------------------------------------
    # [AGENT - Supervisor] 调度器决策字段
    #   next_agent：Supervisor Agent 决定路由到哪个专家智能体
    #               取值：tech_agent / refund_agent / complaint_agent / general_agent
    #   intent：    意图标签，兼容路由逻辑
    #   supervisor_reasoning：Supervisor 的决策推理过程（可用于日志审计）
    # ----------------------------------------------------------
    next_agent: Optional[str]
    intent: Optional[str]
    supervisor_reasoning: Optional[str]

    # ----------------------------------------------------------
    # [AGENT - Refund Investigator] 退款调查报告
    #   由 refund_investigator_node（create_react_agent 子图）生成
    #   人工在 interrupt 时可以读取此报告，做出更有依据的审批决策
    # ----------------------------------------------------------
    investigation_report: Optional[str]

    # ----------------------------------------------------------
    # [Human-in-the-loop] 人工审批字段
    #   通过 app.update_state() 在图挂起时由外部写入
    # ----------------------------------------------------------
    human_approved: Optional[bool]   # True=同意, False=拒绝
    human_feedback: Optional[str]    # 人工补充的说明文字

    # ----------------------------------------------------------
    # [WORKFLOW] 最终处理结果摘要（由业务节点写入，供 save_memory_node 保存）
    # ----------------------------------------------------------
    resolution_summary: Optional[str]
