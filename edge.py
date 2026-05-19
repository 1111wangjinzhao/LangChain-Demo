"""
edge.py — 图的边（路由逻辑）

【LangGraph 的两种边】

1. 普通边 add_edge(from, to)
   无条件连接，A 执行完后必然去 B。
   体现 Workflow 思维：确定性、可预测。

2. 条件边 add_conditional_edges(from, routing_fn, path_map)
   routing_fn 读取 State，返回字符串决定走哪条路。
   路由函数本身是 Workflow 代码（if/else），
   但其读取的 State 字段可能是由 Agent 写入的（如 next_agent）。

【本文件路由函数说明】

  route_by_supervisor：
    读取 Supervisor Agent 写入的 next_agent 字段进行路由。
    → 路由函数是 Workflow，但决策来源于 Agent（混合架构的典型模式）

  tools_condition（来自 langgraph.prebuilt）：
    LangGraph 官方标准工具调用检测函数，替代手写的 should_use_tools。
    返回 "tools"（有工具调用）或 "__end__"（无工具调用）。
    配合 path_map 使用：{"tools": "tool_node", "__end__": "save_memory_node"}

    【为什么用官方 tools_condition 而不是自己写？】
    官方实现更健壮，能正确处理：
      - AIMessage.tool_calls 非空
      - 流式输出中的 tool_calls 拼接
      - 未来版本中 tool_calls 格式变化（官方维护，自动跟进）

  route_after_investigation：
    退款调查完成后的路由，固定导向 refund_executor_node。
"""
from state import CustomerServiceState

# 从 langgraph.prebuilt 导入官方标准工具调用条件路由函数
# tools_condition：检测 state["messages"] 最后一条消息是否包含 tool_calls
#   返回 "tools" → 有工具调用，需要执行工具
#   返回 "__end__" → 无工具调用，流程结束（通过 path_map 可以映射到自定义节点名）
from langgraph.prebuilt import tools_condition  # noqa: F401 （在 graph.py 中通过 path_map 使用）


def route_by_supervisor(state: CustomerServiceState) -> str:
    """
    读取 Supervisor Agent 的决策结果，将流程路由到对应专家节点。

    next_agent 字段由 supervisor_node 写入，取值范围：
      tech_agent / refund_agent / complaint_agent / general_agent
    """
    next_agent = state.get("next_agent", "general_agent")

    route_map = {
        "tech_agent": "tech_support_node",
        "refund_agent": "refund_investigator_node",
        "complaint_agent": "complaint_node",
        "general_agent": "general_node",
    }

    next_node = route_map.get(next_agent, "general_node")
    print(f"🔀 [路由] Supervisor 决策={next_agent} → 节点={next_node}")
    return next_node


def route_after_investigation(state: CustomerServiceState) -> str:
    """
    退款调查完成后的路由：固定导向 refund_executor_node。

    这是 interrupt_before=["refund_executor_node"] 的前置节点，
    图会在"即将进入 refund_executor_node"时挂起，
    人工可查看 investigation_report 做审批决策。

    可扩展：基于调查结论自动判断
      如报告结论是"建议拒绝" → 可提示人工，或自动路由到 complaint_node
    """
    return "refund_executor_node"
