"""
graph.py — 混合架构图（Agent + Workflow）的组装与编译

【完整图结构 + 节点性质标注】

  START
    │
    ▼
  load_memory_node        [WORKFLOW] 确定性：始终执行，加载用户档案
    │
    ▼
  supervisor_node         [AGENT]    LLM 动态决策：分析问题，路由到专家
    │
    ▼ (条件边：读取 supervisor 的 next_agent 决策)
    │
    ├─► tech_support_node [AGENT]    LLM 驱动：ReAct 工具调用循环
    │     │ (条件边：检测 tool_calls)
    │     ├─► tool_node              [WORKFLOW] 确定性：执行工具，返回结果
    │     │      └─► tech_support_node（循环）
    │     └─► save_memory_node
    │
    ├─► refund_investigator_node  [AGENT]    create_react_agent 子图：自动调查
    │     │ (条件边：始终路由到 refund_executor)
    │     ▼
    │   refund_executor_node      [WORKFLOW] ⚠️ interrupt_before 挂起，等待人工
    │     └─► save_memory_node
    │
    ├─► complaint_node    [WORKFLOW] 简单对话生成，无动态决策
    │     └─► save_memory_node
    │
    └─► general_node      [WORKFLOW] 简单对话生成，无动态决策
          └─► save_memory_node
                │
                ▼
              END

【为什么 complaint/general 是 Workflow 而不是 Agent？】
  这两类场景的处理逻辑是固定的（生成一段回复），不需要 LLM 来决定"下一步做什么"。
  只有"需要多步推理、动态决策、工具调用或路由"的节点才值得做成 Agent。
  过度使用 Agent 会增加延迟和 token 消耗，不符合生产实践。

【interrupt_before 位置的变化】
  旧版：interrupt_before=["refund_node"]（在退款执行前暂停）
  新版：interrupt_before=["refund_executor_node"]（同样在执行前暂停，但此时
        State 中已有 investigation_report，人工可以基于报告做决策）
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from state import CustomerServiceState
from nodes import (
    load_memory_node,
    save_memory_node,
    supervisor_node,
    tech_support_node,
    refund_investigator_node,
    refund_executor_node,
    complaint_node,
    general_node,
)
from langgraph.prebuilt import tools_condition
from edge import route_by_supervisor, route_after_investigation
from tools.support_tools import SUPPORT_TOOLS
from memory.short_term import get_checkpointer


def build_graph():
    """
    构建混合架构图：Agent 负责决策，Workflow 负责执行。
    """
    builder = StateGraph(CustomerServiceState)

    # ----------------------------------------------------------------
    # 1. 注册节点
    # ----------------------------------------------------------------

    # [WORKFLOW] 记忆管理节点
    builder.add_node("load_memory_node", load_memory_node)
    builder.add_node("save_memory_node", save_memory_node)

    # [AGENT] Supervisor：多智能体调度器
    builder.add_node("supervisor_node", supervisor_node)

    # [AGENT] 技术支持：ReAct 图级循环（手动实现）
    builder.add_node("tech_support_node", tech_support_node)
    builder.add_node("tool_node", ToolNode(SUPPORT_TOOLS))  # [WORKFLOW] 执行工具

    # [AGENT] 退款调查：create_react_agent 子图
    builder.add_node("refund_investigator_node", refund_investigator_node)

    # [WORKFLOW] 退款执行：人工审批后执行退款
    builder.add_node("refund_executor_node", refund_executor_node)

    # [WORKFLOW] 其他业务节点
    builder.add_node("complaint_node", complaint_node)
    builder.add_node("general_node", general_node)

    # ----------------------------------------------------------------
    # 2. 连接边
    # ----------------------------------------------------------------

    # Workflow 入口：固定路径
    builder.add_edge(START, "load_memory_node")
    builder.add_edge("load_memory_node", "supervisor_node")

    # Agent 路由：Supervisor 决策驱动
    builder.add_conditional_edges(
        "supervisor_node",
        route_by_supervisor,
        {
            "tech_support_node": "tech_support_node",
            "refund_investigator_node": "refund_investigator_node",
            "complaint_node": "complaint_node",
            "general_node": "general_node",
        }
    )

    # Tech Support ReAct 循环
    # tools_condition 是 langgraph.prebuilt 提供的官方标准路由函数：
    #   返回 "tools"    → 最后一条消息含 tool_calls，需要执行工具
    #   返回 "__end__"  → 无工具调用，流程结束
    # 通过 path_map 将官方返回值映射到本图中的实际节点名
    builder.add_conditional_edges(
        "tech_support_node",
        tools_condition,
        {
            "tools": "tool_node",       # "tools" 是 tools_condition 的固定返回值
            "__end__": "save_memory_node",
        }
    )
    builder.add_edge("tool_node", "tech_support_node")

    # 退款流程：调查 → (interrupt) → 执行
    builder.add_conditional_edges(
        "refund_investigator_node",
        route_after_investigation,
        {"refund_executor_node": "refund_executor_node"}
    )
    builder.add_edge("refund_executor_node", "save_memory_node")

    # 其他业务节点 → 保存记忆
    builder.add_edge("complaint_node", "save_memory_node")
    builder.add_edge("general_node", "save_memory_node")

    # 出口
    builder.add_edge("save_memory_node", END)

    # ----------------------------------------------------------------
    # 3. 编译图
    # ----------------------------------------------------------------
    checkpointer = get_checkpointer()

    app = builder.compile(
        checkpointer=checkpointer,
        # 在进入 refund_executor_node 前挂起
        # 此时 State 中已有 investigation_report，人工可以基于报告决策
        interrupt_before=["refund_executor_node"],
    )

    return app, checkpointer


_app = None
_checkpointer = None


def get_app():
    """获取全局 app 实例（单例模式）。"""
    global _app, _checkpointer
    if _app is None:
        _app, _checkpointer = build_graph()
    return _app
