"""
nodes/support_nodes.py — 业务处理节点

包含：
  - tech_support_node：技术支持（含 ReAct 工具调用循环，由图层面实现）
  - refund_node：退款处理（人工审批后执行）
  - complaint_node：投诉处理
  - general_node：一般咨询

【tech_support_node 与 ToolNode 的配合】
  tech_support_node 本身只调用 chat_model（已绑定工具）。
  如果模型决定调用工具，它输出的 AIMessage.tool_calls 不为空。
  Graph 中的条件边 should_use_tools 检测到 tool_calls，
  就把流程导向 ToolNode，ToolNode 执行工具，将结果作为 ToolMessage 追加。
  然后再次回到 tech_support_node，形成循环，直到模型不再调用工具为止。

  这个"节点 → 工具 → 节点"的循环，就是经典的 ReAct 模式：
    Reason（节点思考）→ Act（工具执行）→ Observe（读取结果）→ Reason ...
"""
from langchain_core.messages import SystemMessage, AIMessage
from state import CustomerServiceState
from model import chat_model, base_model
from memory.long_term import long_term_memory


def tech_support_node(state: CustomerServiceState) -> dict:
    """
    技术支持节点。

    使用绑定了工具的 chat_model，让 LLM 自主决定是否需要查询知识库。
    如果 LLM 输出中包含 tool_calls，图会自动路由到 ToolNode 执行工具，
    执行完后再回到本节点继续生成回复（ReAct 循环）。
    """
    print("👨‍💻 [技术节点] 技术专家正在处理中...")

    long_term_context = state.get("long_term_context", "")

    system_prompt = (
        "你是专业的技术支持工程师，耐心、专业、简洁。\n"
        "如果需要查阅知识库，请使用 search_faq 工具。\n"
        "如果用户提到了订单问题，请使用 check_order_status 工具。\n"
        "不要输出任何 <think> 标签或内部思考过程。\n\n"
        f"用户背景：\n{long_term_context}"
    )

    response = chat_model.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])

    # 生成处理摘要，供 save_memory_node 存入长期记忆
    summary = f"技术咨询：{state['messages'][-1].content[:50]}"
    return {
        "messages": [response],
        "resolution_summary": summary
    }


def refund_executor_node(state: CustomerServiceState) -> dict:
    """
    退款执行节点 [WORKFLOW]。

    这是混合架构中纯 Workflow 的典型代表：
      - 不做任何推理，只是"执行"人工已经批准的操作
      - 读取 human_approved / human_feedback（人工通过 update_state 写入）
      - 还可以读取 investigation_report（退款调查 Agent 生成的报告）

    执行时机：
      interrupt_before=["refund_executor_node"] 挂起后，
      人工查看 investigation_report，做出决策，
      通过 update_state 写入审批结果，
      再调用 app.invoke(None, config) 恢复，进入本节点。
    """
    print("💰 [退款执行节点] 财务专员正在处理退款...")

    investigation_report = state.get("investigation_report", "")
    if investigation_report:
        print(f"\n   [参考调查报告]\n{investigation_report}\n")

    human_approved = state.get("human_approved")
    human_feedback = state.get("human_feedback", "")

    if human_approved is True:
        content = (
            "您好！经主管审核，您的退款申请已批准。\n"
            "退款金额将在 1-3 个工作日内原路退回您的支付账户。\n"
            "如有疑问请保留此工单号联系我们。"
        )
        if human_feedback:
            content += f"\n\n客服备注：{human_feedback}"
        summary = "退款申请：人工审核通过，已发起退款"
    elif human_approved is False:
        content = (
            "您好，经核实，您的退款申请暂时无法通过。\n"
            f"原因：{human_feedback or '不符合退款政策条件'}\n"
            "如有异议，请提供更多证明材料，我们将重新评估。"
        )
        summary = f"退款申请：人工审核拒绝，原因：{human_feedback}"
    else:
        # 未经人工审批直接到达（理论上不应发生，作为兜底）
        content = "您的退款请求已收到，正在等待人工审核，预计24小时内处理完毕。"
        summary = "退款申请：等待人工审核"

    msg = AIMessage(content=content)
    return {
        "messages": [msg],
        "resolution_summary": summary
    }


def complaint_node(state: CustomerServiceState) -> dict:
    """投诉处理节点：表达重视、记录问题、给出解决承诺。"""
    print("📋 [投诉节点] 客服主管正在处理投诉...")

    long_term_context = state.get("long_term_context", "")
    system_prompt = (
        "你是客服主管，负责处理用户投诉。态度诚恳，先道歉认错，再给出具体解决方案和时间承诺。\n"
        "绝对不要推卸责任，不要使用套话。\n\n"
        f"用户背景：\n{long_term_context}"
    )

    response = base_model.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])

    summary = f"投诉处理：{state['messages'][-1].content[:50]}"
    return {
        "messages": [response],
        "resolution_summary": summary
    }


def general_node(state: CustomerServiceState) -> dict:
    """一般咨询节点：处理其他类型的问题。"""
    print("💬 [咨询节点] 客服助手正在回复...")

    long_term_context = state.get("long_term_context", "")
    system_prompt = (
        "你是友好的客服助手，回答用户的一般性问题。简洁、准确、有礼貌。\n\n"
        f"用户背景：\n{long_term_context}"
    )

    response = base_model.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"]
    ])

    summary = f"一般咨询：{state['messages'][-1].content[:50]}"
    return {
        "messages": [response],
        "resolution_summary": summary
    }
