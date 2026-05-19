"""
nodes/refund_investigator_node.py — 退款调查 Agent

【版本迁移说明】
  LangGraph 1.0 将 create_react_agent 标记为 @deprecated，
  官方将 Agent 构建能力迁移到 langchain 包，统一入口为：
    from langchain.agents import create_agent

  迁移对照：
    旧（已废弃）: from langgraph.prebuilt import create_react_agent
                  agent = create_react_agent(model, tools, prompt=...)
    新（当前标准）: from langchain.agents import create_agent
                   agent = create_agent(model, tools, system_prompt=...)

  参数变化：
    - prompt       → system_prompt（现在只接受 str 或 SystemMessage）
    - 其余参数基本一致（tools, checkpointer, state_schema 等）

【为什么退款流程需要专门的调查 Agent？】

  旧版流程（纯 Workflow）：
    用户说"退款" → 立即挂起 → 人工看原始聊天记录 → 自己去查订单系统
    问题：人工工作量大，信息分散，决策效率低

  新版流程（Agent + Workflow 混合）：
    用户说"退款" → [Agent 自动调查] 调用工具查订单、核对政策、写报告
                 → 挂起，展示结构化调查报告给人工
                 → 人工基于报告一键审批（同意/拒绝）
    效果：人工只需 10 秒看报告做决策，无需手动查系统

【create_agent 的内部结构】

  create_agent 返回的是一个完整的 CompiledStateGraph，内部节点为：
    __start__ → model（LLM 节点）→ tools（ToolNode）→ model → ... → __end__
  它实现了标准的 ReAct 循环：
    LLM 判断是否需要工具 → 调用工具 → LLM 读取结果 → 继续/结束

【何时用 create_agent vs 手动 ReAct 图？】

  create_agent（适合）：
    - 快速创建标准 ReAct agent，工具列表固定，逻辑简单
    - 作为子任务嵌入更大的图（本节点的用法）
    - 不需要在循环中插入自定义逻辑

  手动 ReAct 图（tech_support_node 的方式，适合）：
    - 需要在工具调用的每一步添加自定义逻辑（日志、条件判断）
    - 需要 interrupt_after（工具调用后暂停让人工确认结果）
    - 需要精细控制循环退出条件
"""
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from state import CustomerServiceState
from model import base_model
from tools.support_tools import check_order_status

# ============================================================
# 构建退款调查子 Agent
#
# system_prompt：注入给 Agent 的系统提示，描述调查任务和报告格式
# tools：Agent 可用的工具列表（create_agent 内部会自动 bind_tools）
# ============================================================
_INVESTIGATOR_SYSTEM = """你是退款调查专员，负责在人工审核前自动核实退款申请的合理性。

【你的调查任务】
1. 从对话中提取用户提到的订单号（如 ORDER-XXXX）
2. 调用 check_order_status 工具查询订单详情
3. 基于查询结果和退款政策，生成一份结构化的调查报告

【调查报告格式】
---退款调查报告---
申请用户：[从对话中提取]
申请订单：[订单号]
订单状态：[查询结果]
退款政策：[是否在退款期内]
调查结论：[支持退款/建议拒绝/需人工核查]
建议处理：[具体建议，1-2句话]
---报告结束---

如果对话中没有明确订单号，请在报告中注明"用户未提供订单号，建议人工核实"。
只输出调查报告，不要输出其他内容。"""

_investigation_agent = create_agent(
    model=base_model,
    tools=[check_order_status],
    system_prompt=_INVESTIGATOR_SYSTEM,
)


def refund_investigator_node(state: CustomerServiceState) -> dict:
    """
    退款调查 Agent 节点。

    调用 create_agent 构建的子 Agent，自动完成：
      1. 分析对话，提取订单号
      2. 调用 check_order_status 工具查询订单信息
      3. 生成结构化调查报告

    报告写入 State["investigation_report"]，
    人工在 interrupt 挂起时可读取此报告，做出有据可查的审批决定。
    """
    print("🔍 [退款调查 Agent] 正在自动调查退款申请...")

    # 将完整对话历史传入子 Agent，让它自行分析
    # create_agent 构建的图以 {"messages": [...]} 作为输入
    investigation_input = {"messages": list(state["messages"])}

    try:
        result = _investigation_agent.invoke(investigation_input)

        # 从子 Agent 的输出消息中提取最终报告
        # create_agent 返回的 messages 包含完整的 ReAct 过程（LLM + Tool 消息交替）
        # 我们只取最后一条有实际文本内容的 AI 消息
        final_messages = result.get("messages", [])
        report = ""
        for msg in reversed(final_messages):
            if (hasattr(msg, 'type') and msg.type == 'ai'
                    and msg.content
                    and not getattr(msg, 'tool_calls', None)):
                report = msg.content
                break

        if not report:
            report = "调查 Agent 未能生成报告，请人工核实。"

        print(f"\n📋 [调查报告]\n{report}\n")
        return {"investigation_report": report}

    except Exception as e:
        print(f"⚠️  退款调查 Agent 失败: {e}")
        return {
            "investigation_report": (
                f"---退款调查报告---\n"
                f"自动调查失败，原因：{e}\n"
                f"建议：人工直接核实对话记录和订单信息\n"
                f"---报告结束---"
            )
        }
