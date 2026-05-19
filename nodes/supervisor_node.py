"""
nodes/supervisor_node.py — Supervisor Agent（多智能体调度器）

【Agent vs Workflow 的本质区别】

  Workflow（工作流）：
    "下一步执行哪个节点"由开发者在代码中写死（if intent == 'refund': ...）
    确定性强，路径可预测，适合简单/固定的业务流程

  Agent（智能体）：
    "下一步做什么"由 LLM 在运行时动态决策
    灵活性强，能处理歧义和复杂情况，适合需要推理判断的场景

【Supervisor Agent 的价值】

  对比旧版 analyzer_node（Workflow 思维）：
    只做简单意图分类，返回一个标签字符串
    无法处理：歧义意图、多意图混合、需要上下文推理的场景

  新版 supervisor_node（Agent 思维）：
    1. 读取完整对话历史和用户档案
    2. 综合推理，考虑多种因素（VIP 等级、历史投诉记录、措辞情绪等）
    3. 通过 with_structured_output 输出决策，包含路由目标和推理过程
    4. 推理过程写入 State，可用于日志审计和后续节点参考

【LangGraph 官方 Multi-Agent Supervisor 模式】

  本节点实现的是 LangGraph 推荐的 Supervisor 模式：
  一个中央 Agent 负责协调和路由，多个专家 Agent 负责执行。

  参考架构：
    Supervisor → 决定路由 → 专家 Agent A（Tech）
                           → 专家 Agent B（Refund Investigation）
                           → 专家 Agent C（Complaint）
                           → 专家 Agent D（General）
"""
from langchain_core.messages import SystemMessage
from state import CustomerServiceState
from model import supervisor_model


# Supervisor 的系统提示：描述它的职责和可用的专家团队
_SUPERVISOR_SYSTEM = """你是一个智能客服调度中心的 AI 主管。
你的任务是分析用户的问题，决定将其分配给哪位专家处理。

【你的专家团队】
- tech_agent（技术支持专家）：处理 API 使用、SDK 安装、功能咨询、登录问题等技术性问题
- refund_agent（退款调查专家）：处理退款申请、售后投诉中涉及退款的请求
- complaint_agent（客诉处理专家）：处理对产品/服务的不满、投诉、差评，但不涉及退款
- general_agent（通用咨询助手）：处理其他一般性问题、产品咨询、功能介绍等

【决策参考因素】
1. 用户的核心诉求（最重要）
2. 用户历史档案（VIP 用户的投诉优先路由到 complaint_agent）
3. 消息的情绪倾向（强烈不满 → complaint；明确退款诉求 → refund）
4. 如果问题涉及退款，即使包含投诉情绪，也优先路由到 refund_agent

【用户背景信息】
{long_term_context}"""


def supervisor_node(state: CustomerServiceState) -> dict:
    """
    Supervisor Agent 节点。

    这是整个图的"大脑"：它不执行具体业务，只负责推理和路由。
    通过 with_structured_output 保证输出格式可靠，避免 JSON 解析失败。

    【与旧版 analyzer_node 的对比】
      旧版：prompt 注入 format_instructions → 解析 JSON 字符串（脆弱）
      新版：with_structured_output → function calling 层面的强制结构化（健壮）
    """
    print("🧠 [Supervisor Agent] 正在分析用户需求并进行路由决策...")

    long_term_context = state.get("long_term_context", "暂无用户历史记录")
    system_prompt = _SUPERVISOR_SYSTEM.format(long_term_context=long_term_context)

    try:
        # with_structured_output 使用 LLM 的 function calling 能力
        # 传入的消息列表包含：系统提示 + 完整对话历史
        # LLM 会综合分析后输出符合 SupervisorDecision schema 的结构
        decision = supervisor_model.invoke([
            SystemMessage(content=system_prompt),
            *state["messages"]
        ])

        print(f"   -> 路由目标: {decision.next_agent}")
        print(f"   -> 意图标签: {decision.intent}")
        print(f"   -> 决策推理: {decision.reasoning}")

        return {
            "next_agent": decision.next_agent,
            "intent": decision.intent,
            "supervisor_reasoning": decision.reasoning,
        }

    except Exception as e:
        print(f"⚠️  Supervisor 决策失败，使用兜底路由: {e}")
        return {
            "next_agent": "general_agent",
            "intent": "general",
            "supervisor_reasoning": f"决策失败，兜底路由到通用助手: {e}",
        }
