"""
model.py — LLM 实例与结构化输出模型定义

【三种 LLM 实例的职责划分】

  base_model      ← 基础模型，无工具绑定，用于对话生成（complaint/general节点）
  chat_model      ← 绑定了客服工具（search_faq, check_order_status），用于 tech_support ReAct 循环
  supervisor_model← 绑定了结构化输出（SupervisorDecision），用于 Supervisor Agent 决策路由

【with_structured_output vs PydanticOutputParser】

  PydanticOutputParser（旧方式，analyzer_node 曾用）：
    prompt 中注入 format_instructions → 模型输出 JSON 字符串 → parser 解析
    缺点：依赖模型严格遵守格式，有解析失败风险

  with_structured_output（新方式，supervisor_node 使用）：
    使用 LLM 的 function calling / tool calling 底层能力，强制输出结构化数据
    优点：由 API 层保证格式，几乎不会解析失败，生产环境更可靠

【为什么 Supervisor 用 with_structured_output 而不是工具路由？】
  LangGraph 官方的 Supervisor 模式有两种实现：
    方式A：with_structured_output → 模型输出 {next_agent: "..."} → 路由函数读取
    方式B：bind_tools 将每个 Agent 当作"工具" → 模型决定调用哪个工具
  方式A 更简洁清晰，适合学习；方式B 更灵活，适合复杂的多智能体场景。
  本项目选用方式A。
"""
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from config import LLM_CONFIG
from tools.support_tools import SUPPORT_TOOLS

# ============================================================
# 底层 LLM 实例（所有 model 实例的基础）
# ============================================================
_llm = ChatOpenAI(
    model=LLM_CONFIG["model"],
    api_key=LLM_CONFIG["api_key"],
    base_url=LLM_CONFIG["base_url"],
    temperature=LLM_CONFIG["temperature"],
    extra_body={"reasoning_split": True},
)

# ============================================================
# base_model：纯对话，无工具绑定
# 用于：complaint_node, general_node, 文本生成场景
# ============================================================
base_model = _llm

# ============================================================
# chat_model：绑定全套客服工具
# 用于：tech_support_node 的 ReAct 工具调用循环
# ============================================================
chat_model = _llm.bind_tools(SUPPORT_TOOLS)

# ============================================================
# Supervisor Agent 的决策结构
# ============================================================
class SupervisorDecision(BaseModel):
    """
    Supervisor Agent 的路由决策。
    with_structured_output 会将此 Pydantic 模型转换为 function calling schema，
    由 LLM API 层保证输出格式合法，比 prompt 中注入 format_instructions 更可靠。
    """
    next_agent: Literal["tech_agent", "refund_agent", "complaint_agent", "general_agent"] = Field(
        description="下一个应该处理此问题的专家智能体名称"
    )
    intent: Literal["tech", "refund", "complaint", "general"] = Field(
        description="用户的核心意图标签"
    )
    reasoning: str = Field(
        description="做出此路由决策的简要推理（1-2句话）"
    )

# ============================================================
# supervisor_model：绑定结构化输出，专用于 Supervisor Agent
# ============================================================
supervisor_model = _llm.with_structured_output(SupervisorDecision)

# ============================================================
# 旧版兼容：IntentRouter（保留，nodes/analyzer_node.py 有引用）
# ============================================================
class IntentRouter(BaseModel):
    intent: str = Field(
        description="用户意图，必须且只能是以下之一："
                    "'refund'（退款/售后），"
                    "'tech'（技术咨询/使用问题），"
                    "'complaint'（投诉/不满），"
                    "'general'（其他一般咨询）"
    )
    confidence: str = Field(
        description="分类置信度：'high'（确定）或 'low'（不确定）"
    )
