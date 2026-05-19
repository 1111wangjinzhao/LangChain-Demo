"""
tools/support_tools.py — 客服工具集

【LangGraph 中的 Tool Calling 架构】

  传统 LangChain：Chain 调用工具
  LangGraph Tool Calling：

    [LLM 节点]  ←→  [ToolNode]
         ↕                ↕
    决定调用哪个工具    实际执行工具函数

  工作流程：
    1. LLM 节点调用 model.bind_tools(tools) 绑定工具列表
    2. LLM 输出 AIMessage，其中包含 tool_calls 字段（工具调用请求）
    3. 图通过条件边判断：如果有 tool_calls → 去 ToolNode；否则 → 结束
    4. ToolNode 执行工具，返回 ToolMessage（结果）追加到 messages
    5. 回到 LLM 节点，LLM 读取工具结果，生成最终回复

  这是一个"ReAct 循环"（Reasoning + Acting）：
    Think → Act（调用工具）→ Observe（获取结果）→ Think → 输出

【@tool 装饰器】
  LangChain 的 @tool 装饰器会自动：
    1. 将函数名变成工具名
    2. 将函数的 docstring 变成工具描述（LLM 通过它决定何时调用）
    3. 将函数参数类型提示转换为 JSON Schema（LLM 用它填写参数）
  所以：docstring 要写清楚！参数类型提示要准确！
"""
from langchain_core.tools import tool


# ================================================================
# 工具 1：FAQ 知识库搜索
# 模拟从知识库/向量数据库中检索常见问题答案
# 生产环境：对接 Elasticsearch 或 ChromaDB 等向量检索引擎
# ================================================================
@tool
def search_faq(query: str) -> str:
    """
    搜索产品知识库，获取常见问题的标准解答。
    适用于技术咨询、使用指南、功能说明等场景。
    参数 query 是用户的核心问题关键词。
    """
    # 模拟知识库（生产环境替换为真实的向量检索）
    knowledge_base = {
        "api并发": "我们的 API 默认支持每秒 100 次并发请求（QPS=100）。"
                   "企业版可申请提升至 1000 QPS。限流超出后返回 429 错误码，"
                   "建议客户端实现指数退避重试策略。",
        "限流": "API 限流策略：免费版 10 QPS，标准版 100 QPS，企业版 1000 QPS。"
                "超出限制会返回 HTTP 429，响应头 Retry-After 字段指示重试等待秒数。",
        "sdk安装": "安装命令：pip install our-sdk。"
                   "支持 Python 3.8+，详细文档见 https://docs.example.com/sdk。",
        "webhook配置": "在控制台 → 设置 → Webhook 中填写回调 URL。"
                       "支持 HTTP/HTTPS，我们会在事件触发时发送 POST 请求。"
                       "建议验证请求头中的 X-Signature 字段以确认来源合法性。",
        "登录失败": "常见原因：1) 密码错误（区分大小写）；2) 账号被锁定（连续5次失败后锁定10分钟）；"
                    "3) IP 限制。如持续无法登录，请联系安全团队重置账号。",
    }

    query_lower = query.lower()
    for key, answer in knowledge_base.items():
        if key in query_lower:
            return f"[知识库检索结果]\n{answer}"

    return "[知识库检索结果]\n未找到完全匹配的条目，建议人工查阅完整文档或上报技术团队。"


# ================================================================
# 工具 2：订单状态查询
# 模拟查询订单系统，用于退款前核实订单信息
# 生产环境：对接 ERP/OMS 系统 API
# ================================================================
@tool
def check_order_status(order_id: str) -> str:
    """
    查询指定订单的状态、金额和付款信息。
    在处理退款请求时，需要先调用此工具核实订单是否存在及其当前状态。
    参数 order_id 是用户提供的订单编号。
    """
    # 模拟订单数据库
    orders = {
        "ORDER-2026-001": {
            "status": "已完成",
            "amount": 299.00,
            "product": "专业版订阅（年付）",
            "paid_at": "2026-01-15",
            "refundable": True,
            "refund_deadline": "2026-04-15",
        },
        "ORDER-2026-002": {
            "status": "已完成",
            "amount": 99.00,
            "product": "标准版订阅（月付）",
            "paid_at": "2026-04-01",
            "refundable": True,
            "refund_deadline": "2026-05-01",
        },
        "ORDER-2026-003": {
            "status": "退款中",
            "amount": 199.00,
            "product": "企业版（月付）",
            "paid_at": "2026-03-10",
            "refundable": False,
            "refund_deadline": "已过期",
        },
    }

    order = orders.get(order_id.upper())
    if not order:
        return f"[订单查询结果]\n订单 {order_id} 不存在，请核对订单号是否正确。"

    refund_info = "支持退款" if order["refundable"] else f"不支持退款（退款截止日：{order['refund_deadline']}）"
    return (
        f"[订单查询结果]\n"
        f"订单号：{order_id}\n"
        f"商品：{order['product']}\n"
        f"金额：¥{order['amount']:.2f}\n"
        f"状态：{order['status']}\n"
        f"付款日期：{order['paid_at']}\n"
        f"退款政策：{refund_info}"
    )


# ================================================================
# 工具列表：导出给 LLM 绑定
# ================================================================
SUPPORT_TOOLS = [search_faq, check_order_status]
