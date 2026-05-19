"""
nodes/analyzer_node.py — 意图分析节点

职责：读取用户最新消息，判断意图类型，写入 State["intent"]。
这是路由决策的核心依据，影响后续走哪条业务路径。

【LCEL 链式调用回顾】
  prompt | model | parser 是 LangChain Expression Language (LCEL) 的链式写法：
  1. prompt.format_messages(...)  生成格式化的消息列表
  2. model.invoke(messages)       调用 LLM，得到 AIMessage
  3. parser.parse(ai_message)     解析 AIMessage 内容，转换为 Pydantic 对象

【PydanticOutputParser 的工作原理】
  parser.get_format_instructions() 会生成类似这样的指令注入到 Prompt：
    "请输出符合以下 JSON Schema 的内容：{...schema...}"
  模型输出 JSON 字符串后，parser 自动 parse 成 IntentRouter 对象。
  好处：强类型，字段有保障，不会因为模型输出格式错乱导致后续崩溃。
"""
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from state import CustomerServiceState
from model import base_model, IntentRouter

_parser = PydanticOutputParser(pydantic_object=IntentRouter)


def analyzer_node(state: CustomerServiceState) -> dict:
    """
    意图分析节点：判断用户意图，返回路由标签。

    同时读取 long_term_context，让模型在分类时考虑用户背景
    （例如：VIP 用户投诉，可以直接归类为高优先级 complaint）。
    """
    print("🤖 [分析节点] 正在识别用户意图...")

    user_message = state["messages"][-1].content
    long_term_context = state.get("long_term_context", "暂无用户历史记录")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是后端意图分类引擎，只输出 JSON，不输出任何解释文字。\n"
         "用户背景信息（供参考，不要输出）：\n{long_term_context}\n\n"
         "严格按以下格式输出：\n{format_instructions}"),
        ("human", "用户说：{user_message}")
    ])

    chain = prompt | base_model | _parser

    try:
        result = chain.invoke({
            "user_message": user_message,
            "long_term_context": long_term_context,
            "format_instructions": _parser.get_format_instructions()
        })
        print(f"   -> 意图: {result.intent} (置信度: {result.confidence})")
        return {"intent": result.intent}

    except Exception as e:
        print(f"⚠️  意图解析失败，触发兜底逻辑: {e}")
        return {"intent": "general"}
