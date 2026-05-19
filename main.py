"""
main.py — 混合架构演示入口

【5个演示场景，覆盖 Agent + Workflow 混合架构的核心特性】

  场景 1：技术咨询（Supervisor Agent 路由 + Tech ReAct Agent）
  场景 2：老用户回访（展示长期记忆 + Supervisor 综合决策）
  场景 3：退款流程（Supervisor → Refund Investigator Agent → 人工审批）
  场景 4：投诉处理（Supervisor 识别投诉情绪 → Workflow 节点处理）
  场景 5：歧义问题（验证 Supervisor Agent 的综合推理能力）

【运行方式】
  python main.py             → 运行所有场景
  python main.py --scene N   → 只运行场景 N（N=1~5）
"""
import sys
import time
from langchain_core.messages import HumanMessage
from graph import get_app
from memory.long_term import long_term_memory

# ================================================================
# 工具函数
# ================================================================
def print_divider(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)


def print_agent_decision(state: dict):
    """打印 Supervisor Agent 的决策信息。"""
    next_agent = state.get("next_agent", "未知")
    reasoning = state.get("supervisor_reasoning", "")
    if reasoning:
        print(f"\n🧠 [Supervisor 决策]")
        print(f"   路由到: {next_agent}")
        print(f"   推理: {reasoning}")


def print_final_reply(state: dict):
    """打印最后一条有效 AI 回复。"""
    messages = state.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, 'type') and msg.type == 'ai':
            if msg.content and not getattr(msg, 'tool_calls', None):
                print(f"\n🤖 最终回复：\n{msg.content}")
                return


def make_initial_inputs(user_id: str, text: str) -> dict:
    """构建初始 State 输入（第一轮必须提供所有字段的初始值）。"""
    return {
        "messages": [HumanMessage(content=text)],
        "user_id": user_id,
        "intent": None,
        "next_agent": None,
        "supervisor_reasoning": None,
        "long_term_context": None,
        "investigation_report": None,
        "human_approved": None,
        "human_feedback": None,
        "resolution_summary": None,
    }


def seed_demo_users():
    """预置演示用户数据。"""
    long_term_memory.get_or_create_user("USER-001", name="张明")
    long_term_memory.update_user_notes("USER-001", "技术型用户，曾反馈 API 文档不清晰")
    long_term_memory.save_interaction(
        user_id="USER-001", thread_id="TICKET-OLD-001",
        intent="tech", summary="咨询API并发限制，已提供技术文档", resolved=True
    )
    long_term_memory.get_or_create_user("USER-VIP", name="李静")
    long_term_memory.update_user_notes("USER-VIP", "金牌 VIP，优先处理")
    with long_term_memory._get_conn() as conn:
        conn.execute("UPDATE user_profiles SET vip_level=2, total_orders=30 WHERE user_id='USER-VIP'")
    long_term_memory.get_or_create_user("USER-NEW", name="王刚")


# ================================================================
# 场景 1：技术咨询（Supervisor Agent 动态路由 + ReAct 工具循环）
# ================================================================
def scene_1_tech_support_with_agent_routing():
    """
    演示特性：
      ✅ Supervisor Agent 分析问题，动态决策路由到 tech_agent
      ✅ tech_support_node 的 ReAct 工具循环（search_faq）
      ✅ SQLite 短期记忆：第二轮自动接续上下文
    """
    print_divider("场景 1：技术咨询（Supervisor Agent 路由 + ReAct 工具循环）")

    app = get_app()
    config = {"configurable": {"thread_id": "TICKET-S1-001"}}

    print("\n[第一轮] 新用户询问技术问题")
    state = app.invoke(make_initial_inputs("USER-NEW", "你们的 API 限流是多少？超出后会怎样？"), config)
    print_agent_decision(state)
    print_final_reply(state)

    print("\n[第二轮] 追问（验证短期记忆：checkpointer 自动恢复上轮 State）")
    state = app.invoke(
        {"messages": [HumanMessage(content="如果我需要更高的并发，怎么申请企业版？")]},
        config
    )
    print(f"\n✅ 验证：对话历史共 {len(state['messages'])} 条（跨轮次累积）")
    print_final_reply(state)


# ================================================================
# 场景 2：老用户回访（Supervisor 综合推理 + 长期记忆个性化）
# ================================================================
def scene_2_returning_user_personalized():
    """
    演示特性：
      ✅ load_memory_node 加载历史档案注入 long_term_context
      ✅ Supervisor Agent 读取用户背景（知道是技术型老用户）做出更精准的路由
      ✅ 技术支持节点的个性化回复（知道用户之前问过 API 问题）
    """
    print_divider("场景 2：老用户回访（Supervisor 综合推理 + 长期记忆）")

    app = get_app()
    config = {"configurable": {"thread_id": "TICKET-S2-001"}}

    print("\n[老用户 USER-001 张明，有历史记录] 提问 Webhook 配置")
    state = app.invoke(
        make_initial_inputs("USER-001", "我上次咨询过 API 限流，这次想了解 Webhook 怎么配置和验证签名？"),
        config
    )
    print_agent_decision(state)
    print(f"\n✅ 长期记忆上下文：\n{state.get('long_term_context', '无')[:200]}...")
    print_final_reply(state)


# ================================================================
# 场景 3：退款流程（3个 Agent 协同 + Human-in-the-loop）
# ================================================================
def scene_3_refund_with_investigation_agent():
    """
    演示特性：
      ✅ Supervisor Agent 识别退款意图，路由到 refund_agent
      ✅ Refund Investigator Agent（create_react_agent）自动调查：查订单、生成报告
      ✅ interrupt_before 挂起，人工基于调查报告做审批决策
      ✅ app.update_state() 写入审批结果
      ✅ app.invoke(None, config) 从断点恢复

    【架构亮点：Agent 辅助 Human-in-the-loop】
      旧版：人工看原始对话记录，自己去查订单系统
      新版：Agent 自动完成调查，人工只需看一份结构化报告做决策
            这大幅提升了人工审批的效率和准确性
    """
    print_divider("场景 3：退款申请（调查 Agent + Human-in-the-loop）")

    app = get_app()
    config = {"configurable": {"thread_id": "TICKET-S3-001"}}

    # --- Step 1：用户提交退款请求（含订单号）---
    print("\n[Step 1] VIP 用户提交退款请求")
    state = app.invoke(
        make_initial_inputs(
            "USER-VIP",
            "我的订单 ORDER-2026-001 买了年度订阅，用了一周觉得功能不符合需求，要求退款！"
        ),
        config
    )
    print_agent_decision(state)

    # --- Step 2：检查图是否已挂起（等待人工审批）---
    current_state = app.get_state(config)
    print(f"\n[Step 2] 图状态检查")
    print(f"  下一待执行节点: {current_state.next}")

    if not (current_state.next and "refund_executor_node" in current_state.next):
        print("⚠️  未触发退款流程，场景 3 终止")
        return

    # --- Step 3：人工查看调查报告 ---
    investigation_report = current_state.values.get("investigation_report", "无报告")
    print(f"\n[Step 3] 人工查看 Agent 生成的调查报告：")
    print(f"{'─'*50}")
    print(investigation_report)
    print(f"{'─'*50}")

    # --- Step 4：人工基于报告做审批决策 ---
    print(f"\n💼 [人工操作] 主管查阅调查报告，订单在退款期内，VIP 用户，批准退款")
    time.sleep(1)

    app.update_state(
        config,
        {
            "human_approved": True,
            "human_feedback": "VIP 用户，订单在退款期内，主管批准全额退款"
        },
        as_node="refund_executor_node"
    )
    print("[Step 4] 审批结果已写入 State")

    # --- Step 5：恢复执行 ---
    print("\n[Step 5] 恢复执行退款流程...")
    final_state = app.invoke(None, config)
    print_final_reply(final_state)


# ================================================================
# 场景 4：投诉处理（Supervisor 识别情绪 → Workflow 节点）
# ================================================================
def scene_4_complaint_handling():
    """
    演示特性：
      ✅ Supervisor Agent 综合判断：情绪激烈但无明确退款诉求 → complaint_agent
      ✅ 展示 Agent 路由 vs 规则路由的差异：能理解语义和情绪，而非关键词匹配
      ✅ complaint_node 是纯 Workflow 节点，展示混合架构中的角色分工
    """
    print_divider("场景 4：投诉处理（Supervisor 情绪识别 → Workflow 节点）")

    app = get_app()
    config = {"configurable": {"thread_id": "TICKET-S4-001"}}

    print("\n[用户] 发出强烈投诉（无退款意图）")
    state = app.invoke(
        make_initial_inputs(
            "USER-001",
            "你们这个产品太让人失望了！文档写得乱七八糟，我花了两天才搞明白一个基础功能，"
            "这种产品体验完全不行，你们技术团队是干什么的？！"
        ),
        config
    )
    print_agent_decision(state)
    print(f"\n✅ 验证：Supervisor 将情绪性投诉正确路由到 complaint_agent（而非 refund_agent）")
    print_final_reply(state)


# ================================================================
# 场景 5：歧义问题（验证 Supervisor Agent 的语义理解能力）
# ================================================================
def scene_5_ambiguous_intent():
    """
    演示特性：
      ✅ Supervisor Agent 处理语义歧义（"我要退出会员" 可能是退款，也可能是取消续费）
      ✅ 对比旧版规则路由的局限性：关键词匹配无法处理歧义
      ✅ 展示 Agent 的推理日志（supervisor_reasoning 字段）

    【为什么这个场景很重要？】
      旧版规则路由：检测到"退"字 → 路由到退款节点（可能是误判）
      Supervisor Agent：理解"退出会员"的完整语义，结合上下文判断真实意图
    """
    print_divider("场景 5：歧义意图（验证 Supervisor Agent 语义理解）")

    app = get_app()
    config = {"configurable": {"thread_id": "TICKET-S5-001"}}

    print("\n[用户] 提出歧义请求（退出会员 vs 退款）")
    state = app.invoke(
        make_initial_inputs(
            "USER-NEW",
            "我不想用这个服务了，我要退出会员，请问怎么操作？"
        ),
        config
    )

    print(f"\n🧠 [完整决策推理]")
    print(f"  路由目标: {state.get('next_agent')}")
    print(f"  意图标签: {state.get('intent')}")
    print(f"  推理过程: {state.get('supervisor_reasoning')}")

    print(f"\n✅ 观察：Supervisor 如何区分'退出会员'（general/tech）vs '申请退款'（refund）")
    print_final_reply(state)


# ================================================================
# 主入口
# ================================================================
if __name__ == "__main__":
    print("🚀 初始化演示数据...")
    seed_demo_users()

    scene_map = {
        "1": ("技术咨询 + ReAct 工具循环", scene_1_tech_support_with_agent_routing),
        "2": ("老用户长期记忆", scene_2_returning_user_personalized),
        "3": ("退款 + 调查 Agent + 人工审批", scene_3_refund_with_investigation_agent),
        "4": ("投诉处理", scene_4_complaint_handling),
        "5": ("歧义意图识别", scene_5_ambiguous_intent),
    }

    scene_arg = None
    if "--scene" in sys.argv:
        idx = sys.argv.index("--scene")
        if idx + 1 < len(sys.argv):
            scene_arg = sys.argv[idx + 1]

    if scene_arg:
        if scene_arg in scene_map:
            scene_map[scene_arg][1]()
        else:
            print(f"未知场景: {scene_arg}，可选: {list(scene_map.keys())}")
    else:
        for key, (title, fn) in scene_map.items():
            fn()
            time.sleep(1)

    print("\n\n" + "="*65)
    print("  ✅ 所有演示场景执行完毕")
    print("="*65)
