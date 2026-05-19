# LangGraph 生产级客服系统 — 学习实践项目

本项目是一个完整的 **LangGraph 生产级最佳实践** Demo，
以"智能客服路由系统"为背景，覆盖 LangGraph 所有核心特性。

---

## 项目结构

```
langGraph/
├── config.py                  # 配置中心（API Key、DB路径、意图标签）
├── state.py                   # 全局状态定义（数据总线）
├── model.py                   # LLM 实例 + 结构化输出模型
├── edge.py                    # 路由函数（条件边逻辑）
├── graph.py                   # 图的组装与编译
├── main.py                    # 演示入口（4个场景）
│
├── memory/
│   ├── short_term.py          # 短期记忆：SQLite Checkpointer
│   └── long_term.py           # 长期记忆：用户档案 + 历史工单
│
├── nodes/
│   ├── memory_nodes.py        # load_memory_node / save_memory_node
│   ├── analyzer_node.py       # 意图分类节点
│   └── support_nodes.py       # 业务节点（tech/refund/complaint/general）
│
├── tools/
│   └── support_tools.py       # 工具定义（search_faq, check_order_status）
│
└── data/                      # SQLite 数据库文件（自动创建）
    ├── checkpoints.db          # 短期记忆存储
    └── long_term_memory.db     # 长期记忆存储
```

---

## 图执行流程

```
START
  │
  ▼
load_memory_node        ← 加载用户长期档案注入上下文
  │
  ▼
analyzer_node           ← LLM 意图分类 (refund/tech/complaint/general)
  │
  ▼ (条件边)
  ├── refund_node ◄──── ⚠️ interrupt_before 挂起，等待人工审批
  │
  ├── tech_support_node ← ReAct 循环
  │     │
  │     ▼ (条件边)
  │     ├── tool_node ──► tech_support_node (循环)
  │     └── save_memory_node
  │
  ├── complaint_node
  └── general_node
        │
        ▼
      save_memory_node   ← 保存本次交互到长期记忆
        │
        ▼
       END
```

---

## 核心特性说明

### 1. State（状态）— 数据总线

所有节点共享同一个 `CustomerServiceState` 字典。
节点通过返回字典来"更新"State 中的字段（非整体替换）。

`messages` 字段使用 `add_messages` Reducer，执行追加而非覆盖。

```python
class CustomerServiceState(TypedDict):
    messages: Annotated[list, add_messages]  # Reducer：追加
    intent: str                               # 普通字段：覆盖
    user_id: str
    long_term_context: Optional[str]
    human_approved: Optional[bool]
    human_feedback: Optional[str]
    resolution_summary: Optional[str]
```

---

### 2. 短期记忆 vs 长期记忆

| 维度 | 短期记忆 | 长期记忆 |
|------|---------|---------|
| 存储位置 | `data/checkpoints.db` | `data/long_term_memory.db` |
| 键 | `thread_id`（工单ID） | `user_id`（用户ID） |
| 生命周期 | 一次会话/工单 | 永久（或按业务规则清理） |
| 技术实现 | LangGraph `SqliteSaver` | 自定义 SQLite 操作类 |
| 核心价值 | 多轮对话 + 断点续传 | 个性化服务 + 历史感知 |
| 生产替换 | → `PostgresSaver` | → PostgreSQL / MySQL |

---

### 3. Tool Calling — ReAct 循环

```
tech_support_node
    │ (模型输出 tool_calls)
    ▼
tool_node (执行工具，追加 ToolMessage)
    │
    ▼
tech_support_node (读取工具结果，继续推理)
    │ (模型不再调用工具)
    ▼
save_memory_node
```

工具通过 `@tool` 装饰器定义，`model.bind_tools(tools)` 将工具注入模型。
`ToolNode` 是 LangGraph 内置节点，自动执行 `AIMessage.tool_calls`。

---

### 4. Human-in-the-loop（人机协作）

```python
# 编译时声明：在进入 refund_node 前挂起
app = builder.compile(
    interrupt_before=["refund_node"],
    checkpointer=checkpointer,
)

# 执行（图在 interrupt 点返回，State 已存档）
app.invoke(inputs, config)

# 查看当前挂起状态
state = app.get_state(config)
print(state.next)  # ['refund_node']

# 人工写入审批结果
app.update_state(config, {"human_approved": True}, as_node="refund_node")

# 从断点恢复（None = 不新增消息）
app.invoke(None, config)
```

---

### 5. 如何切换到 PostgreSQL（生产环境）

只需修改 `memory/short_term.py` 中的 Checkpointer：

```python
# 安装依赖
# pip install langgraph-checkpoint-postgres psycopg[binary]

from langgraph.checkpoint.postgres import PostgresSaver
import psycopg

def get_checkpointer():
    conn = psycopg.connect("postgresql://user:pass@host:5432/mydb")
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # 建表，只执行一次
    return checkpointer
```

其余所有代码无需改动。

---

## 运行方式

```bash
# 安装依赖
pip install -r requirements.txt

# 运行所有演示场景
python main.py

# 只运行指定场景
python main.py --scene 1   # 新用户技术咨询（工具调用）
python main.py --scene 2   # 老用户回访（长期记忆）
python main.py --scene 3   # 退款审批通过
python main.py --scene 4   # 退款审批拒绝
```

---

## 学习路径建议

1. **先读 `state.py`** — 理解数据流转的基础结构
2. **再读 `graph.py`** — 理解图的整体架构和节点连接
3. **读 `memory/`** — 理解短期记忆（Checkpointer）和长期记忆（自定义DB）的区别
4. **读 `tools/` + `nodes/support_nodes.py`** — 理解 ReAct 工具调用循环
5. **读 `main.py`** — 理解 Human-in-the-loop 的完整操作流程
6. **动手改造** — 尝试添加新的意图类型、新的工具、新的记忆字段
