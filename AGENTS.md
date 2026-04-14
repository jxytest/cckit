# cckit — Claude Agent Kit

> 本文档面向 AI 编程助手和后续开发者，帮助快速理解项目架构、核心概念和扩展方式。

## 1. 项目定位

`cckit` 是一个 **pip 可安装的多 Agent SDK**，基于 `claude-agent-sdk`（封装 Claude CLI 的 Python SDK）构建。提供：

- **声明式 Agent 定义** — `Agent()` 实例化即定义，无需继承
- **统一模型接入** — 所有模型配置统一按 LiteLLM 语义解释，cckit 内部负责桥接到 Claude SDK
- **子 Agent 组合** — `sub_agents=[...]` 递归组合
- **可插拔中间件** — 重试、限流、日志等横切关注点
- **沙箱级工作空间隔离** — 每个任务独立临时目录
- **Skills 即插即用** — 通过路径参数加载 Claude Agent Skills
- **Git/GitLab 集成** — 异步 git 操作 + MR 创建
- **MCP 工具扩展** — 可选内置工具包，用户也可自定义

**安装**：`pip install cckit`

## 2. 技术栈

| 组件 | 说明 |
|------|------|
| Python | >= 3.12 |
| claude-agent-sdk | 底层 subprocess 调 Claude CLI |
| pydantic / pydantic-settings | 数据模型 & 配置 |
| litellm | provider 解析 + 模型协议桥接 |
| starlette / uvicorn | 运行期本地 Anthropic bridge |
| python-gitlab | GitLab API 客户端 |

## 3. 目录结构

```
cckit/                              # pip install cckit
├── __init__.py                    # 公共 API: Agent, Runner, RunContext 等
├── agent.py                       # Agent 类（核心）
├── runner.py                      # Runner 类（执行引擎）
├── types.py                       # ModelConfig, GitConfig, RunContext, AgentResult, StreamResult...
├── exceptions.py                  # CckitError 异常层次
├── _models.py                     # CustomModel 基类（内部）
├── _cli.py                        # Claude CLI 检测
├── _engine/                       # SDK 桥接层（私有）
│   ├── sdk_bridge.py              # query 交互
│   └── model_bridge.py            # LiteLLM bridge 路由
├── middleware/                     # 可插拔中间件
│   ├── base.py                    # Middleware 抽象类
│   ├── retry.py                   # RetryMiddleware（指数退避）
│   ├── concurrency.py             # ConcurrencyMiddleware（Semaphore 限流）
│   └── logging.py                 # LoggingMiddleware（耗时、费用记录）
├── sandbox/                       # 工作空间 & 沙箱
│   ├── workspace.py               # WorkspaceManager（文件系统生命周期）
│   └── config.py                  # SandboxConfigBuilder
├── skill/                         # Skill 供给
│   └── provisioner.py             # SkillProvisioner（安全复制 skills 到 workspace）
├── git/                           # Git 操作 + GitLab
│   ├── operations.py              # async git CLI（run_git/clone/branch/commit/push/status/diff）
│   └── gitlab_client.py           # GitLabClient
└── tools/                         # 可选内置工具包
    └── platform.py                # 平台 MCP 工具（用户显式导入）

examples/                          # 不随 pip 发布
├── fix_agent.py                   # FixAgent 示例
├── code_modify_agent.py           # CodeModifyAgent 示例
└── README.md

scripts/
└── smoke_test.py                  # 冒烟测试便捷入口（调用 pytest）

tests/
└── test_smoke.py                  # 冒烟测试（pytest 格式，不需要 API Key）
```

## 4. 核心三层设计

```
Agent = "我是谁" → name, instruction, tools, sub_agents, skills, model, sandbox, hooks, task_budget, context
RunContext = "怎么跑" → workspace, git(GitConfig), prompt, params, env
Runner = "执行引擎" → default_model、workspace_root、中间件、并发控制、SDK 桥接
```

### 4.1 Agent — 声明式定义

```python
from cckit import Agent, ModelConfig

# 最简用法
assistant = Agent(
    name="assistant",
    model=ModelConfig(model="claude-sonnet-4-6", api_key="sk-..."),
    instruction="You are a helpful assistant.",
    tools=["Bash", "Read", "Write"],
)

# 走 OpenAI
gpt_via_bridge = Agent(
    name="gpt-via-bridge",
    model=ModelConfig(
        model="openai/gpt-4o-mini",
        base_url="https://api.openai.com/v1",
        api_key="sk-openai",
    ),
    tools=["Read", "Grep"],
)

# 走 OpenAI-compatible 网关上的 Claude
claude_via_gateway = Agent(
    name="claude-via-gateway",
    model=ModelConfig(
        model="openai/claude-sonnet-4-6",
        base_url="http://localhost:4000/v1",
        api_key="sk-gateway",
    ),
    tools=["Read", "Grep"],
)

# 动态 instruction + 子 Agent
def build_instruction(ctx):
    return f"You are a {ctx.params.get('lang', 'Python')} expert."

analyzer = Agent(name="analyzer", description="Log analyzer", tools=["Read", "Grep"])

fix_agent = Agent(
    name="fix",
    instruction=build_instruction,
    tools=["Bash", "Read", "Write", "Edit"],
    sub_agents=[analyzer],
    required_params=["test_file", "error_log"],
    skills=["playwright-testing"],
    skills_dir="/opt/my-skills",
    on_after=create_merge_request,
)
```

### 4.2 RunContext — 运行时上下文

```python
from cckit import RunContext, WorkspaceConfig

# 带 git clone
ctx = RunContext(
    prompt="Fix the broken locator",
    workspace=WorkspaceConfig(enabled=True),
    git=GitConfig(
        repo_url="https://gitlab.com/team/repo.git",
        token="glpat-xxx",
        clone=True,
    ),
    params={"test_file": "tests/test_login.py"},
)

# 轻量执行（无 workspace）
ctx = RunContext(prompt="Explain this code", workspace=WorkspaceConfig(enabled=False))

# 会话恢复
ctx = RunContext(prompt="Add tests", resume_session_id="abc-123", workspace_dir=prev_dir)
```

### 4.3 Runner — 执行引擎

```python
from cckit import Runner, RunnerConfig, ModelConfig
from cckit.middleware import ConcurrencyMiddleware, RetryMiddleware, LoggingMiddleware

runner = Runner(
    config=RunnerConfig(
        default_model=ModelConfig(model="claude-sonnet-4-6", api_key="sk-..."),
    ),
    middlewares=[
        ConcurrencyMiddleware(max_concurrent=5),
        RetryMiddleware(max_retries=3),
        LoggingMiddleware(),
    ],
)

# 流式
async for event in runner.run_stream(fix_agent, ctx):
    print(f"[{event.event_type}] {event.text}")

# 流式 + 最终结果
stream = runner.run_stream(fix_agent, ctx)
async for event in stream:
    print(f"[{event.event_type}] {event.text}")
result = stream.result  # on_after 回调修改的同一个对象

# 一次性
result = await runner.run(fix_agent, ctx)
```

## 5. 执行流程

```
Runner.run_stream(agent, ctx) → StreamResult:
  ├── _validate_context()
  │   ├── 检查 agent.required_params
  │   ├── git.clone=True 但无 git.repo_url → 报错
  │   └── skills 非空但 workspace.enabled=False → 报错
  ├── agent.before_execute(ctx)
  ├── Workspace:
  │   ├── resume → WorkspaceManager.resume()
  │   ├── workspace.enabled=True → create() + git clone + provision skills
  │   └── workspace.enabled=False → 跳过
  ├── instruction = agent.resolve_instruction(ctx)
  ├── model = _resolve_model(agent)
  ├── options = _build_options(...)
  ├── [中间件链] → SDK bridge → yield SDK messages
  ├── agent.after_execute(ctx, result) / agent.error_execute(ctx, error)
  └── workspace cleanup / suspend

Runner.run(agent, ctx) → AgentResult:
  ├── stream = run_stream(agent, ctx)
  ├── 消费所有事件
  └── return stream.result（与 on_after 操作的是同一个对象）
```

## 6. SDK 隔离

只有 2 个文件导入 `claude_agent_sdk`（全部惰性导入）：

| 文件 | 导入内容 |
|------|----------|
| `cckit/_engine/sdk_bridge.py` | `query`, `ClaudeAgentOptions`, `AgentDefinition` |
| `cckit/tools/platform.py` | `create_sdk_mcp_server`, `tool` |

`import cckit` 不会触发 SDK 导入。

## 7. 配置系统

**无全局单例** — 配置通过构造函数向下传递。
沙箱策略定义在 `Agent(..., sandbox=...)`；`RunnerConfig` 只负责运行器级配置，例如 `workspace_root`。

```python
# 方式 1: 显式配置
config = RunnerConfig(
    default_model=ModelConfig(model="claude-sonnet-4-6", api_key="sk-..."),
    workspace_root=Path("/data/ws"),
)

# 方式 2: 从环境变量读取
config = RunnerConfig.from_env()  # 读 ANTHROPIC_*、PLATFORM_*
```

## 8. 模型配置

model 参数支持两种形式：
- `str`：`"claude-sonnet-4-6"`、`"openai/gpt-4o-mini"` 等 LiteLLM 模型名
- `ModelConfig`：显式指定 `api_key`/`base_url`/`max_tokens` 等

**解析优先级**：Agent.model > RunnerConfig.default_model > 环境变量

接入规则：
- `cckit` 统一按 LiteLLM 语义解释 `model`、`base_url`、`api_key`
- 对外字段名仍叫 `base_url`，但底层等效于 LiteLLM 的 `api_base`
- Claude 模型走 Anthropic 协议时直接写 `claude-sonnet-4-6`
- Claude 模型走 OpenAI 协议时写 `openai/claude-sonnet-4-6`
- 当同一个模型名可能出现在不同协议下时，用 provider 前缀消歧义，例如 `openai/claude-sonnet-4-6`
- `base_url` 填 API base，不要填具体 endpoint；若误填 `.../v1/messages`、`.../chat/completions`、`.../responses`，会自动规范化

## 9. MCP 工具

平台 MCP 工具在 `cckit/tools/platform.py`，用户显式导入：

```python
from cckit.tools.platform import get_platform_mcp_server

runner = Runner(mcp_servers={"platform": get_platform_mcp_server})
agent = Agent(name="fix", mcp_tools=["get_failure_info"], ...)
```

自定义 MCP 工具包：

```python
def my_mcp_server():
    from claude_agent_sdk import create_sdk_mcp_server, tool
    @tool("my_tool", "Description", {"arg": str})
    async def my_tool(args):
        return {"content": [{"type": "text", "text": "result"}]}
    return create_sdk_mcp_server("my-tools", tools=[my_tool])

runner = Runner(mcp_servers={"custom": my_mcp_server})
```

## 10. Skills

Agent 通过 `skills=["name"]` + `skills_dir="/path"` 声明使用。Runner 自动将 skills 复制到 `workspace/.claude/skills/`，SDK 按需加载。

安全机制：名称正则校验、路径遍历防护、符号链接拒绝、`.claude/` 预清理。

## 11. 中间件

继承 `Middleware`，实现 `async def wrap(self, next_call, prompt, options, state, ctx)`：

```python
from cckit.middleware import Middleware

class AuditMiddleware(Middleware):
    async def wrap(self, next_call, prompt, options, state, ctx):
        logger.info("Starting task %s", ctx.task_id)
        async for event in next_call(prompt, options, state):
            yield event
```

## 12. 异常体系

```
CckitError                        # 根异常
├── AgentExecutionError           # SDK 调用失败、超时等
├── WorkspaceError                # 工作目录创建/清理失败
├── SkillError                    # skill 未找到、校验失败、复制失败
├── GitOperationError             # git CLI 命令失败
└── GitLabAPIError                # GitLab API 调用失败
```

## 16. Claude Hooks（原生钩子）

cckit 通过 `Agent(hooks=...)` 直接透传 SDK 的 `hooks` 参数，支持 Claude 原生 10 种 Hook 事件。Hook 回调在 **SDK 进程内**（Python 侧）同步执行，无需外部进程。

### 支持的 Hook 事件

| 事件 | 触发时机 |
|------|----------|
| `PreToolUse` | 工具调用前（可拦截 / 修改输入） |
| `PostToolUse` | 工具调用成功后 |
| `PostToolUseFailure` | 工具调用失败后 |
| `UserPromptSubmit` | 用户提示提交时 |
| `Stop` | 主 Agent 停止时 |
| `SubagentStop` | 子 Agent 停止时 |
| `SubagentStart` | 子 Agent 启动时 |
| `PreCompact` | 上下文压缩前 |
| `Notification` | Claude 发出通知时 |
| `PermissionRequest` | 权限请求时 |

### Hook 回调签名

```python
async def my_hook(
    input: HookInput,         # 事件输入（强类型 TypedDict）
    tool_use_id: str | None,  # 工具调用 ID（仅工具类事件有值）
    ctx: HookContext,         # Hook 上下文（含 abort signal）
) -> HookJSONOutput:          # 返回控制指令（可返回空 {} 表示放行）
    ...
```

`HookJSONOutput` 可以是 `SyncHookJSONOutput`（同步控制）或 `AsyncHookJSONOutput`（异步延迟执行）。

### HookMatcher

```python
from claude_agent_sdk import HookMatcher

HookMatcher(
    matcher="Bash",           # 匹配字符串（如工具名），None = 匹配所有
    hooks=[my_hook],          # 回调列表
    timeout=30.0,             # 单个 Hook 超时秒数（默认 60）
)
```

## 17. TaskBudget（Token 预算）

`TaskBudgetConfig` 向模型声明 token 预算，使模型能主动控制工具使用节奏，在达到上限前优雅收尾。由 SDK 以 `task-budgets-2026-03-13` Beta Header 发送。

```python
TaskBudgetConfig(
    total=50_000,  # 总 token 预算（输入 + 输出之和）
)
```

## 18. ContextConfig（上下文窗口与自动压缩）

`ContextConfig` 控制 Claude Code 的上下文窗口大小和自动压缩触发时机。cckit 将其转换为 CLI 环境变量注入到 Agent 子进程中，**每个 Agent 独立进程，互不影响**。

> **重要**：`max_context_tokens` 是上下文窗口大小（输入+输出总量），与 `ModelConfig.max_tokens`（最大输出 token 数）是不同概念。`max_context_tokens` 默认为 `None`，表示使用模型原生上下文窗口，不会回退到 `ModelConfig.max_tokens`。

```python
ContextConfig(
    max_context_tokens=None,   # None = 使用模型原生上下文窗口（非 max_tokens）
    auto_compact_pct=80,       # 上下文使用率 80% 时触发压缩
    disable_auto_compact=False, # 禁用自动压缩
)
```

### 环境变量映射

| ContextConfig 字段 | CLI 环境变量 | 说明 |
|---|---|---|
| `max_context_tokens` | `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | 上下文窗口大小（仅非 None 时设置） |
| `auto_compact_pct` | `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | 压缩触发百分比（仅非 80 时设置） |
| `disable_auto_compact` | `DISABLE_AUTO_COMPACT` | 禁用自动压缩 |

### CLI 源码中的阈值计算

```
effectiveContextWindow = min(modelContextWindow, CLAUDE_CODE_AUTO_COMPACT_WINDOW)
                       - min(maxOutputTokens, 20_000)

autoCompactThreshold = effectiveContextWindow - 13_000  // AUTOCOMPACT_BUFFER_TOKENS

// 或用百分比覆盖：
autoCompactThreshold = min(
    effectiveContextWindow * CLAUDE_AUTOCOMPACT_PCT_OVERRIDE%,
    effectiveContextWindow - 13_000
)
```

### Runner 中的注入位置

`runner.py` `_build_options()` 中，在 `env` 构建之后、模型端点覆盖之前：

```python
env: dict[str, str] = dict(ctx.env)
context_cfg = agent.context
if context_cfg is not None:
    env.update(context_cfg.to_env())
```

## 13. 验证方式

```bash
# 冒烟测试（不需要 API Key）— 两种方式
pytest tests/ -v
python scripts/smoke_test.py  # 便捷包装，实际调用 pytest

# 代码风格
ruff check cckit/
```

## 14. 扩展路线

| 方向 | 做法 |
|------|------|
| 新增 Agent | `Agent(name="xxx", ...)` 实例化 |
| 新增 Skill | 在 skills 目录放 `SKILL.md`，Agent 声明 `skills=["name"]` |
| 新增 MCP 工具 | 自定义 MCP server factory，传给 `Runner(mcp_servers=...)` |
| 自定义中间件 | 继承 `Middleware`，实现 `wrap()` |
| FastAPI 集成 | 调用 `runner.run_stream()` + SSE 推送 |
| 会话恢复 | `RunContext(resume_session_id="...", workspace_dir=...)` |

## 15. 编码规范

- Python >= 3.12，所有 I/O 使用 async
- datetime 必须 timezone-aware UTC
- 同步第三方 SDK（如 python-gitlab）通过 `loop.run_in_executor` 包装
- Pydantic 模型继承 `CustomModel`
- Ruff 格式化，行宽 100
