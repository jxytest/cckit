# cckit — Claude Agent Kit

[![Python](https://img.shields.io/badge/python-%3E%3D3.12-blue)](https://www.python.org/)

**cckit** 是一个基于 `claude-agent-sdk`的多 Agent SDK，对其进一步抽象。可以通过声明式 API 快速定义、组合和运行 AI Agent。

## 安装

### 前置条件

- Python >= 3.12
- [Claude CLI](https://docs.anthropic.com/en/docs/claude-cli) 已安装（`claude-agent-sdk` 通过 subprocess 调用 Claude CLI）

### 安装 cckit

```bash
pip install git+<repo-url>
```

依赖项会自动安装：`claude-agent-sdk`、`pydantic`、`pydantic-settings`、`python-gitlab`、`python-dotenv`。

### 开发环境安装

```bash
git clone <repo-url> && cd cckit
pip install -e ".[dev]"
```

## 快速开始

```python
import asyncio
from cckit import Agent, Runner, RunContext, ModelConfig

# 1. 定义 Agent
assistant = Agent(
    name="assistant",
    model=ModelConfig(model="claude-sonnet-4-6", api_key="sk-..."),
    instruction="You are a helpful assistant.",
    tools=["Bash", "Read", "Write"],
)

# 2. 创建运行上下文
ctx = RunContext(prompt="Explain what this project does.")

# 3. 执行
async def main():
    runner = Runner()
    result = await runner.run(assistant, ctx)
    print(result)

asyncio.run(main())
```

### 流式输出

```python
runner = Runner()

# 基础用法（向后兼容）
async for event in runner.run_stream(assistant, ctx):
    print(f"[{event.event_type}] {event.text}")

# 流式 + 最终结果
stream = runner.run_stream(assistant, ctx)
async for event in stream:
    print(f"[{event.event_type}] {event.text}")
final = stream.result  # 流结束后获取最终 AgentResult
print(f"Cost: ${final.cost_usd}, Extra: {final.extra}")
```

## 核心概念

cckit 采用三层设计，将**定义**、**上下文**和**执行**彻底分离：

| 层 | 类 | 职责 |
|---|---|---|
| **Agent** | `Agent()` | "我是谁" — name、instruction、tools、sub_agents、skills、model |
| **RunContext** | `RunContext()` | "怎么跑" — workspace、git（GitConfig）、prompt、params、env |
| **Runner** | `Runner()` | "执行引擎" — 中间件、并发控制、SDK 桥接 |

同一个 Agent 定义可在不同 RunContext 中复用。业务凭证通过 `RunContext.params` 按任务传递。

## Agent 定义

```python
from cckit import Agent, ModelConfig, LiteLlm

# 最简用法
assistant = Agent(
    name="assistant",
    model=ModelConfig(model="claude-sonnet-4-6", api_key="sk-..."),
    instruction="You are a helpful assistant.",
    tools=["Bash", "Read", "Write"],
)

# 使用 LiteLLM 桥接第三方模型
gemini = Agent(
    name="gemini",
    model=LiteLlm(model="openai/gemini-3-pro", api_base="http://localhost:4000"),
    tools=["Read", "Grep"],
)

# 动态 instruction + 子 Agent 组合
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

### 模型配置

`model` 参数支持三种形式：

| 形式 | 示例 | 说明 |
|------|------|------|
| `str` | `"claude-sonnet-4-6"` | 直连 Anthropic |
| `ModelConfig` | `ModelConfig(model=..., api_key=..., max_tokens=...)` | 显式配置 |
| `LiteLlm` | `LiteLlm(model="openai/gpt-4", api_base=...)` | 通过 LiteLLM 代理桥接任意模型 |

**解析优先级**：Agent.model > RunnerConfig.default_model > 环境变量

## 运行上下文

```python
from cckit import RunContext, WorkspaceConfig, GitConfig

# 带 git clone 的沙箱工作空间（凭证与 Agent 进程隔离）
ctx = RunContext(
    prompt="Fix the broken locator",
    workspace=WorkspaceConfig(enabled=True),
    git=GitConfig(
        repo_url="https://gitlab.com/team/repo.git",
        token="glpat-xxx",   # 仅注入 git 子进程，Agent 不可见
        clone=True,
    ),
    params={"test_file": "tests/test_login.py"},
)

# 轻量执行（无工作空间）
ctx = RunContext(prompt="Explain this code", workspace=WorkspaceConfig(enabled=False))

# 会话恢复 — 续接之前的对话
ctx = RunContext(
    prompt="Add tests",
    resume_session_id="abc-123",
    workspace_dir=prev_dir,
)
```

## Runner 与中间件

```python
from cckit import Runner, RunnerConfig, ModelConfig
from cckit.middleware import ConcurrencyMiddleware, RetryMiddleware, LoggingMiddleware

runner = Runner(
    config=RunnerConfig(
        default_model=ModelConfig(model="claude-sonnet-4-6", api_key="sk-..."),
    ),
    middlewares=[
        ConcurrencyMiddleware(max_concurrent=5),   # Semaphore 限流
        RetryMiddleware(max_retries=3),             # 指数退避重试
        LoggingMiddleware(),                        # 耗时、费用记录
    ],
)

# 一次性获取结果（on_after 回调修改的 result.extra 会保留）
result = await runner.run(fix_agent, ctx)

# 流式获取事件
async for event in runner.run_stream(fix_agent, ctx):
    print(f"[{event.event_type}] {event.text}")

# 流式 + 最终结果
stream = runner.run_stream(fix_agent, ctx)
async for event in stream:
    print(f"[{event.event_type}] {event.text}")
result = stream.result  # 与 on_after 操作的是同一个对象
```

### 自定义中间件

继承 `Middleware`，实现 `wrap` 方法：

```python
from cckit.middleware import Middleware

class AuditMiddleware(Middleware):
    async def wrap(self, next_call, prompt, options, collector, ctx):
        logger.info("Starting task %s", ctx.task_id)
        async for event in next_call(prompt, options, collector):
            yield event
```

## 配置

**无全局单例** — 所有配置通过构造函数显式传递。

```python
from cckit import RunnerConfig, ModelConfig
from cckit.sandbox import SandboxOptions
from pathlib import Path

# 方式 1: 显式配置
config = RunnerConfig(
    default_model=ModelConfig(model="claude-sonnet-4-6", api_key="sk-..."),
    sandbox=SandboxOptions(enabled=True, workspace_root=Path("/data/ws")),
    permission_mode="bypassPermissions",  # 沙箱开启时自动覆盖为 dontAsk
)
runner = Runner(config=config)

# 方式 2: 从环境变量读取（兼容旧部署）
config = RunnerConfig.from_env()  # 读取 ANTHROPIC_*、SANDBOX_*、PLATFORM_*
```

### 环境变量一览

| 变量 | 对应字段 | 默认值 |
|------|----------|--------|
| `ANTHROPIC_API_KEY` | `default_model.api_key` | — |
| `ANTHROPIC_BASE_URL` | `default_model.base_url` | — |
| `ANTHROPIC_MODEL` | `default_model.model` | `claude-sonnet-4-6` |
| `ANTHROPIC_MAX_TOKENS` | `default_model.max_tokens` | `16384` |
| `ANTHROPIC_MAX_TURNS` | `default_model.max_turns` | `50` |
| `ANTHROPIC_TIMEOUT_SECONDS` | `default_model.timeout_seconds` | `300` |
| `SANDBOX_ENABLED` | `sandbox.enabled` | `false` |
| `SANDBOX_WORKSPACE_ROOT` | `sandbox.workspace_root` | `/tmp/cckit_workspaces` |
| `SANDBOX_DENY_READ` | `sandbox.deny_read` | `["~/"]` |
| `PLATFORM_PERMISSION_MODE` | `permission_mode` | `bypassPermissions` |
| `PLATFORM_MAX_CONCURRENT_AGENTS` | `max_concurrent_agents` | `5` |
| `PLATFORM_SKILLS_DIR` | `skills_dir` | `/opt/agent-platform/skills` |

## 沙箱隔离

> **平台支持**：沙箱依赖 macOS Seatbelt / Linux bubblewrap，**Windows 原生不支持**（cckit 会在启动时打印警告）。WSL2 可正常使用。

### 默认行为说明

开启沙箱（`enabled=True`）后，默认规则如下：

| 能力 | 默认行为 |
|------|----------|
| **写入** | 仅允许写入当前任务的 workspace 目录，其他路径均不可写 |
| **读取** | 可读取大部分系统路径；**家目录（`~/`）被屏蔽**（防止访问 `~/.ssh` 等敏感文件） |
| **网络** | Bash 子进程可访问所有出站网络 |

> 沙箱**不是**"只读工作目录"的白名单模式。读取限制仅屏蔽家目录，`/etc`、`/usr` 等系统路径仍可读。
> 如需更严格的隔离，使用 `deny_read` 追加更多路径。

### 最小安全配置

对于大多数场景，直接开启即可——默认已屏蔽家目录（`~/.ssh`、`~/.aws` 等）：

```python
from cckit import Runner, RunnerConfig
from cckit.sandbox import SandboxOptions

runner = Runner(
    config=RunnerConfig(
        sandbox=SandboxOptions(enabled=True),
    )
)
```

沙箱开启后，cckit 会自动将 `permission_mode` 切换为 `dontAsk`，确保 `deny` 规则对 Claude 内置工具（Read / Edit / Glob）同样生效。

### 两套隔离机制

cckit 同时配置 SDK 的两个隔离层，单独配置任意一层都会有死角：

| 机制 | 覆盖范围 |
|------|----------|
| `sandbox.filesystem`（OS 级） | 限制 **Bash 子进程**的文件读写 |
| `permissions.deny`（SDK 级） | 限制 Claude **内置工具**（Read / Edit / Glob / Grep） |

`deny_read` / `deny_write` 中配置的路径会自动同步到两侧，无需手动维护。

### 追加更严格的限制

如果需要在默认基础上进一步收紧：

```python
SandboxOptions(
    enabled=True,
    # 额外屏蔽敏感目录（家目录已默认屏蔽）
    deny_read=["~/", "/etc/ssl", "/run/secrets"],
    # 限制出站网络，只允许访问指定域名
    allowed_domains=["api.example.com"],
    # git 在沙箱外运行，保留完整网络权限
    excluded_commands=["git"],
)
```

### permission_mode 说明

`permission_mode` 位于 `RunnerConfig`（非 `ModelConfig`），控制 Claude 内置工具的权限策略：

| 值 | 行为 |
|----|------|
| `bypassPermissions`（默认） | 跳过所有权限检查，**deny 规则不生效** |
| `dontAsk` | 遵守 deny 规则，不弹出审批提示 |

沙箱开启时 cckit 自动切换为 `dontAsk`，**无需手动设置**。

## MCP 工具


| 类型 | TypedDict | 说明 |
|------|-----------|------|
| `stdio` | `McpStdioServerConfig` | 启动子进程（`command` + `args`） |
| `sse` | `McpSSEServerConfig` | 连接远程 SSE 端点 |
| `http` | `McpHttpServerConfig` | 连接远程 HTTP 端点 |
| `sdk` | `McpSdkServerConfig` | 进程内 SDK 服务（`create_sdk_mcp_server`） |

### Playwright MCP（浏览器自动化）

cckit 内置 `playwright_mcp_server()` 工厂函数，开箱即用：

```python
from cckit.tools.platform import playwright_mcp_server
from cckit import Agent

agent = Agent(
    name="browser-agent",
    tools=["Bash"],
    mcp_servers={"playwright": playwright_mcp_server()},  # 默认 headless=True
    instruction="Use playwright MCP tools to automate browser tasks.",
)
```

> **前置条件**：需要 Node.js / `npx`。`@playwright/mcp` 会在首次运行时自动下载，
> 或提前安装：`npm install -g @playwright/mcp`

可选参数：

```python
playwright_mcp_server(headless=False)                      # 显示浏览器窗口（调试用）
playwright_mcp_server(extra_args=["--timeout", "30000"])   # 追加 CLI 参数
```

### 自定义 SDK MCP 工具

使用 `@tool` 装饰器定义进程内工具，返回 `McpSdkServerConfig`：

```python
from claude_agent_sdk import create_sdk_mcp_server, tool
from claude_agent_sdk.types import McpSdkServerConfig
from cckit import Agent

@tool("fetch_data", "Fetch data from the platform", {"id": str})
async def fetch_data(args: dict) -> dict:
    result = await my_api.get(args["id"])
    return {"content": [{"type": "text", "text": result}]}

server = create_sdk_mcp_server("my-tools", tools=[fetch_data])

agent = Agent(
    name="my-agent",
    mcp_servers={"my-tools": McpSdkServerConfig(type="sdk", name="my-tools", instance=server)},
)
```

`cckit.tools.platform.sdk_mcp_server()` 提供了一个带 `echo` 工具的完整示例，可直接复制修改。

### 使用 SSE / HTTP 远程服务

```python
from claude_agent_sdk.types import McpSSEServerConfig, McpHttpServerConfig
from cckit import Agent

agent = Agent(
    name="my-agent",
    mcp_servers={
        "remote-sse":  McpSSEServerConfig(type="sse",  url="https://my-mcp.example.com/sse"),
        "remote-http": McpHttpServerConfig(type="http", url="https://my-mcp.example.com/mcp"),
    },
)
```

## Skills

Agent 通过 `skills` + `skills_dir` 声明使用技能。Runner 自动将 skills 安全复制到工作空间，SDK 按需加载。

```python
agent = Agent(
    name="tester",
    skills=["playwright-testing"],
    skills_dir="/opt/my-skills",
    ...
)
```

## 示例

完整示例见 [`examples/`](examples/) 目录：

- **FixAgent** — 自动修复 + 创建 Merge Request
- **CodeModifyAgent** — 代码修改 Agent


## 已知问题
- [ ] `Windows`下，系统提示词设置无法生效， 属于`claude-agent-sdk`本身问题