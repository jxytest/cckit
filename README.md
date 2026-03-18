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
)
runner = Runner(config=config)

# 方式 2: 从环境变量读取（兼容旧部署）
config = RunnerConfig.from_env()  # 读取 ANTHROPIC_*、SANDBOX_*、PLATFORM_*
```

## MCP 工具

### 使用内置平台工具

```python
from cckit.tools.platform import get_platform_mcp_server

runner = Runner(mcp_servers={"platform": get_platform_mcp_server})
agent = Agent(name="fix", mcp_tools=["get_failure_info"], ...)
```

### 自定义 MCP 工具

```python
def my_mcp_server():
    from claude_agent_sdk import create_sdk_mcp_server, tool

    @tool("my_tool", "Description", {"arg": str})
    async def my_tool(args):
        return {"content": [{"type": "text", "text": "result"}]}

    return create_sdk_mcp_server("my-tools", tools=[my_tool])

runner = Runner(mcp_servers={"custom": my_mcp_server})
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
