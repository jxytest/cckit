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
    model=ModelConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-..."),
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

# 流式
stream = runner.run_stream(assistant, ctx)
async for message in stream:
    print(message)
final = stream.result  # 流结束后获取最终 AgentResult
print(f"Cost: ${final.cost_usd}, Extra: {final.extra}")
```

## 核心概念

cckit 采用三层设计，将**定义**、**上下文**和**执行**彻底分离：

| 层 | 类 | 职责 |
|---|---|---|
| **Agent** | `Agent()` | "我是谁" — name、instruction、tools、sub_agents、skills、model、sandbox |
| **RunContext** | `RunContext()` | "怎么跑" — workspace、git（GitConfig）、prompt、params、env |
| **Runner** | `Runner()` | "执行引擎" — default_model、workspace_root、中间件、并发控制 |

同一个 Agent 定义可在不同 RunContext 中复用。业务凭证通过 `RunContext.params` 按任务传递。

## Agent 定义

```python
from cckit import Agent, ModelConfig

# 最简用法
assistant = Agent(
    name="assistant",
    model=ModelConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-..."),
    instruction="You are a helpful assistant.",
    tools=["Bash", "Read", "Write"],
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

配置仅需三个字段：`model`、`api_key` 和 `base_url`（大部分情况不填）。

#### 协议路由

前缀决定请求走哪种协议：

| 前缀 | 协议 | 说明 |
|------|------|------|
| `anthropic/` | Anthropic Messages | 直连 Claude SDK，**零开销**，不经过桥接 |
| `openai/` | OpenAI Responses API | 走 LiteLLM 的 Responses API 适配器 |
| `responses/` | OpenAI Responses API | 与 `openai/` 相同，语义更明确 |
| `deepseek/`、`dashscope/` 等 | Chat Completions | 走 LiteLLM 的 chat/completions 适配器 |
| 无前缀 | Chat Completions | 默认按 `chat/completions` 处理；也可通过 `base_url` 后缀自动推断协议 |

> **最佳实践**：**强烈建议使用大模型原厂前缀**（如 `dashscope/`、`deepseek/`）。原厂前缀拥有最好的工具调用（Tool Calling）兼容性和精准的流式计费支持，体验全面优于将其视为通用 `openai/` 前缀。

```python
from cckit import ModelConfig

# Anthropic 直连（不经过桥接层）
anthropic = ModelConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-ant...")

# OpenAI Responses API
openai = ModelConfig(model="openai/gpt-4o", api_key="sk-...")

# 使用 responses/ 前缀也走 Responses API（适用于 openai 新模型）
responses = ModelConfig(model="responses/gpt-5.2", api_key="sk-...")

# 其他厂商走 chat/completions
qwen = ModelConfig(model="dashscope/qwen-max", api_key="sk-dashscope...")
deepseek = ModelConfig(model="deepseek/deepseek-chat", api_key="sk-ds...")

# 代理/自定义端点
proxy = ModelConfig(model="openai/gpt-4o", base_url="https://api.proxy.com/v1", api_key="sk-...")
```

#### 常见厂商前缀及配置指南

> 💡 LiteLLM 支持 100+ 厂商，完整列表请查阅 [LiteLLM Providers](https://docs.litellm.ai/docs/providers)。
> 对于原生前缀（如 `dashscope/` 或 `deepseek/`），**清空并省略 `base_url`** 是保障兼容性最安全的做法。

| 厂商生态 | 推荐前缀示例 | `base_url` 配置指南 |
|---|---|---|
| **Anthropic** | `anthropic/claude-sonnet-4-6` | 默认空直连官网。反代填 `https://api.anthropic.proxy/v1` |
| **OpenAI** | `openai/gpt-4o` | 默认空连官方。自建/代理填 `https://api.yourproxy.com/v1` |
| **OpenAI (Responses)** | `responses/gpt-5.2` | 与 `openai/` 等价，语义更明确 |
| **通义千问** | `dashscope/qwen-max` | 默认空。若必须填请写全 `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| **DeepSeek** | `deepseek/deepseek-chat` | 默认空。中转端点填 `https://api.deepseek.com/v1` |
| **智谱 GLM** | `zhipu/glm-4` | 默认空连官方端点 |
| **Kimi** | `moonshot/moonshot-v1-auto` | 默认空。代理端点填 `https://api.moonshot.cn/v1` |
| **豆包** | `volcengine/ep-xxx...` | 默认空连官方端点 |
| **Gemini** | `gemini/gemini-1.5-pro` | 默认空连官方端点 |
| **本地 Ollama** | `ollama/qwen2.5` | ✅必填局域网地址：`http://localhost:11434` |
| **云托管服务** | `azure/`、`bedrock/`、`vertex_ai/` | ✅按厂商规范必填真实资源端点和专属格式配置 |

> **⚠️ API_BASE 排雷提示**：如果你配置了 `base_url` 却遇到了 `404 Not Found`（特别是 path 包含 `/chat/completions` 的报错），说明你只填了裸机域名导致底层路由寻址越界。遇到此报错，**直接移除 `base_url` 参数，或者在其后补足 `/v1` 即可恢复正常**。


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
        default_model=ModelConfig(model="anthropic/claude-sonnet-4-6", api_key="sk-..."),
    ),
    middlewares=[
        ConcurrencyMiddleware(max_concurrent=5),   # Semaphore 限流
        RetryMiddleware(max_retries=3),             # 指数退避重试
        LoggingMiddleware(),                        # 耗时、费用记录
    ],
)

# 一次性获取结果（on_after 回调修改的 result.extra 会保留）
result = await runner.run(fix_agent, ctx)

# 流式获取消息
async for message in runner.run_stream(fix_agent, ctx):
    print(message)

# 流式 + 最终结果
stream = runner.run_stream(fix_agent, ctx)
async for message in stream:
    print(message)
result = stream.result  # 与 on_after 操作的是同一个对象
```

### 自定义中间件

继承 `Middleware`，实现 `wrap` 方法：

```python
from cckit.middleware import Middleware

class AuditMiddleware(Middleware):
    async def wrap(self, next_call, prompt, options, state, ctx):
        logger.info("Starting task %s", ctx.task_id)
        async for message in next_call(prompt, options, state):
            yield message
```

## 沙箱隔离

> **平台支持**：沙箱依赖 macOS Seatbelt / Linux bubblewrap，**Windows 原生不支持**（cckit 会在启动时打印警告）。WSL2 可正常使用。

### 默认行为说明

在 Agent 上开启沙箱（`enabled=True`）后，默认规则如下：

| 能力 | 默认行为 |
|------|----------|
| **写入** | 仅允许写入当前任务的 workspace 目录，其他路径均不可写 |
| **读取** | 可读取大部分系统路径；**家目录（`~/`）被屏蔽**（防止访问 `~/.ssh` 等敏感文件） |
| **网络** | 默认阻断所有出站网络；如需联网，显式配置 `allowed_domains`，或将特定命令加入 `excluded_commands` |

> 沙箱**不是**"只读工作目录"的白名单模式。读取限制仅屏蔽家目录，`/etc`、`/usr` 等系统路径仍可读。
> 如需更严格的隔离，使用 `deny_read` 追加更多路径。

### 最小安全配置

对于大多数场景，直接在 Agent 上开启即可，默认已屏蔽家目录（`~/.ssh`、`~/.aws` 等）：

```python
from cckit import Agent, Runner, SandboxOptions

agent = Agent(
    name="safe-agent",
    instruction="You are a sandboxed assistant.",
    sandbox=SandboxOptions(enabled=True),
)
runner = Runner()
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
agent = Agent(
    name="strict-agent",
    sandbox=SandboxOptions(
        enabled=True,
        # 额外屏蔽敏感目录（家目录已默认屏蔽）
        deny_read=["~/", "/etc/ssl", "/run/secrets"],
        # 限制出站网络，只允许访问指定域名
        allowed_domains=["api.example.com"],
        # git 在沙箱外运行，保留完整网络权限
        excluded_commands=["git"],
    ),
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

使用 `@tool` 装饰器定义进程内工具，`create_sdk_mcp_server` 直接返回 `McpSdkServerConfig`：

```python
from claude_agent_sdk import create_sdk_mcp_server, tool
from cckit import Agent

@tool("fetch_data", "Fetch data from the platform", {"id": str})
async def fetch_data(args: dict) -> dict:
    result = await my_api.get(args["id"])
    return {"content": [{"type": "text", "text": result}]}

# create_sdk_mcp_server already returns McpSdkServerConfig, no need to wrap again
mcp_server = create_sdk_mcp_server("my-tools", tools=[fetch_data])

agent = Agent(
    name="my-agent",
    mcp_servers={"my-tools": mcp_server},
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

## Hooks / TaskBudget / ContextConfig

### Hooks（原生钩子）

`hooks` 透传 Claude SDK 原生钩子，拦截工具调用、压缩、通知等事件。支持 10 种事件：`PreToolUse`、`PostToolUse`、`PostToolUseFailure`、`UserPromptSubmit`、`Stop`、`SubagentStop`、`SubagentStart`、`PreCompact`、`Notification`、`PermissionRequest`。

```python
from cckit import Agent
from claude_agent_sdk import HookMatcher
from claude_agent_sdk.types import PreToolUseHookInput, SyncHookJSONOutput

async def block_rm(input: PreToolUseHookInput, tool_use_id, ctx) -> SyncHookJSONOutput:
    if "rm -rf" in input["tool_input"].get("command", ""):
        return {"continue_": False, "stopReason": "rm -rf is not allowed"}
    return {}  # 空 dict = 放行

agent = Agent(
    name="safe-agent",
    tools=["Bash"],
    hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[block_rm])]},
)
```

### TaskBudget（Token 预算）

声明 token 总预算，模型会在接近上限时主动收尾，避免硬截断。

```python
from cckit import Agent, TaskBudgetConfig

agent = Agent(
    name="budget-agent",
    task_budget=TaskBudgetConfig(total=50_000),  # 输入 + 输出 token 总预算
)
```

### ContextConfig（上下文窗口与自动压缩）

控制上下文窗口大小和自动压缩触发时机，适用于小上下文模型（如 8K）。每个 Agent 独立子进程，配置互不影响。

> **注意**：
> - `max_context_tokens` 是上下文窗口大小（输入+输出总量），与 `ModelConfig.max_tokens`（最大输出 token 数）不同，不要混淆。
> - `max_context_tokens` 必须 **大于 `max_tokens + 23000`**，否则自动压缩不会触发。例如 `max_tokens=8000` 时，`max_context_tokens` 至少设为 `31000`。
> - 设置 `max_tokens` 后，cckit 会自动将其同步注入给 Claude CLI，无需手动配置环境变量。

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `max_context_tokens` | `None`（使用模型原生窗口） | 上下文窗口大小 |
| `auto_compact_pct` | `80` | 使用率达到此百分比时触发压缩 |
| `disable_auto_compact` | `False` | 禁用自动压缩 |

```python
from cckit import Agent, ContextConfig, ModelConfig

# 8K 上下文模型 — 60% 时压缩
agent = Agent(
    name="small-model",
    model=ModelConfig(model="openai/gpt-4o-mini", max_tokens=4096),
    context=ContextConfig(max_context_tokens=8192, auto_compact_pct=60),
)

# 只调整压缩百分比，窗口大小用模型默认值
agent = Agent(
    name="early-compact",
    context=ContextConfig(auto_compact_pct=50),
)
```

## 示例

完整示例见 [`examples/`](examples/) 目录：

- **FixAgent** — 自动修复 + 创建 Merge Request
- **CodeModifyAgent** — 代码修改 Agent


## 已知问题
- [ ] `Windows`下，系统提示词设置无法生效， 属于`claude-agent-sdk`本身问题
