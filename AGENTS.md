 # Agent Core Infrastructure Guide

> 本文档面向 AI 编程助手和后续开发者，帮助快速理解项目架构、核心概念和扩展方式。

## 1. 项目定位

这是一个 **多 Agent UI 自动化测试平台** 的核心基础设施层。基于 `claude-agent-sdk`（封装 Claude CLI 的 Python SDK）构建，提供：

- 声明式 Agent 定义（新 Agent 约 10-20 行代码）
- 子 Agent 声明式编排（Claude 运行时自主决策是否调用）
- MCP 工具作为平台能力扩展点
- 中间件链（重试、限流、日志等横切关注点）
- 沙箱级工作空间隔离
- 与 GitLab 的 git 操作和 MR 创建集成

**重要约束**：`core/` 层不依赖 FastAPI，是纯 async 接口。FastAPI 属于上层业务层。

## 2. 技术栈

| 组件 | 说明 |
|------|------|
| Python | >= 3.12 |
| claude-agent-sdk | 底层 subprocess 调 Claude CLI，封装了 MCP、流式解析、子 Agent 等 |
| pydantic / pydantic-settings | 数据模型 & 环境变量配置 |
| python-gitlab | GitLab API 客户端 |
| anyio | SDK 内部使用的 async 运行时 |

## 3. 目录结构

```
new/
├── .env.example                    # 核心环境变量示例（不含业务配置）
├── .gitignore                      # Git 忽略规则
├── pyproject.toml                  # 项目依赖
├── AGENTS.md                       # 本文档
├── rule.md                         # FastAPI 编码规范（供上层业务参考）
├── skills/                         # ====== 平台 Skills（即插即用）======
│   └── <skill-name>/              # 每个 skill 一个目录
│       └── SKILL.md               # Claude Agent Skills 官方格式
├── scripts/
│   └── smoke_test.py               # 冒烟测试（不需要 API Key）
└── src/
    ├── __init__.py
    ├── core/                        # ====== 公共基础设施 ======
    │   ├── __init__.py              # 统一 re-export 所有关键类
    │   ├── config.py                # Pydantic BaseSettings（核心 + 业务分离）
    │   ├── models.py                # CustomModel 基类
    │   ├── exceptions.py            # 异常层次（CoreError 为根）
    │   └── agent/                   # ====== Agent 完整运行时（所有子系统内聚于此）======
    │       ├── schemas.py           # 纯数据模型：AgentConfig, ExecutionContext, AgentResult, AgentEvent
    │       ├── base.py              # BaseAgent 抽象类（含 on_error 钩子、required_params 声明）
    │       ├── registry.py          # AgentRegistry + @register 装饰器
    │       ├── executor.py          # AgentExecutor —— 唯一接触 SDK 的模块
    │       ├── middleware/          # 中间件（独立目录，易扩展）
    │       │   ├── __init__.py      # re-export：AgentMiddleware, Retry, Concurrency, Logging
    │       │   ├── base.py          # AgentMiddleware 抽象基类
    │       │   ├── retry.py         # RetryMiddleware（指数退避）
    │       │   ├── concurrency.py   # ConcurrencyMiddleware（Semaphore 限流）
    │       │   └── logging.py       # LoggingMiddleware（耗时、费用记录）
    │       ├── sandbox/             # 工作空间隔离
    │       │   ├── workspace.py     # WorkspaceManager（文件系统生命周期）
    │       │   └── config.py        # SandboxConfig -> SDK sandbox dict
    │       ├── skill/               # Skills 沙箱部署
    │       │   └── provisioner.py   # SkillProvisioner（安全复制 skills 到 workspace）
    │       ├── mcp/                 # 平台 MCP 工具
    │       │   └── platform_tools.py
    │       ├── git/                 # Git 操作 + GitLab 集成
    │       │   ├── operations.py    # async git CLI（clone/branch/commit/push）
    │       │   └── gitlab_client.py # GitLabClient（支持构造器注入凭证）
    │       └── streaming/           # SDK 消息流式采集
    │           └── collector.py     # StreamCollector：SDK 消息 -> AgentEvent
    └── agents/                      # ====== 业务 Agent ======
        ├── __init__.py              # 自动注册（import 即触发）
        └── fix_agent.py             # 示例：FixAgent（修复 UI 定位器）
```

**设计原则**：`core/` 只放跨领域共享的公共设施（config、models、exceptions）。
所有 Agent 运行时的子系统（middleware、sandbox、skill、mcp、git、streaming）全部内聚在 `core/agent/` 包内。
后续扩展别的领域（如调度、通知）时，在 `core/` 下新建平级目录即可，不会和 Agent 的东西混在一起。

## 4. 核心架构

### 4.1 数据流

```
                    ┌─────────────┐
                    │ 调用方 (API) │
                    └──────┬──────┘
                           │ ExecutionContext
                           ▼
                    ┌─────────────┐
                    │AgentExecutor│  ← 唯一接触 SDK 的模块
                    └──────┬──────┘
                           │
         ┌─────────────────┼──────────────────┐
         ▼                 ▼                  ▼
  validate_context    workspace          build_prompt
         │            creation                │
         ▼                ▼                   ▼
  on_before_execute  ┌─────────┐    ┌──────────────┐
         │           │git clone│    │ClaudeAgentOpts│
         │           └────┬────┘    └──────┬───────┘
         │                ▼                │
         │         ┌──────────────┐        │
         │         │ provision    │        │
         │         │ skills →     │        │
         │         │ workspace/   │        │
         │         │ .claude/     │        │
         │         │ skills/      │        │
         │         └──────────────┘        │
         │                    ┌────────────────────┐
         │                    │  Middleware Chain   │
         │                    │ ├─ConcurrencyLimit │
         │                    │ ├─RetryBackoff     │
         │                    │ └─Logging          │
         │                    └────────┬───────────┘
         │                             │
         │                    ┌──────────────────┐
         │                    │ SDK query() 流式  │
         │                    └────────┬─────────┘
         │                             │ SDK Messages
         │                             ▼
         │                    ┌──────────────────┐
         │                    │ StreamCollector   │
         │                    └────────┬─────────┘
         │                             │ AgentEvent
         ▼                             ▼
  on_after_execute  ◄──────  AgentResult
  (or on_error)
         │
         ▼
   cleanup workspace
```

### 4.2 SDK 隔离原则

**只有三个文件导入 `claude_agent_sdk`**：

| 文件 | 导入内容 |
|------|----------|
| `core/agent/executor.py` | `query`, `ClaudeAgentOptions`, `AgentDefinition` |
| `core/agent/streaming/collector.py` | `AssistantMessage`, `ResultMessage`, `SystemMessage` |
| `core/agent/mcp/platform_tools.py` | `create_sdk_mcp_server`, `tool` |

其余所有模块为纯 Python，无 SDK 依赖。这意味着：
- 测试大部分逻辑时不需要安装 SDK
- 未来替换 SDK 只需改这三个文件

### 4.3 中间件架构

Executor 支持可插拔的中间件链，用于注入横切关注点：

```python
executor = AgentExecutor(
    middlewares=[
        ConcurrencyMiddleware(max_concurrent=5),   # 限流
        RetryMiddleware(max_retries=3),             # 重试
        LoggingMiddleware(),                        # 日志
    ],
)
```

中间件执行顺序：第一个是最外层包装，最后一个紧邻 SDK 调用。

**内置中间件**：

| 中间件 | 功能 |
|--------|------|
| `RetryMiddleware` | 指数退避重试，自动识别永久性错误（如 API Key 无效）不重试 |
| `ConcurrencyMiddleware` | asyncio.Semaphore 限制并发 Agent 数，默认读取 `PLATFORM_MAX_CONCURRENT_AGENTS` |
| `LoggingMiddleware` | 记录 SDK 查询的开始/结束、耗时和费用 |

**自定义中间件**：

新增中间件只需在 `middleware/` 下新建文件：

```python
# core/agent/middleware/audit.py
from core.agent.middleware.base import AgentMiddleware

class AuditMiddleware(AgentMiddleware):
    async def wrap(self, next_call, prompt, options, collector, ctx):
        logger.info("Audit: agent starting for task %s", ctx.task_id)
        async for event in next_call(prompt, options, collector):
            yield event
        logger.info("Audit: agent finished for task %s", ctx.task_id)
```

然后在 `middleware/__init__.py` 中添加 re-export 即可。

### 4.4 多项目并发隔离

多个项目（不同 GitLab token）同时执行时，通过三层机制保证互不干扰：

**1. 文件系统隔离**：每个任务的 `WorkspaceManager.create()` 使用 `tempfile.mkdtemp` 生成唯一目录，`task_id` 基于 UUID：
```
/tmp/agent_workspaces/
├── agent_3a7f1b2c9e01_xxxxx/   ← 项目 A (token-A)
├── agent_8d4e2f6a1b03_xxxxx/   ← 项目 B (token-B)
└── agent_c5b9e3d7f204_xxxxx/   ← 项目 A (另一个任务)
```

**2. 凭证隔离**：`git clone` / `git push` 支持 `extra_env` 参数，Executor 将 `ctx.env` 透传给 git 进程。每个子进程拥有独立环境变量，不污染全局 git config。业务层应通过以下方式之一传递凭证：
- 将 token 嵌入 URL：`https://oauth2:glpat-xxx@gitlab.com/repo.git`
- 通过 `ctx.env` 注入 `GIT_ASKPASS` 脚本

**3. clone 并发控制**：Executor 内置 `asyncio.Semaphore`（默认值取自 `PLATFORM_MAX_CONCURRENT_AGENTS`），防止大量任务同时 clone 耗尽网络/磁盘资源。可通过构造参数 `max_concurrent_clones` 覆盖。

> **注意**：`GitLabClient` 不再有全局 fallback。构造时必须显式传入 `url` 和 `token`。
> 业务层应将 GitLab 凭证放在 `ExecutionContext.extra_params` 中按项目绑定。

## 5. 如何定义新 Agent

### 5.1 最简示例（约 10 行）

在 `src/agents/` 下新建文件，例如 `src/agents/gen_agent.py`：

```python
from core.agent.base import BaseAgent
from core.agent.registry import agent_registry
from core.agent.schemas import AgentConfig, ExecutionContext


@agent_registry.register
class GenAgent(BaseAgent):
    def config(self) -> AgentConfig:
        return AgentConfig(
            agent_type="gen",
            display_name="Test Case Generator",
            system_prompt="You generate UI test cases from skill descriptions...",
            allowed_tools=["Read", "Write", "Glob", "Grep"],
        )

    def build_prompt(self, ctx: ExecutionContext) -> str:
        skill = ctx.extra_params.get("skill_description", "")
        return f"Generate test cases for this skill:\n{skill}"
```

无需其他配置 —— `agents/__init__.py` 会自动发现并注册。

### 5.2 带参数校验和子 Agent 的示例

```python
@agent_registry.register
class TestRunnerAgent(BaseAgent):
    def config(self) -> AgentConfig:
        return AgentConfig(
            agent_type="test_runner",
            display_name="Test Runner",
            system_prompt="You execute tests and coordinate sub-agents...",
            allowed_tools=["Bash", "Read", "Grep"],
            max_turns=80,  # 覆盖全局 max_turns（默认 0 = 使用全局值）
            sub_agents=[
                SubAgentConfig(
                    name="log-analyzer",
                    description="Analyze test logs for failure patterns",
                    tools=["Read", "Grep"],
                ),
                SubAgentConfig(
                    name="fix-suggester",
                    description="Suggest code fixes based on failures",
                    tools=["Read", "Grep"],
                ),
            ],
        )

    def required_params(self) -> list[str]:
        """声明 extra_params 中必须存在的键。Executor 在执行前校验。"""
        return ["test_suite", "branch"]
    ...
```

Executor 自动将 `SubAgentConfig` 映射为 SDK 的 `AgentDefinition`，并将 `Agent` 工具加入 allowed_tools。Claude 在运行时自主决定是否调用子 Agent。

### 5.3 Agent 生命周期钩子

```python
class MyAgent(BaseAgent):
    def required_params(self) -> list[str]:
        """声明必需的 extra_params 键（执行前自动校验）"""
        return ["test_file"]

    async def on_before_execute(self, ctx: ExecutionContext) -> None:
        """执行前回调：可用于写入 fixture 文件、获取外部数据等"""

    async def on_after_execute(self, ctx: ExecutionContext, result: AgentResult) -> None:
        """执行后回调：可用于创建 MR、上传产物、发送通知等"""

    async def on_error(self, ctx: ExecutionContext, error: Exception) -> None:
        """执行失败回调：可用于清理资源、发送告警等（在 on_after_execute 之前调用）"""
```

参考 `fix_agent.py` 中 `on_after_execute` 创建 GitLab MR 的实现。

## 6. 关键类速查

### 6.1 AgentConfig（静态声明 "我是谁"）

```python
class AgentConfig:
    agent_type: str          # 唯一标识，如 "fix", "gen"
    display_name: str        # 显示名称
    system_prompt: str       # Claude system prompt
    allowed_tools: list[str] # SDK 内置工具：Read, Write, Edit, Bash, Glob, Grep 等
    mcp_tool_names: list[str]# 需要的平台 MCP 工具名
    sub_agents: list[SubAgentConfig]  # 子 Agent 声明
    max_tokens: int          # 默认 16384
    max_turns: int           # 默认 0 = 使用全局 ANTHROPIC_MAX_TURNS（可按 Agent 覆盖）
    needs_workspace: bool    # 是否需要临时工作目录（默认 True）
    needs_git_clone: bool    # 是否需要克隆 git 仓库（默认 False）
    skills: list[str]        # 需要的 skill 名称列表（默认 []）
```

### 6.2 ExecutionContext（运行时 "环境是什么"）

```python
class ExecutionContext:
    task_id: str             # 自动生成的 12 位 hex ID
    workspace_dir: Path | None  # 由 Executor 注入，Agent 无需关心
    env: dict[str, str]      # 额外环境变量
    prompt: str              # 可选 prompt 覆盖
    extra_params: dict       # 业务参数（test_file, error_log 等）
    git_repo_url: str        # 要克隆的仓库地址
    git_branch: str          # 分支名
```

**设计要点**：`cwd`、`env` 不在 AgentConfig 中，而是通过 ExecutionContext 在运行时注入，实现同一 Agent 定义在不同上下文中复用。

### 6.3 AgentResult（执行结果）

```python
class AgentResult:
    task_id: str
    agent_type: str
    status: TaskStatus       # pending | running | completed | failed | cancelled
    result_text: str         # Claude 最终回复文本
    cost_usd: float          # API 消耗（美元）
    duration_seconds: float
    is_error: bool
    error_message: str
    session_id: str          # SDK session ID，可用于恢复对话
    extra: dict              # 钩子附加数据（如 mr_url, mr_error）
```

### 6.4 AgentEvent（流式事件）

```python
class AgentEvent:
    event_type: AgentEventType  # assistant_text | tool_use | tool_result | result | error
    task_id: str
    timestamp: datetime      # timezone-aware UTC
    data: dict               # 事件特定数据（如 tool_name, cost_usd）
    text: str                # 文本内容
```

## 7. 配置系统

配置通过环境变量注入，按域拆分为 Pydantic BaseSettings 类。`core/config.py` **只包含核心设施配置**：

| 类名 | 环境变量前缀 | 主要字段 |
|------|------------|----------|
| `AnthropicSettings` | `ANTHROPIC_` | `api_key`, `base_url`, `model`, `max_tokens`, `max_turns`, `permission_mode` |
| `SandboxSettings` | `SANDBOX_` | `enabled`, `workspace_root`, `network_allowed_hosts` |
| `PlatformSettings` | `PLATFORM_` | `debug`, `log_level`, `max_concurrent_agents`, `skills_dir` |

> **设计原则**：业务凭证（GitLab token、数据库连接等）不在 core 层定义。
> `GitLabClient` 要求构造时传入 `url`/`token`，业务层按项目绑定。
> Agent 通过 `ExecutionContext.extra_params` 传递业务凭证（如 `gitlab_url`、`gitlab_token`）。

参考 `.env.example` 了解核心环境变量列表。

示例 `.env`：
```
ANTHROPIC_API_KEY=sk-xxx
ANTHROPIC_BASE_URL=http://172.23.48.1:7000
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_MAX_TURNS=50
ANTHROPIC_PERMISSION_MODE=bypassPermissions
SANDBOX_ENABLED=false
```

## 8. MCP 平台工具扩展

MCP 工具是 Agent 调用平台能力的扩展点。在 `core/mcp/platform_tools.py` 中定义：

```python
@tool("my_tool_name", "Tool description for Claude", {"param1": str, "param2": int})
async def my_tool(args):
    result = await call_platform_api(args["param1"])
    return {"content": [{"type": "text", "text": result}]}
```

Agent 通过 `mcp_tool_names=["my_tool_name"]` 声明需要哪些工具。Executor 自动注入 MCP server。

**当前已定义的平台工具**：
- `get_failure_info` — 获取测试用例失败详情
- `write_test_cases` — 将生成的测试用例写回平台
- `report_progress` — 上报 Agent 执行进度

> 这些工具的实现目前是 placeholder，需要在业务层接入真实的平台 API。

## 9. Skills 即插即用能力包

Skills 是 Agent 可声明使用的可复用能力包，遵循 Claude Agent Skills 官方 SKILL.md 格式。从 skills 市场下载后放入 `skills/` 目录即可被 Agent 引用。

### 9.1 SKILL.md 格式

```markdown
---
name: playwright-testing
description: >
  Expert knowledge for UI test automation with Playwright.
  Use when: Agent needs to write, fix, or analyze Playwright-based UI tests.
allowed-tools: Bash Read Write Edit Glob Grep
metadata:
  author: test-platform-team
  version: "1.0.0"
---

You are an expert in Playwright-based UI test automation.
...
```

必填字段：`name`（≤64 字符，小写+数字+连字符）、`description`（≤1024 字符）。
可选字段：`allowed-tools`、`license`、`compatibility`、`metadata` 等。

### 9.2 Agent 使用 Skills

在 `config()` 中声明 `skills` 列表即可：

```python
@agent_registry.register
class TestAgent(BaseAgent):
    def config(self) -> AgentConfig:
        return AgentConfig(
            agent_type="test",
            skills=["playwright-testing", "api-testing"],
            allowed_tools=["Bash", "Read"],
            system_prompt="You are a test expert...",
        )

    def build_prompt(self, ctx: ExecutionContext) -> str:
        return f"Fix: {ctx.extra_params['test_file']}"
```

Executor 自动处理剩余工作：
1. 将 skills 复制到沙箱 `workspace/.claude/skills/`
2. 设置 `setting_sources=["project"]` + `allowed_tools` 追加 `"Skill"`
3. SDK 自动发现并加载 skill，Claude 按需触发

### 9.3 安全机制

| 威胁 | 防御 |
|------|------|
| skill name 路径遍历 | 正则校验 `^[a-z0-9][a-z0-9-]{0,62}$` + 真实路径确认 |
| 符号链接攻击 | 拒绝符号链接目录 |
| git clone 注入 `.claude/` | provision 前删除 workspace 已有 `.claude/` |
| 宿主机 skills 泄漏 | 仅使用 `setting_sources=["project"]`，不加载 `"user"` 源 |

### 9.4 配置

Skills 源目录通过 `PLATFORM_SKILLS_DIR` 环境变量配置，默认 `/opt/agent-platform/skills`。

## 10. 异常体系

```
CoreError                      # 根异常
├── AgentNotFoundError         # agent_type 未注册
├── AgentExecutionError        # SDK 调用失败、超时等
├── WorkspaceError             # 工作目录创建/清理失败
├── SandboxConfigError         # 沙箱配置无效
├── SkillError                 # skill 未找到、校验失败、复制失败
├── GitOperationError          # git CLI 命令失败
└── GitLabAPIError             # GitLab API 调用失败
```

所有异常支持 `detail` 关键字参数，用于传递底层错误信息。

## 11. 如何调用 Agent（上层业务集成示例）

### 10.1 基本用法

```python
import asyncio
from core.agent.executor import AgentExecutor
from core.agent.registry import agent_registry
from core.agent.schemas import ExecutionContext
import agents  # 触发自动注册

async def run_fix():
    executor = AgentExecutor()
    agent = agent_registry.get_instance("fix")
    ctx = ExecutionContext(
        git_repo_url="https://gitlab.com/team/ui-tests.git",
        git_branch="main",
        extra_params={
            "test_file": "tests/test_login.py",
            "error_log": "NoSuchElementException: id=login-btn",
            "project_id": "42",
            "gitlab_project_id": 123,
        },
    )

    async for event in executor.execute_stream(agent, ctx):
        print(f"[{event.event_type}] {event.text[:100] if event.text else event.data}")

asyncio.run(run_fix())
```

### 10.2 带中间件（推荐生产使用）

```python
from core.agent.middleware import RetryMiddleware, ConcurrencyMiddleware, LoggingMiddleware

executor = AgentExecutor(
    middlewares=[
        ConcurrencyMiddleware(),    # 默认读取 PLATFORM_MAX_CONCURRENT_AGENTS
        RetryMiddleware(max_retries=3, base_delay=1.0),
        LoggingMiddleware(),
    ],
)
```

## 12. 验证方式

```bash
# 冒烟测试（不需要 API Key，验证所有模块 import + 注册 + schema + workspace + 中间件）
cd new && python scripts/smoke_test.py

# 检查代码风格
cd new && ruff check src/
```

## 13. 后续扩展路线

| 方向 | 说明 | 涉及模块 |
|------|------|----------|
| 新增 Agent | 在 `agents/` 下新建文件，实现 `config()` + `build_prompt()` | `agents/` |
| 新增 Skill | 在 `skills/` 下新建目录 + `SKILL.md`，Agent 声明 `skills=["name"]` | `skills/` |
| 新增 MCP 工具 | 在 `platform_tools.py` 添加 `@tool` 函数 | `core/mcp/` |
| 自定义中间件 | 在 `middleware/` 下新建文件，继承 `AgentMiddleware`，实现 `wrap()` | `core/agent/middleware/` |
| FastAPI 集成 | 创建 `api/` 层，调用 `AgentExecutor.execute_stream()` + SSE 推送 | 上层业务 |
| WebSocket 推送 | 在 `on_after_execute` 或 MCP `report_progress` 中推送事件 | `core/mcp/` |
| 并行 Agent | 在业务层 `asyncio.gather` 多个 `execute_stream`（ConcurrencyMiddleware 自动限流） | 上层业务 |
| 沙箱启用 | 设置 `SANDBOX_ENABLED=true` + 安装 `@anthropic-ai/sandbox-runtime` | `core/sandbox/` |
| 数据库持久化 | 在 `on_after_execute` 中存储 `AgentResult` | 上层业务 |
| 会话恢复 | 利用 `AgentResult.session_id` 恢复中断的 Agent 对话 | 待实现 |

## 14. 编码规范

遵循 `rule.md` 中的 FastAPI 最佳实践：
- Pydantic BaseSettings 按域拆分
- 自定义 BaseModel（`CustomModel`）统一序列化行为
- 同步 SDK 在线程池中运行（见 `gitlab_client.py` 的 `run_in_executor`）
- Ruff 格式化，行宽 100，目标 Python 3.12
- 所有 I/O 操作使用 async
- 所有 datetime 使用 timezone-aware UTC（`datetime.now(timezone.utc)`）
