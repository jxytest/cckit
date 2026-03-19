# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是 **cckit**（Claude Agent Builder）—— 一个 pip 可安装的多 Agent SDK，基于 `claude-agent-sdk` 构建。

**核心三层设计**：
- `Agent` = "我是谁"（name、instruction、tools、sub_agents、skills、model）
- `RunContext` = "怎么跑"（workspace、git、prompt、params、env）
- `Runner` = "执行引擎"（中间件、并发控制、SDK 桥接）

## SDK 隔离原则

仅 3 个文件导入 `claude_agent_sdk`，其余均为纯 Python：

| 文件 | 导入内容 |
|------|----------|
| `cckit/_engine/sdk_bridge.py` | `query`, `ClaudeAgentOptions` |
| `cckit/_engine/collector.py` | `AssistantMessage`, `ResultMessage`, `SystemMessage`, `TextBlock`, `ThinkingBlock`, `ToolUseBlock`, `ToolResultBlock`, `TaskStartedMessage`, `TaskProgressMessage`, `TaskNotificationMessage` |
| `cckit/tools/platform.py` | `create_sdk_mcp_server`, `tool` |

所有导入都是惰性的（在函数内部）。`import cckit` 不触发 SDK 导入。

## Agent 定义与执行分离

- **`Agent`** — 声明式定义，实例化即可，不需要继承
- **`RunContext`** — 运行时上下文（workspace、git 等控制在此，不在 Agent）
- **`Runner`** — 执行引擎，管理 workspace/middleware/SDK 桥接

同一 Agent 定义可在不同 RunContext 中复用。业务凭证通过 `RunContext.params` 按任务传递。

## 配置系统

**无全局单例**。所有配置通过 `RunnerConfig` 显式传入 `Runner(config=...)`。
`RunnerConfig.from_env()` 是便捷方法，从 `ANTHROPIC_*`、`SANDBOX_*`、`PLATFORM_*` 环境变量读取。

子模块不再有全局 fallback：
- `WorkspaceManager(root: Path)` — root 必填
- `SkillProvisioner(skills_dir: Path)` — skills_dir 必填
- `ConcurrencyMiddleware(max_concurrent: int)` — 显式传入

## 会话恢复（Resume）

Agent 无状态，会话上下文通过 `RunContext.resume_session_id` 传入。SDK/CLI 自动持久化对话历史。

```
第一次执行: ctx1 = RunContext(
    prompt="修复 X",
    workspace=WorkspaceConfig(enabled=True),
    git=GitConfig(repo_url="...", token="glpat-xxx", clone=True),
)
  → AgentResult(session_id="abc-123")
  → workspace 被 suspend

第二次执行: ctx2 = RunContext(
    prompt="加单元测试",
    resume_session_id="abc-123",
    workspace_dir=ctx1.workspace_dir,
)
  → 跳过 create/clone/provision，SDK 自动恢复对话
```

## 执行流程

```
Runner.run_stream(agent, ctx):
  ├── _validate_context()（required_params、git.repo_url、skills+workspace）
  ├── agent.before_execute(ctx)
  ├── preflight_check（可选：API key + 网络连通性检测）
  ├── workspace:
  │   ├── resume → WorkspaceManager.resume()
  │   ├── workspace.enabled=True → create() + git clone（凭证隔离） + provision skills
  │   └── workspace.enabled=False → 跳过
  ├── resolve_instruction + _resolve_model
  ├── _build_options（ClaudeAgentOptions: 子 Agent、MCP、沙箱、skills、resume）
  │   └── env 只含 ANTHROPIC_* + ctx.env，不含 git 凭证
  ├── [中间件链] → SDK bridge → StreamCollector → yield AgentEvent
  ├── agent.after_execute / error_execute
  └── workspace cleanup / suspend
```

## 模块职责

| 文件 | 职责 |
|------|------|
| `agent.py` | Agent 类 — 声明式定义（name、instruction、tools、sub_agents、callbacks） |
| `runner.py` | Runner 类 — 唯一编排 SDK 调用的模块 |
| `types.py` | 纯数据模型：ModelConfig、LiteLlm、GitConfig、RunContext、WorkspaceConfig、AgentResult、AgentEvent、RunnerConfig |
| `exceptions.py` | CckitError 异常层次（含 ConnectivityError） |
| `_cli.py` | Claude CLI 检测 + API 连通性预检（`check_api_connectivity`） |
| `_engine/sdk_bridge.py` | SDK 交互：connect → receive_response → disconnect |
| `_engine/collector.py` | StreamCollector：SDK 消息 → AgentEvent |
| `middleware/` | 可插拔中间件链（Middleware 基类 + retry/concurrency/logging） |
| `sandbox/workspace.py` | WorkspaceManager（create/suspend/resume/cleanup） |
| `sandbox/config.py` | SandboxConfigBuilder → SDK sandbox dict |
| `skill/provisioner.py` | SkillProvisioner（安全复制 SKILL.md 到 workspace） |
| `git/operations.py` | 异步 git CLI 操作 |
| `git/gitlab_client.py` | GitLabClient（构造时必须传入 url/token） |
| `tools/platform.py` | 可选平台 MCP 工具 |

## 中间件链

第一个中间件是最外层包装。自定义中间件：继承 `Middleware`，实现 `async def wrap(self, next_call, prompt, options, collector, ctx)`。

## 模型支持

model 参数支持三种形式：
- `str`："claude-sonnet-4-6"
- `ModelConfig`：显式 api_key/base_url/max_tokens
- `LiteLlm`：通过 LiteLLM 代理桥接任意模型

Agent 内部统一归一化为 `ModelConfig`。Runner 的 `_resolve_model()` 合并 Agent 级与 Runner 默认配置。

## 多项目并发隔离

- **文件系统**：每个任务通过 `WorkspaceManager.create(task_id)` 获得独立临时目录
- **Git 凭证**：`GitConfig.build_git_env()` 仅注入 git 子进程，不污染 Agent 环境
- **Agent 环境**：`ctx.env` 仅传递给 Agent 子进程（API key、feature flags 等）
- **并发**：`asyncio.Semaphore` 防止资源耗尽
- **会话**：每次 `run_stream` 内的 SDK 连接在调用结束后释放

## Git 凭证隔离

`RunContext` 通过 `git: GitConfig` 管理所有 git 相关配置。凭证与 Agent 子进程环境**完全隔离**：

```python
ctx = RunContext(
    prompt="修复 Bug",
    env={"ANTHROPIC_API_KEY": "sk-..."},      # → Agent 子进程
    git=GitConfig(
        repo_url="https://gitlab.com/team/project.git",
        token="glpat-secret",                  # → 仅 git 子进程
        branch="main",
        clone=True,
        depth=1,
    ),
)
# git.token 不会出现在 Agent 的 Bash 环境中
# Agent 无法通过 `env` 命令读取 git 凭证
```

`GitConfig.build_git_env()` 是 git 凭证的**唯一出口**，被 Runner 内部的 clone/push 使用。
`on_after` 回调中 push 也应使用 `ctx._resolved_git().build_git_env()` 获取凭证。

## 编码约定

- Python >= 3.12，所有 I/O 使用 async
- datetime 必须 timezone-aware UTC（`datetime.now(UTC)`）
- 同步第三方 SDK（如 python-gitlab）通过 `loop.run_in_executor` 包装
- Pydantic 模型继承 `CustomModel`（`from_attributes=True`、`use_enum_values=True`）
- `GitLabClient` 构造时必须显式传入 `url`/`token`，无全局 fallback
- Ruff 格式化，行宽 100，目标 Python 3.12

## API 连通性预检（Preflight Check）

执行前可选检测 API key 有效性和网络连通性，避免等待 CLI 初始化超时才发现配置问题。

### 方式一：Runner 自动检测

每次执行前自动检测，适合生产环境：

```python
runner = Runner(preflight_check=True)

# 正常使用，如果 key 无效或网络不通会立即抛 ConnectivityError
result = await runner.run(agent, ctx)
```

### 方式二：手动调用

应用启动时检测一次，或在自定义流程中按需调用：

```python
from cckit import check_api_connectivity
from cckit.exceptions import ConnectivityError

# 使用环境变量中的 ANTHROPIC_API_KEY 和 ANTHROPIC_BASE_URL
try:
    check_api_connectivity()
except ConnectivityError as e:
    print(f"连接失败: {e}")

# 显式指定（适合代理接口场景）
check_api_connectivity(
    api_key="sk-...",
    base_url="https://your-proxy.example.com",
    model="your-model-name",
)
```

### 错误类型

| 场景 | 错误信息 |
|------|----------|
| 未设置 API key | `ANTHROPIC_API_KEY is not set` |
| key 无效 | `API key is invalid (HTTP 401)` |
| key 无权限 | `API key lacks permission (HTTP 403)` |
| 网络不通 | `Cannot reach API at {url}` |
| 连接超时 | `API connection timed out after {n}s` |

所有错误抛出 `ConnectivityError`（继承自 `CckitError`）。
