# Skills 支持设计文档

> 为 Agent 核心基础设施添加 Claude Agent SDK 原生 Skills 支持。

## 1. 目标

让平台 Agent 能够声明式地使用 Skills —— 从外部下载的 SKILL.md 能力包放到 skills 目录下即可被 Agent 引用，实现即插即用。

## 2. 约束

- **遵循 SDK 原生机制**：不自建 SKILL.md 解析逻辑，依赖 `claude-agent-sdk` 的 `setting_sources` + `Skill` 工具完成发现和加载。
- **沙箱隔离**：Agent 运行在沙箱 workspace 内，skills 文件需部署到沙箱中才能被 SDK 发现。
- **遵循现有架构**：新子系统内聚在 `core/agent/skill/`，与 middleware/、mcp/、sandbox/ 平级。
- **SKILL.md 格式对齐官方规范**：必填 `name` + `description`，可选 `allowed-tools`、`license`、`compatibility`、`metadata` 等。

## 3. SKILL.md 格式

遵循 Claude Agent Skills 官方规范：

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

## Best Practices
- Always use `data-testid` selectors for stability
...
```

### 必填字段

| 字段 | 约束 |
|------|------|
| `name` | ≤64 字符，小写字母+数字+连字符，与目录名一致，禁用保留字 |
| `description` | ≤1024 字符，禁止 XML 标签，建议包含 `Use when:` 触发条件 |

### 可选字段

| 字段 | 作用 |
|------|------|
| `license` | 开源协议 |
| `compatibility` | 环境要求（≤500 字符）|
| `metadata` | 自定义键值对 |
| `allowed-tools` | 空格分隔的工具列表（SDK 原生字段，由 SDK 运行时处理，平台不合并不校验）|
| `disable-model-invocation` | 禁止 AI 自动调用 |
| `user-invocable` | 是否在斜杠命令菜单展示 |
| `argument-hint` | 斜杠命令参数提示 |

> **注意**：SKILL.md 中的 `allowed-tools` 是 SDK 原生字段，由 SDK 运行时自行处理。平台不对其与 `AgentConfig.allowed_tools` 做合并或校验。

## 4. 目录结构

### Skills 存放

```
new/
└── skills/                          # 默认位置，通过 PLATFORM_SKILLS_DIR 可配
    ├── playwright-testing/
    │   └── SKILL.md
    ├── api-testing/
    │   └── SKILL.md
    └── selenium-compat/
        └── SKILL.md
```

### 新增代码

```
src/core/agent/
└── skill/                           # 新子系统（与 middleware/、mcp/ 平级）
    ├── __init__.py                  # re-export SkillProvisioner
    └── provisioner.py               # 唯一模块：部署 skills 到沙箱
```

## 5. 核心设计

### 5.1 原理

SDK 通过 `setting_sources` 配置从文件系统自动发现 Skills。`"project"` 源从 `{cwd}/.claude/skills/` 加载，`"user"` 源从 `~/.claude/skills/` 加载。

平台使用 **`setting_sources=["project"]`（仅 project）**，原因：
- Agent 运行在沙箱 workspace 内，`cwd` = workspace 目录
- 只加载 `"project"` 源，确保 SDK 仅从沙箱内的 `.claude/skills/` 发现 skills
- **不使用 `"user"` 源**：避免加载宿主机 `~/.claude/skills/` 中未经审核的 skills，保证沙箱隔离

### 5.2 执行流程

```
Executor.execute_stream()
  ├── 1. validate context            （现有）
  │       └── 【新增】skills 非空时校验 needs_workspace=True
  ├── 2. on_before_execute           （现有）
  ├── 3. create workspace            （现有）
  ├── 4. git clone                   （现有）
  ├── 5.【新增】provision skills     ← SkillProvisioner 按 agent 声明的
  │       ├── 清除 workspace/.claude/     skills 列表，从源目录复制到
  │       │   中已有的 skills（防止        workspace/.claude/skills/ 下
  │       │   git clone 引入不可信内容）
  │       └── 复制平台审核的 skills
  ├── 6. build_prompt                （现有）
  └── 7. _build_options              （改动：加 setting_sources + "Skill"）
```

### 5.3 安全：处理 git clone 引入的 `.claude/` 内容

当 `needs_git_clone=True` 时，克隆的仓库可能自带 `.claude/` 目录（含 settings 或 skills）。`setting_sources=["project"]` 会让 SDK 加载这些内容，带来安全风险。

**处理策略**：在 provision 步骤中，如果 Agent 声明了 skills：
1. **删除** `workspace/.claude/` 目录（如果 git clone 带入了）
2. **重新创建** `workspace/.claude/skills/`，仅放入平台审核的 skills

这确保 SDK 只加载平台控制的 skills，不受仓库内容影响。

### 5.4 SkillProvisioner

职责单一：**从平台 skills 源目录按名称选取 skill 目录，安全地复制到沙箱工作空间的 `.claude/skills/` 下。**

```python
class SkillProvisioner:
    def __init__(self, skills_dir: Path): ...
    async def provision(self, skill_names: list[str], workspace_dir: Path) -> list[str]: ...
    def list_available(self) -> list[str]: ...
    def _validate_skill_name(self, skill_name: str) -> None: ...
    def _validate_skill_dir(self, skill_name: str) -> None: ...
```

- `provision()`：清除 workspace 已有 `.claude/`，复制指定 skills，返回已部署名称列表
- `list_available()`：列出源目录中所有可用 skill
- `_validate_skill_name()`：**安全校验**——正则 `^[a-z0-9][a-z0-9-]{0,62}$`，拒绝路径遍历
- `_validate_skill_dir()`：校验 skill 目录存在、包含 SKILL.md、解析真实路径确认在 skills_dir 下、拒绝符号链接

### 5.5 配置变更

`PlatformSettings` 新增：

```python
skills_dir: Path = Path("/opt/agent-platform/skills")  # PLATFORM_SKILLS_DIR 环境变量
```

使用绝对路径作为默认值，与 `SandboxSettings.workspace_root = Path("/tmp/agent_workspaces")` 保持一致。

当 skills 目录不存在时：日志 warning，视为无可用 skills，不阻止 Agent 运行。

### 5.6 Schema 变更

`AgentConfig` 新增：

```python
skills: list[str] = Field(default_factory=list)  # skill name 列表
```

### 5.7 Executor 变更

**构造函数**：新增可选的 `skill_provisioner` 参数，遵循现有 DI 模式：

```python
class AgentExecutor:
    def __init__(
        self,
        *,
        workspace_manager: WorkspaceManager | None = None,
        sandbox_config: SandboxConfig | None = None,
        skill_provisioner: SkillProvisioner | None = None,  # 新增
        middlewares: list[AgentMiddleware] | None = None,
        max_concurrent_clones: int | None = None,
    ) -> None:
        ...
        self._skill_provisioner = skill_provisioner or SkillProvisioner()
```

**`_validate_context`**：新增校验——skills 非空时 `needs_workspace` 必须为 `True`。

**`execute_stream`**：workspace 创建后、build_prompt 前，调用 `self._skill_provisioner.provision()`。

**`_build_options`**：当 skills 非空时，`allowed_tools` 追加 `"Skill"`，设置 `setting_sources=["project"]`。

### 5.8 异常

`exceptions.py` 新增：

```python
class SkillError(CoreError):
    """Raised when skill provisioning fails (not found, invalid, copy error)."""
```

### 5.9 清理

Skills 文件位于 `workspace/.claude/skills/` 内。当 `WorkspaceManager.cleanup()` 执行 `shutil.rmtree(workspace)` 时，skills 随 workspace 一同清理，无需额外逻辑。

### 5.10 安全总结

| 威胁 | 防御 |
|------|------|
| skill name 路径遍历（`../../etc`）| 正则校验 + resolve 真实路径确认在 skills_dir 内 |
| 符号链接攻击 | 拒绝符号链接目录 |
| git clone 注入 `.claude/` | provision 前删除 workspace 已有 `.claude/` |
| 宿主机 skills 泄漏 | `setting_sources=["project"]`，不加载 `"user"` 源 |
| SKILL.md 内容注入 | Skills 视为受信代码，仅部署平台审核的 skills |

## 6. Agent 使用方式

```python
@agent_registry.register
class TestAgent(BaseAgent):
    def config(self) -> AgentConfig:
        return AgentConfig(
            agent_type="test",
            skills=["playwright-testing"],
            allowed_tools=["Bash", "Read"],
            system_prompt="You are a test expert...",
        )

    def build_prompt(self, ctx: ExecutionContext) -> str:
        return f"Fix this test: {ctx.extra_params['test_file']}"
```

无需其他配置。Executor 自动：
1. 清除 workspace 中 git clone 可能带入的 `.claude/`
2. 将 `playwright-testing/` 复制到 `workspace/.claude/skills/`
3. 设置 `setting_sources=["project"]` + `allowed_tools` 追加 `"Skill"`
4. SDK 自动发现并加载 skill
5. Claude 按需触发

## 7. 改动清单

### 新建文件

| 文件 | 内容 |
|------|------|
| `core/agent/skill/__init__.py` | re-export `SkillProvisioner` |
| `core/agent/skill/provisioner.py` | `SkillProvisioner` 实现（含安全校验） |

### 修改现有文件（均为 feature 完整性的阻塞交付项）

| 文件 | 内容 |
|------|------|
| `core/config.py` | `PlatformSettings` 新增 `skills_dir` 字段 |
| `core/agent/schemas.py` | `AgentConfig` 新增 `skills` 字段 |
| `core/agent/executor.py` | 构造函数新增 `skill_provisioner` DI；`_validate_context` 新增 skills+workspace 校验；`execute_stream` 新增 provision 步骤；`_build_options` 新增 `setting_sources` + `"Skill"` |
| `core/exceptions.py` | 新增 `SkillError(CoreError)` 异常类 |
| `core/__init__.py` | `__all__` 新增 re-export `SkillError`（`SkillProvisioner` 为内部类，不 re-export）|
| `scripts/smoke_test.py` | 新增 skill provisioning 相关冒烟测试（目录扫描、安全校验、provision 复制、清理）|
| `AGENTS.md` | 补充 Skills 章节文档 |
