# Examples

Usage examples for the `clab` SDK.

## Fix Agent

Repairs broken UI test locators, with optional GitLab MR creation.

```bash
python examples/fix_agent.py
```

## Code Modify Agent

Applies code modifications per user instructions, with optional MR.

```bash
python examples/code_modify_agent.py
```

## Prerequisites

1. Install clab: `pip install -e .`
2. Install Claude CLI: `npm install -g @anthropic-ai/claude-code`
3. Login: `claude login`
4. Set environment variables (or pass via `RunnerConfig`):
   ```
   ANTHROPIC_API_KEY=sk-xxx
   ANTHROPIC_BASE_URL=http://172.23.48.1:7000  # optional
   ```
