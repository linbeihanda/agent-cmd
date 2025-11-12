# Agent Commands - 智能化 Vibe Coding 命令集

这是一系列实用的命令工具，专为配合 Vibe Coding 开发流程而设计。项目通过 Claude Code Hooks 实现了 spec-kit 开发流程的自动化，核心原理是劫持 Claude Code 的 Stop Hooks，将返回信息发送给第三方大模型进行智能解析，自动判断下一步应该执行的 spec-kit 命令，然后自动恢复会话，实现全自动化的 spec-kit 流程开发。

## 使用方法

1. **安装 specify 命令**：首先确保系统中已安装 specify 命令工具

2. **初始化项目**：使用 specify 初始化一个 spec-kit + Claude Code 的项目目录
    ```bash
    specify init project-name
    cd project-name

    claude --dangerously-skip-permissions

    # 进入 Claude Code 后先把 constitution 这一步做了
    /speckit.constitution [你的constitution内容]

    然后连按2次 ctrl+C 退出 Claude Code
    ```

3. **下载本项目**：将本项目下载到该目录中
    ```bash
    git clone https://github.com/linbeihanda/agent-cmd.git
    cd agent-cmd
    ```

4. **安装依赖**：
    ```bash
    # 如果你使用 pyenv 等工具管理虚拟环境，可以 pyenv activate venv-project-name 进入，再安装
    pip install -r requirements.txt
    ```

5. **配置环境变量**：
   ```bash
   cp .env.example .env
   # 编辑 .env 文件，设置所需的环境变量

   ```

6. **返回工作目录**：
   ```bash
   cd ..
   ```

7. **启动 Claude Code**：
   ```bash
   claude --dangerously-skip-permissions
   ```

8. **设置 Hooks**：在 Claude Code 中使用 `/hooks` 命令设置 hooks，选择 "6. Stop" 类型，命令设置为：
   ```
   python agent-cmd/src/spec-kit-next-step.py
   # 如果你是使用虚拟环境，这里要用该 venv 的绝对路径，如 $HOME/.pyenv/versions/venv-project-name/bin/python
   ```

9. **开始开发**：现在可以按照 spec-kit 的流程进行开发了。当你使用 `specify` 命令定义一个需求之后，后续的 Hooks 会智能地触发下一步操作，直到达到设置的次数限制，实现真正的自动化开发流程。
