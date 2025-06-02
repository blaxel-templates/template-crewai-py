# Blaxel CrewAI Agent

<p align="center">
  <img src="https://blaxel.ai/logo.png" alt="Blaxel" width="200"/>
</p>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/CrewAI-powered-brightgreen.svg)](https://www.crewai.com/)
[![GPT-4](https://img.shields.io/badge/GPT--4-enabled-orange.svg)](https://openai.com/gpt-4)

</div>

A template implementation of a multi-agent system using CrewAI and GPT-4. This template demonstrates the power of CrewAI for orchestrating teams of AI agents that collaborate to solve complex tasks with specialized roles and advanced coordination capabilities.

## 📑 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Locally](#running-the-server-locally)
  - [Testing](#testing-your-agent)
  - [Deployment](#deploying-to-blaxel)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

## ✨ Features

- Multi-agent collaboration with specialized roles and responsibilities
- Hierarchical task delegation and coordination
- Tool integration support across multiple agents
- Streaming responses for real-time interaction monitoring
- Built on CrewAI for sophisticated agent orchestration and workflow management
- Customizable agent personas and capabilities
- Easy deployment and integration with Blaxel platform

## 🚀 Quick Start

For those who want to get up and running quickly:

```bash
# Clone the repository
git clone https://github.com/blaxel-ai/template-crewai-py.git

# Navigate to the project directory
cd template-crewai-py

# Install dependencies
uv sync

# Start the server
bl serve --hotreload

# In another terminal, test the agent crew
bl chat --local blaxel-agent
```

## 📋 Prerequisites

- **Python:** 3.10 or later
- **[UV](https://github.com/astral-sh/uv):** An extremely fast Python package and project manager, written in Rust
- **Blaxel Platform Setup:** Complete Blaxel setup by following the [quickstart guide](https://docs.blaxel.ai/Get-started#quickstart)
  - **[Blaxel CLI](https://docs.blaxel.ai/Get-started):** Ensure you have the Blaxel CLI installed. If not, install it globally:
    ```bash
    curl -fsSL https://raw.githubusercontent.com/blaxel-ai/toolkit/main/install.sh | BINDIR=/usr/local/bin sudo -E sh
    ```
  - **Blaxel login:** Login to Blaxel platform
    ```bash
    bl login YOUR-WORKSPACE
    ```

## 💻 Installation

**Clone the repository and install dependencies:**

```bash
git clone https://github.com/blaxel-ai/template-crewai-py.git
cd template-crewai-py
uv sync
```

## 🔧 Usage

### Running the Server Locally

Start the development server with hot reloading:

```bash
bl serve --hotreload
```

_Note:_ This command starts the server and enables hot reload so that changes to the source code are automatically reflected.

### Testing your agent

You can test your agent crew using the chat interface:

```bash
bl chat --local blaxel-agent
```

Or run it directly with specific input:

```bash
bl run agent blaxel-agent --local --data '{"input": "Research and write a comprehensive report on renewable energy trends"}'
```

### Deploying to Blaxel

When you are ready to deploy your application:

```bash
bl deploy
```

This command uses your code and the configuration files under the `.blaxel` directory to deploy your application.

## 📁 Project Structure

- **src/main.py** - Application entry point
- **src/agent.py** - Core agent implementation with CrewAI integration
- **src/crew/** - Crew configuration and agent definitions
  - **agents.py** - Individual agent definitions and roles
  - **tasks.py** - Task definitions and workflows
  - **tools.py** - Custom tools for agent use
- **src/server/** - Server implementation and routing
  - **router.py** - API route definitions
  - **middleware.py** - Request/response middleware
- **pyproject.toml** - UV package manager configuration
- **blaxel.toml** - Blaxel deployment configuration

## ❓ Troubleshooting

### Common Issues

1. **Blaxel Platform Issues**:
   - Ensure you're logged in to your workspace: `bl login MY-WORKSPACE`
   - Verify models are available: `bl get models`
   - Check that functions exist: `bl get functions`

2. **Agent Coordination Problems**:
   - Check crew configuration and agent role definitions
   - Verify task dependencies and execution order
   - Monitor agent communication logs

3. **Performance and Timeout Issues**:
   - Adjust max_iterations in crew configuration
   - Optimize task complexity and scope
   - Monitor resource usage during crew execution

4. **Tool Integration Failures**:
   - Verify tool permissions in Blaxel platform
   - Check tool compatibility with agent roles
   - Review tool execution logs for errors

5. **Memory and Context Management**:
   - Monitor crew memory usage with large tasks
   - Implement context summarization for long workflows
   - Optimize agent prompt engineering

For more help, please [submit an issue](https://github.com/blaxel-templates/template-crewai-py/issues) on GitHub.

## 👥 Contributing

Contributions are welcome! Here's how you can contribute:

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes:
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push** to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Submit** a Pull Request

Please make sure to update tests as appropriate and follow the code style of the project.

## 🆘 Support

If you need help with this template:

- [Submit an issue](https://github.com/blaxel-templates/template-crewai-py/issues) for bug reports or feature requests
- Visit the [Blaxel Documentation](https://docs.blaxel.ai) for platform guidance
- Check the [CrewAI Documentation](https://docs.crewai.com/) for framework-specific help
- Join our [Discord Community](https://discord.gg/G3NqzUPcHP) for real-time assistance

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
