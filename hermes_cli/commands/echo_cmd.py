"""Echo agent CLI command - interactive chat with local LLM."""

import click
from loguru import logger

from hermes_cli.agents.echo.agent import create_echo_graph
from hermes_cli.agents.echo.state import EchoState
from hermes_cli.persistence.config_repository import ConfigRepository


@click.command()
@click.option("--model", "-m", help="Override the default model")
@click.option("--prompt", "-p", help="Single prompt mode (non-interactive)")
@click.option("--yes", "-y", is_flag=True, help="Skip destructive command confirmations")
@click.option("--config", "show_config", is_flag=True, help="Show Echo configuration and exit")
def echo(model, prompt, yes, show_config):
    """Launch the Echo agent in interactive chat mode."""
    config_repo = ConfigRepository()
    hermes_config = config_repo.load()

    if show_config:
        click.echo("Echo Agent Configuration:")
        click.echo(f"  Model: {hermes_config.echo.model}")
        click.echo(f"  Max tool calls: {hermes_config.echo.max_tool_calls}")
        click.echo(f"  Context messages: {hermes_config.echo.context_messages}")
        click.echo(f"  Shell timeout: {hermes_config.echo.shell_timeout}s")
        click.echo(f"  Memory dir: {hermes_config.echo.memory_dir}")
        click.echo(f"  Confirm destructive: {hermes_config.echo.confirm_destructive}")
        click.echo(f"  Auto memory: {hermes_config.echo.auto_memory}")
        return

    agent_model = model or hermes_config.echo.model

    click.echo("")
    click.echo("  Hermes Echo Agent")
    click.echo(f"  Model: {agent_model} (local GPU via Ollama)")
    click.echo(f"  Memory: {hermes_config.echo.memory_dir}")
    click.echo("  Type /help for commands, /exit to quit")
    click.echo("")

    graph = create_echo_graph()

    state = EchoState(
        config={
            "api_url": hermes_config.ollama.api_url,
            "model": agent_model,
            "max_tokens": hermes_config.ollama.max_tokens,
            "temperature": hermes_config.ollama.temperature,
            "max_tool_calls": hermes_config.echo.max_tool_calls,
            "context_messages": hermes_config.echo.context_messages,
            "shell_timeout": hermes_config.echo.shell_timeout,
            "confirm_destructive": hermes_config.echo.confirm_destructive and not yes,
            "memory_dir": str(hermes_config.echo.memory_dir),
        },
        messages=[],
    )

    if prompt:
        state["user_input"] = prompt
        try:
            final_state = graph.invoke(state)
            response = final_state.get("response", "(no response)")
            click.echo("")
            click.echo(f"  {response}")
            click.echo("")
        except Exception as e:
            logger.error(f"Agent error: {e}")
            click.echo(f"Error: {e}")
        return

    while True:
        try:
            user_input = click.prompt("you", prompt_suffix=" > ")
        except (EOFError, KeyboardInterrupt):
            click.echo("")
            click.echo("Goodbye.")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input == "/exit":
            click.echo("Session saved. Goodbye.")
            break
        elif user_input == "/help":
            click.echo("Session commands: /help, /clear, /model, /exit")
            continue
        elif user_input == "/clear":
            state["messages"] = []
            click.echo("Context cleared.")
            continue
        elif user_input == "/model":
            click.echo(f"Model: {agent_model}")
            continue

        state["user_input"] = user_input

        try:
            final_state = graph.invoke(state)
            response = final_state.get("response", "(no response)")
            click.echo("")
            click.echo(f"  hermes > {response}")
            click.echo("")
            state["messages"] = final_state.get("messages", [])
        except Exception as e:
            logger.error(f"Agent error: {e}")
            click.echo(f"Error: {e}")
            click.echo("")

