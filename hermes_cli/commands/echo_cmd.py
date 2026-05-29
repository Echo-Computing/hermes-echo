"""Echo agent CLI command - interactive chat and collaborative research with local LLM."""

import asyncio
import click
from loguru import logger

from hermes_cli.agents.echo.agent import create_echo_graph
from hermes_cli.agents.echo.state import EchoState
from hermes_cli.agents.echo.learning.detector import detect_command
from hermes_cli.persistence.config_repository import ConfigRepository


@click.command()
@click.option("--model", "-m", help="Override the default model")
@click.option("--prompt", "-p", help="Single prompt mode (non-interactive)")
@click.option("--yes", "-y", is_flag=True, help="Skip destructive command confirmations")
@click.option("--config", "show_config", is_flag=True, help="Show Echo configuration and exit")
@click.option("--research", "-r", "research_prompt", default=None, help="Collaborative multi-agent research mode")
@click.option("--rounds", "research_rounds", type=int, default=None, help="Max research rounds (override config)")
@click.option("--debates", "research_debates", type=int, default=None, help="Max debates per round (override config)")
def echo(model, prompt, yes, show_config, research_prompt, research_rounds, research_debates):
    """Launch the Echo agent in interactive chat or collaborative research mode."""
    config_repo = ConfigRepository()
    hermes_config = config_repo.load()

    # --- Research mode ---
    if research_prompt:
        return _run_research(
            hermes_config,
            research_prompt,
            model=model,
            rounds_override=research_rounds,
            debates_override=research_debates,
        )

    if show_config:
        click.echo("Echo Agent Configuration:")
        click.echo("  Model: {}".format(hermes_config.echo.model))
        click.echo("  Max tool calls: {}".format(hermes_config.echo.max_tool_calls))
        click.echo("  Context messages: {}".format(hermes_config.echo.context_messages))
        click.echo("  Shell timeout: {}s".format(hermes_config.echo.shell_timeout))
        click.echo("  Memory dir: {}".format(hermes_config.echo.memory_dir))
        click.echo("  History dir: {}".format(hermes_config.echo.history_dir))
        click.echo("  Confirm destructive: {}".format(hermes_config.echo.confirm_destructive))
        click.echo("  Auto memory: {}".format(hermes_config.echo.auto_memory))
        click.echo("  Learning enabled: {}".format(hermes_config.echo.learning.enabled))
        click.echo("")
        click.echo("Research Configuration:")
        click.echo("  Max rounds: {}".format(hermes_config.echo.research.max_rounds))
        click.echo("  Debates per round: {}".format(hermes_config.echo.research.debates_per_round))
        click.echo("  Hypotheses per round: {}".format(hermes_config.echo.research.hypotheses_per_round))
        click.echo("  Parallel instances: {}".format(hermes_config.echo.research.parallel_instances))
        click.echo("  Code timeout: {}s".format(hermes_config.echo.research.code_timeout))
        return

    agent_model = model or hermes_config.echo.model

    click.echo("")
    click.echo("  Hermes Echo Agent")
    click.echo("  Model: {} (via Ollama)".format(agent_model))
    click.echo("  Memory: {}".format(hermes_config.echo.memory_dir))
    click.echo("  Type /help for commands, /exit to quit")
    click.echo("  Use --research for multi-agent deep research")
    click.echo("")

    graph = create_echo_graph()

    # Build learning config dict from EchoConfig
    learning_config = {
        "enabled": hermes_config.echo.learning.enabled,
        "auto_memory": hermes_config.echo.learning.auto_memory,
        "auto_memory_max_per_session": hermes_config.echo.learning.auto_memory_max_per_session,
        "correction_reflection": hermes_config.echo.learning.correction_reflection,
        "session_summary": hermes_config.echo.learning.session_summary,
        "history_search": hermes_config.echo.learning.history_search,
        "history_search_limit": hermes_config.echo.learning.history_search_limit,
    }

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
            "history_dir": str(hermes_config.echo.history_dir),
            "learning": learning_config,
        },
        messages=[],
    )

    if prompt:
        state["user_input"] = prompt
        try:
            final_state = graph.invoke(state)
            response = final_state.get("response", "(no response)")
            click.echo("")
            click.echo("  {}".format(response))
            click.echo("")
        except Exception as e:
            logger.error("Agent error: {}".format(e))
            click.echo("Error: {}".format(e))
        return

    # Track ideation state in the CLI loop
    idea_active = False
    idea_start_index = 0

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

        # Handle built-in commands
        cmd = detect_command(user_input)

        if cmd["command"] == "exit":
            # Trigger session summary before exit
            state["pending_session_action"] = "summarize"
            state["user_input"] = user_input
            try:
                final_state = graph.invoke(state)
                response = final_state.get("response", "Session saved. Goodbye.")
                click.echo("")
                click.echo("  hermes > {}".format(response))
                click.echo("")
            except Exception as e:
                logger.error("Agent error during exit: {}".format(e))
            click.echo("Session saved. Goodbye.")
            break

        elif cmd["command"] == "idea":
            arg = cmd["arg"]
            state["user_input"] = user_input
            state["idea_active"] = True
            state["idea_start_index"] = len(state.get("messages", []))
            state["pending_idea"] = None
            idea_active = True
            idea_start_index = len(state.get("messages", []))

            try:
                final_state = graph.invoke(state)
                response = final_state.get("response", "(no response)")
                click.echo("")
                click.echo("  hermes > {}".format(response))
                click.echo("")
                state["messages"] = final_state.get("messages", [])
                state["idea_active"] = final_state.get("idea_active", True)
            except Exception as e:
                logger.error("Agent error: {}".format(e))
                click.echo("Error: {}".format(e))
                click.echo("")

        elif cmd["command"] == "idea_save":
            state["user_input"] = user_input
            state["pending_idea"] = state.get("pending_idea", "Project idea")
            state["idea_active"] = True
            state["idea_start_index"] = idea_start_index

            try:
                final_state = graph.invoke(state)
                response = final_state.get("response", "(no response)")
                click.echo("")
                click.echo("  hermes > {}".format(response))
                click.echo("")
                state["messages"] = final_state.get("messages", [])
                idea_active = False
                idea_start_index = 0
            except Exception as e:
                logger.error("Agent error: {}".format(e))
                click.echo("Error: {}".format(e))
                click.echo("")

        elif user_input == "/help":
            click.echo("Session commands:")
            click.echo("  /help       Show this help")
            click.echo("  /clear      Clear conversation context")
            click.echo("  /model      Show current model")
            click.echo("  /idea <text>  Explore a project idea with the agent")
            click.echo("  /idea save  Save the current idea exploration")
            click.echo("  /history <query> Search past session transcripts")
            click.echo("  /exit       Save session summary and exit")
            continue

        elif user_input == "/clear":
            state["messages"] = []
            idea_active = False
            idea_start_index = 0
            click.echo("Context cleared.")
            continue

        elif user_input == "/model":
            click.echo("Model: {}".format(agent_model))
            continue

        elif user_input.startswith("/history"):
            from hermes_cli.agents.echo.tools.search_tools import search_history
            query = user_input[len("/history"):].strip() or ""
            result = search_history(query) if query else "Usage: /history <search term>"
            click.echo("")
            click.echo("  {}".format(result))
            click.echo("")
            continue

        else:
            # Normal message — maintain ideation state
            state["user_input"] = user_input
            if idea_active:
                state["idea_active"] = True
                state["idea_start_index"] = idea_start_index

            try:
                final_state = graph.invoke(state)
                response = final_state.get("response", "(no response)")
                click.echo("")
                click.echo("  hermes > {}".format(response))
                click.echo("")
                state["messages"] = final_state.get("messages", [])
                idea_active = final_state.get("idea_active", False)
                idea_start_index = final_state.get("idea_start_index", 0)
            except Exception as e:
                logger.error("Agent error: {}".format(e))
                click.echo("Error: {}".format(e))
                click.echo("")


def _run_research(hermes_config, research_prompt, model=None, rounds_override=None, debates_override=None):
    """Run collaborative multi-agent research mode."""
    from hermes_cli.agents.echo.research.graph import create_research_graph
    from pathlib import Path
    from datetime import datetime

    agent_model = model or hermes_config.echo.model

    click.echo("")
    click.echo("  Hermes Echo — Collaborative Research Mode")
    click.echo("  Architecture: Co-Scientist + Robin (multi-agent)")
    click.echo("  Model: {}".format(agent_model))
    click.echo("  Goal: {}".format(research_prompt))
    click.echo("")

    research_config = {
        "max_rounds": rounds_override if rounds_override is not None else hermes_config.echo.research.max_rounds,
        "debates_per_round": debates_override if debates_override is not None else hermes_config.echo.research.debates_per_round,
        "hypotheses_per_round": hermes_config.echo.research.hypotheses_per_round,
        "parallel_instances": hermes_config.echo.research.parallel_instances,
        "code_timeout": hermes_config.echo.research.code_timeout,
        "search_results_per_query": hermes_config.echo.research.search_results_per_query,
    }

    initial_state = {
        "research_goal": research_prompt,
        "config": {
            "ollama": {
                "api_url": hermes_config.ollama.api_url,
                "model": agent_model,
                "timeout": hermes_config.ollama.timeout,
                "retry": hermes_config.ollama.retry,
            },
            "research": research_config,
        },
        "hypotheses": [],
        "search_results": [],
        "tournament_results": [],
        "code_execution_results": [],
        "errors": [],
    }

    graph = create_research_graph()
    start_time = datetime.now()

    try:
        click.echo("  Starting research...")
        click.echo("  Rounds: {}, Debates/round: {}, Hypotheses/round: {}".format(
            research_config["max_rounds"],
            research_config["debates_per_round"],
            research_config["hypotheses_per_round"],
        ))
        click.echo("")

        # Run the graph (async)
        result = asyncio.run(graph.ainvoke(initial_state, {"recursion_limit": 100}))

        final_report = result.get("final_report", {})
        hypotheses = result.get("hypotheses", [])
        errors = result.get("errors", [])

        # Display results
        click.echo("=" * 60)
        click.echo("  RESEARCH COMPLETE")
        click.echo("=" * 60)
        click.echo("")

        leaderboard = final_report.get("leaderboard", [])
        if leaderboard:
            click.echo("  TOP HYPOTHESES (ELO Leaderboard):")
            click.echo("  ─" * 30)
            for entry in leaderboard[:5]:
                click.echo("  #{rank}. [{elo:.0f} ELO] {title}".format(
                    rank=entry.get("rank", "?"),
                    elo=entry.get("elo_rating", 1500),
                    title=entry.get("title", "")[:80],
                ))
                click.echo("     {}".format(entry.get("description", "")[:120]))
                click.echo("")

        tourney = final_report.get("tournament_summary", {})
        if tourney:
            click.echo("  TOURNAMENT: {} debates | {} surviving | Top ELO: {:.0f}".format(
                tourney.get("total_debates", 0),
                final_report.get("surviving_hypotheses", 0),
                tourney.get("top_elo", 1500),
            ))

        validation = final_report.get("experimental_validation", {})
        if validation.get("experiments_run", 0) > 0:
            click.echo("  EXPERIMENTS: {} run | {} with consensus | {} accepted".format(
                validation.get("experiments_run", 0),
                validation.get("experiments_with_consensus", 0),
                validation.get("accepted_findings", 0),
            ))

        if errors:
            click.echo("")
            click.echo("  Errors encountered: {}".format(len(errors)))

        # Save report to history
        duration = (datetime.now() - start_time).total_seconds()
        history_dir = Path.home() / ".hermes" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        import json
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        report_path = history_dir / "research-{}.json".format(timestamp)

        report_data = {
            "goal": research_prompt,
            "model": agent_model,
            "duration_seconds": duration,
            "config": research_config,
            "report": final_report,
            "errors": errors,
        }
        report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False), encoding="utf-8")

        # Also save as markdown summary
        md_path = history_dir / "research-{}.md".format(timestamp)
        md_content = _format_markdown_report(report_data)
        md_path.write_text(md_content, encoding="utf-8")

        click.echo("")
        click.echo("  Duration: {:.1f}s".format(duration))
        click.echo("  Report saved: {}".format(md_path))
        click.echo("  JSON saved: {}".format(report_path))
        click.echo("")

    except Exception as e:
        import traceback
        logger.error("Research error: {}".format(e))
        logger.error("Traceback: {}".format(traceback.format_exc()))
        click.echo("")
        click.echo("  Research failed: {}".format(e))
        click.echo("")


def _format_markdown_report(data: dict) -> str:
    """Format research results as markdown."""
    goal = data.get("goal", "")
    report = data.get("report", {})
    config = data.get("config", {})
    duration = data.get("duration_seconds", 0)

    lines = [
        "# Research Report: {}".format(goal),
        "",
        "**Model:** {} | **Duration:** {:.1f}s | **Rounds:** {}".format(
            data.get("model", ""),
            duration,
            report.get("rounds_completed", 0),
        ),
        "",
        "---",
        "",
        "## Configuration",
        "- Max rounds: {}".format(config.get("max_rounds", 3)),
        "- Debates per round: {}".format(config.get("debates_per_round", 10)),
        "- Hypotheses per round: {}".format(config.get("hypotheses_per_round", 5)),
        "- Parallel instances: {}".format(config.get("parallel_instances", 3)),
        "",
        "## Summary",
        "- {} total hypotheses generated".format(report.get("total_hypotheses", 0)),
        "- {} survived to final round".format(report.get("surviving_hypotheses", 0)),
        "- {} eliminated".format(report.get("eliminated_hypotheses", 0)),
        "- {} merged into stronger hypotheses".format(report.get("merged_hypotheses", 0)),
        "",
    ]

    tourney = report.get("tournament_summary", {})
    if tourney:
        lines.extend([
            "## Tournament Results",
            "- {} head-to-head debates".format(tourney.get("total_debates", 0)),
            "- Top hypothesis: **{}** ({:.0f} ELO)".format(
                tourney.get("top_hypothesis", "None"),
                tourney.get("top_elo", 1500),
            ),
            "",
        ])

    validation = report.get("experimental_validation", {})
    if validation.get("experiments_run", 0) > 0:
        lines.extend([
            "## Experimental Validation",
            "- {} experiments run".format(validation.get("experiments_run", 0)),
            "- {} reached consensus".format(validation.get("experiments_with_consensus", 0)),
            "- {} accepted findings".format(validation.get("accepted_findings", 0)),
            "",
        ])

    leaderboard = report.get("leaderboard", [])
    if leaderboard:
        lines.append("## Hypothesis Leaderboard")
        lines.append("")
        for entry in leaderboard:
            lines.append("### {}. {} ({:.0f} ELO)".format(
                entry.get("rank", "?"),
                entry.get("title", ""),
                entry.get("elo_rating", 1500),
            ))
            lines.append("")
            lines.append(entry.get("description", ""))
            lines.append("")
            lines.append("**Mechanism:** {}".format(entry.get("mechanism", "Not specified")))
            lines.append("")
            if entry.get("evidence"):
                lines.append("**Evidence:**")
                for ev in entry["evidence"]:
                    lines.append("- {}".format(ev))
                lines.append("")

    errors = data.get("errors", [])
    if errors:
        lines.append("## Errors")
        for err in errors:
            lines.append("- {}".format(err))
        lines.append("")

    return "\n".join(lines)
