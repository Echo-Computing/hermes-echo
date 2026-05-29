"""Research agent nodes."""

from hermes_cli.agents.echo.research.nodes.supervisor import run_supervisor
from hermes_cli.agents.echo.research.nodes.generation import run_generation
from hermes_cli.agents.echo.research.nodes.reflection import run_reflection
from hermes_cli.agents.echo.research.nodes.proximity import run_proximity
from hermes_cli.agents.echo.research.nodes.evolution import run_evolution
from hermes_cli.agents.echo.research.nodes.ranking import run_ranking
from hermes_cli.agents.echo.research.nodes.code_execution import run_code_execution
from hermes_cli.agents.echo.research.nodes.consensus import run_consensus

__all__ = [
    "run_supervisor",
    "run_generation",
    "run_reflection",
    "run_proximity",
    "run_evolution",
    "run_ranking",
    "run_code_execution",
    "run_consensus",
]
