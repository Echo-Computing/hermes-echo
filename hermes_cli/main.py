"""Main CLI entry point for Hermes"""

import click
from hermes_cli import __version__
from hermes_cli.commands import init, task, run, log, history, echo


@click.group()
@click.version_option(version=__version__)
def cli():
    """Hermes - ローカルLLM情報収集エージェント"""
    pass


# コマンド登録
cli.add_command(init)
cli.add_command(task)
cli.add_command(run)
cli.add_command(log)
cli.add_command(history)
cli.add_command(echo)


if __name__ == "__main__":
    cli()
