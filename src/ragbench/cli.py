"""CLI entry point for ragbench."""

import click

from . import __version__


@click.group()
@click.version_option(version=__version__)
def main():
    """ragbench - Lightweight RAG pipeline benchmarking CLI."""


@main.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="Path to YAML config file")
@click.option("--dataset", "-d", required=True, type=click.Path(exists=True),
              help="Path to dataset (JSONL format)")
@click.option("--output", "-o", default="results.json",
              help="Output file path")
@click.option("--format", "-f", "output_format", type=click.Choice(["json", "csv"]),
              default="json", help="Output format")
def eval(config, dataset, output, output_format):
    """Evaluate RAG pipeline performance on a dataset."""
    click.echo(f"Loading config: {config}")
    click.echo(f"Loading dataset: {dataset}")
    click.echo(f"Output format: {output_format}")
    click.echo(f"Results will be written to: {output}")


@main.command()
@click.option("--num-queries", "-n", default=100, help="Number of queries to generate")
@click.option("--num-docs", "-d", default=1000, help="Number of documents to generate")
@click.option("--output", "-o", default="dataset.jsonl", help="Output file path")
def generate(num_queries, num_docs, output):
    """Generate a synthetic RAG evaluation dataset."""
    click.echo(f"Generating {num_queries} queries across {num_docs} documents")
    click.echo(f"Writing to: {output}")


if __name__ == "__main__":
    main()
