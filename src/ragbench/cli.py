"""CLI entry point for ragbench."""

import click

from . import __version__
from .config import load_config
from .dataset import load_jsonl, generate_synthetic_dataset, save_jsonl
from .evaluator import run_evaluation, save_results


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
    eval_config = load_config(config)
    queries = load_jsonl(dataset)
    results = run_evaluation(eval_config, queries)
    save_results(results, output, output_format)

    click.echo("Evaluation complete.")
    for metric, stats in results["aggregates"].items():
        click.echo(f"  {metric}: mean={stats['mean']:.4f}, "
                   f"min={stats['min']:.4f}, max={stats['max']:.4f}")


@main.command()
@click.option("--num-queries", "-n", default=100, help="Number of queries to generate")
@click.option("--num-docs", "-d", default=1000, help="Number of documents to generate")
@click.option("--output", "-o", default="dataset.jsonl", help="Output file path")
@click.option("--seed", "-s", default=42, help="Random seed")
def generate(num_queries, num_docs, output, seed):
    """Generate a synthetic RAG evaluation dataset."""
    queries = generate_synthetic_dataset(num_queries, num_docs, seed=seed)
    save_jsonl(queries, output)
    click.echo(f"Generated {len(queries)} queries across {num_docs} documents.")
    click.echo(f"Written to: {output}")


if __name__ == "__main__":
    main()
