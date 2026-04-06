"""Main CLI entry point for AgentLegatus."""

import asyncio
import json
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentlegatus.core.event_bus import EventBus
from agentlegatus.core.graph import PortableExecutionGraph
from agentlegatus.core.state import InMemoryStateBackend, StateManager
from agentlegatus.core.workflow import (
    ExecutionStrategy,
    RetryPolicy,
    WorkflowDefinition,
    WorkflowStatus,
    WorkflowStep,
)
from agentlegatus.hierarchy.centurion import Centurion
from agentlegatus.hierarchy.legatus import Legatus
from agentlegatus.observability.benchmark import BenchmarkEngine
from agentlegatus.observability.metrics import MetricsCollector
from agentlegatus.providers.registry import ProviderRegistry
from agentlegatus.tools.registry import ToolRegistry

console = Console()
error_console = Console(stderr=True)

DEFAULT_CONFIG_FILE = "legatus.yaml"


def _get_provider_registry() -> ProviderRegistry:
    """Create a ProviderRegistry with built-in providers registered."""
    from agentlegatus.providers.langgraph import LangGraphProvider
    from agentlegatus.providers.mock import MockProvider

    registry = ProviderRegistry()
    registry.register_provider("mock", MockProvider)
    registry.register_provider("langgraph", LangGraphProvider)
    return registry


def _load_workflow_file(filepath: str) -> dict:
    """Load a workflow definition from a YAML or JSON file.

    Args:
        filepath: Path to the workflow file.

    Returns:
        Parsed workflow data as a dictionary.

    Raises:
        click.ClickException: If the file cannot be loaded.
    """
    path = Path(filepath)
    if not path.exists():
        raise click.ClickException(f"Workflow file not found: {filepath}")

    try:
        content = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(content)
        elif path.suffix == ".json":
            return json.loads(content)
        else:
            raise click.ClickException(
                f"Unsupported file format: {path.suffix}. Use .yaml, .yml, or .json"
            )
    except (yaml.YAMLError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Failed to parse {filepath}: {exc}") from exc


def _build_workflow_definition(
    data: dict, provider_override: str | None = None
) -> WorkflowDefinition:
    """Build a WorkflowDefinition from parsed file data.

    Args:
        data: Parsed workflow data dictionary.
        provider_override: Optional provider name to override the one in the file.

    Returns:
        WorkflowDefinition instance.
    """
    steps = []
    for step_data in data.get("steps", []):
        retry = None
        if "retry_policy" in step_data:
            rp = step_data["retry_policy"]
            retry = RetryPolicy(
                max_attempts=rp.get("max_attempts", 3),
                backoff_multiplier=rp.get("backoff_multiplier", 2.0),
                initial_delay=rp.get("initial_delay", 1.0),
                max_delay=rp.get("max_delay", 60.0),
            )
        steps.append(
            WorkflowStep(
                step_id=step_data["step_id"],
                step_type=step_data.get("step_type", "agent"),
                config=step_data.get("config", {}),
                depends_on=step_data.get("depends_on", []),
                timeout=step_data.get("timeout"),
                retry_policy=retry,
            )
        )

    strategy_str = data.get("execution_strategy", "sequential").upper()
    try:
        strategy = ExecutionStrategy[strategy_str]
    except KeyError:
        strategy = ExecutionStrategy.SEQUENTIAL

    return WorkflowDefinition(
        workflow_id=data.get("workflow_id", "default"),
        name=data.get("name", "Unnamed Workflow"),
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
        provider=provider_override or data.get("provider", "mock"),
        steps=steps,
        initial_state=data.get("initial_state", {}),
        metadata=data.get("metadata", {}),
        timeout=data.get("timeout"),
        execution_strategy=strategy,
    )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="agentlegatus", prog_name="legatus")
def cli():
    """AgentLegatus — Terraform for AI Agents.

    A vendor-agnostic agent framework abstraction layer.
    """


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--provider", default="mock", help="Default provider to use.")
@click.option(
    "--config",
    "config_path",
    default=DEFAULT_CONFIG_FILE,
    help="Path for the configuration file.",
)
def init(provider: str, config_path: str):
    """Initialize AgentLegatus with a default configuration file."""
    path = Path(config_path)
    if path.exists():
        console.print(f"[yellow]Configuration file already exists:[/yellow] {path}")
        return

    registry = _get_provider_registry()
    available = registry.list_providers()
    if provider not in available:
        error_console.print(
            f"[red]Error:[/red] Unknown provider '{provider}'. "
            f"Available: {', '.join(available)}"
        )
        sys.exit(1)

    default_config = {
        "provider": provider,
        "workflow_id": "my-workflow",
        "name": "My Workflow",
        "description": "A new AgentLegatus workflow",
        "version": "1.0.0",
        "execution_strategy": "sequential",
        "steps": [
            {
                "step_id": "step-1",
                "step_type": "agent",
                "config": {"task": "hello"},
            }
        ],
    }

    path.write_text(yaml.dump(default_config, default_flow_style=False, sort_keys=False))
    console.print(
        Panel(
            f"Initialized configuration at [bold]{path}[/bold] with provider [cyan]{provider}[/cyan].",
            title="legatus init",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# apply command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("workflow_file")
@click.option(
    "--provider", default=None, help="Override the provider specified in the workflow file."
)
@click.option("--dry-run", is_flag=True, help="Validate the workflow without executing it.")
def apply(workflow_file: str, provider: str | None, dry_run: bool):
    """Execute a workflow from a file.

    WORKFLOW_FILE is the path to a YAML or JSON workflow definition.
    """
    data = _load_workflow_file(workflow_file)
    workflow_def = _build_workflow_definition(data, provider_override=provider)

    # Validate
    is_valid, errors = workflow_def.validate()
    if not is_valid:
        error_console.print("[red]Workflow validation failed:[/red]")
        for err in errors:
            error_console.print(f"  • {err}")
        sys.exit(1)

    if dry_run:
        console.print(
            Panel(
                "Workflow is valid. No execution performed (--dry-run).",
                title="legatus apply --dry-run",
                border_style="green",
            )
        )
        return

    # Execute
    try:
        result = asyncio.run(_run_workflow(workflow_def))
    except Exception as exc:
        error_console.print(f"[red]Workflow execution failed:[/red] {exc}")
        sys.exit(1)

    # Display result
    status_color = {
        WorkflowStatus.COMPLETED: "green",
        WorkflowStatus.FAILED: "red",
        WorkflowStatus.CANCELLED: "yellow",
    }.get(result.status, "white")

    table = Table(title="Workflow Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Status", f"[{status_color}]{result.status.value}[/{status_color}]")
    table.add_row("Execution Time", f"{result.execution_time:.3f}s")
    if result.metrics:
        table.add_row("Metrics", json.dumps(result.metrics, indent=2, default=str))
    if result.output is not None:
        table.add_row("Output", str(result.output))
    if result.error is not None:
        table.add_row("Error", f"[red]{result.error}[/red]")
    console.print(table)

    if result.status == WorkflowStatus.FAILED:
        sys.exit(1)


async def _run_workflow(workflow_def: WorkflowDefinition):
    """Run a workflow asynchronously.

    Args:
        workflow_def: The workflow definition to execute.

    Returns:
        WorkflowResult from the Legatus orchestrator.
    """
    from agentlegatus.core.executor import WorkflowExecutor

    registry = _get_provider_registry()
    event_bus = EventBus()
    state_backend = InMemoryStateBackend()
    state_manager = StateManager(backend=state_backend, event_bus=event_bus)
    tool_registry = ToolRegistry()

    provider = registry.get_provider(workflow_def.provider)

    executor = WorkflowExecutor(
        provider=provider,
        state_manager=state_manager,
        tool_registry=tool_registry,
        event_bus=event_bus,
    )

    legatus = Legatus(config={"provider": workflow_def.provider}, event_bus=event_bus)

    centurion = Centurion(
        name="main",
        strategy=workflow_def.execution_strategy,
        event_bus=event_bus,
    )
    await legatus.add_centurion(centurion)

    return await legatus.execute_workflow(
        workflow_def,
        initial_state=workflow_def.initial_state,
        executor=executor,
        state_manager=state_manager,
    )


# ---------------------------------------------------------------------------
# plan command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("workflow_file")
def plan(workflow_file: str):
    """Display the execution plan for a workflow without running it.

    WORKFLOW_FILE is the path to a YAML or JSON workflow definition.
    """
    data = _load_workflow_file(workflow_file)
    workflow_def = _build_workflow_definition(data)

    is_valid, errors = workflow_def.validate()
    if not is_valid:
        error_console.print("[red]Workflow validation failed:[/red]")
        for err in errors:
            error_console.print(f"  • {err}")
        sys.exit(1)

    # Build execution plan via Centurion's topological sort
    centurion = Centurion(
        name="planner",
        strategy=workflow_def.execution_strategy,
        event_bus=EventBus(),
    )
    try:
        ordered_steps = centurion.build_execution_plan(workflow_def.steps)
    except ValueError as exc:
        error_console.print(f"[red]Plan error:[/red] {exc}")
        sys.exit(1)

    table = Table(title=f"Execution Plan: {workflow_def.name}")
    table.add_column("#", style="dim", width=4)
    table.add_column("Step ID", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Dependencies")
    table.add_column("Timeout")

    for idx, step in enumerate(ordered_steps, 1):
        deps = ", ".join(step.depends_on) if step.depends_on else "—"
        timeout = f"{step.timeout}s" if step.timeout else "—"
        table.add_row(str(idx), step.step_id, step.step_type, deps, timeout)

    console.print(table)
    console.print(
        f"\nStrategy: [cyan]{workflow_def.execution_strategy.value}[/cyan]  "
        f"Provider: [cyan]{workflow_def.provider}[/cyan]  "
        f"Steps: [cyan]{len(ordered_steps)}[/cyan]"
    )


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("workflow_file")
@click.option(
    "--providers",
    default="mock",
    help="Comma-separated list of providers to benchmark.",
)
@click.option("--iterations", default=10, type=int, help="Number of iterations per provider.")
@click.option("--parallel", is_flag=True, help="Run provider benchmarks in parallel.")
@click.option(
    "--format",
    "report_format",
    default="table",
    type=click.Choice(["table", "json"]),
    help="Report format.",
)
def benchmark(
    workflow_file: str, providers: str, iterations: int, parallel: bool, report_format: str
):
    """Benchmark a workflow across multiple providers.

    WORKFLOW_FILE is the path to a YAML or JSON workflow definition.
    """
    data = _load_workflow_file(workflow_file)
    workflow_def = _build_workflow_definition(data)

    is_valid, errors = workflow_def.validate()
    if not is_valid:
        error_console.print("[red]Workflow validation failed:[/red]")
        for err in errors:
            error_console.print(f"  • {err}")
        sys.exit(1)

    provider_list = [p.strip() for p in providers.split(",") if p.strip()]
    if not provider_list:
        error_console.print("[red]Error:[/red] No providers specified.")
        sys.exit(1)

    try:
        results = asyncio.run(_run_benchmark(workflow_def, provider_list, iterations, parallel))
    except Exception as exc:
        error_console.print(f"[red]Benchmark failed:[/red] {exc}")
        sys.exit(1)

    # Build a PEG for the report generator (it expects one)
    registry = _get_provider_registry()
    engine = BenchmarkEngine(
        provider_registry=registry,
        metrics_collector=MetricsCollector(),
    )
    report = engine.generate_report(results, format=report_format)
    console.print(report)


async def _run_benchmark(workflow_def, provider_list, iterations, parallel):
    """Run benchmark asynchronously."""
    registry = _get_provider_registry()
    engine = BenchmarkEngine(
        provider_registry=registry,
        metrics_collector=MetricsCollector(),
    )

    # Build a simple PEG from the workflow definition
    graph = PortableExecutionGraph()
    from agentlegatus.core.graph import PEGEdge, PEGNode

    for step in workflow_def.steps:
        graph.add_node(
            PEGNode(
                node_id=step.step_id,
                node_type=step.step_type,
                config=step.config,
            )
        )
    for step in workflow_def.steps:
        for dep in step.depends_on:
            graph.add_edge(PEGEdge(source=dep, target=step.step_id))

    return await engine.run_benchmark(
        workflow=graph,
        providers=provider_list,
        iterations=iterations,
        parallel=parallel,
    )


# ---------------------------------------------------------------------------
# switch command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("provider_name")
@click.option(
    "--config",
    "config_path",
    default=DEFAULT_CONFIG_FILE,
    help="Path to the configuration file to update.",
)
def switch(provider_name: str, config_path: str):
    """Switch the default provider in the configuration file.

    PROVIDER_NAME is the name of the provider to switch to.
    """
    registry = _get_provider_registry()
    available = registry.list_providers()
    if provider_name not in available:
        error_console.print(
            f"[red]Error:[/red] Unknown provider '{provider_name}'. "
            f"Available: {', '.join(available)}"
        )
        sys.exit(1)

    path = Path(config_path)
    if not path.exists():
        error_console.print(
            f"[red]Error:[/red] Configuration file not found: {config_path}. "
            "Run 'legatus init' first."
        )
        sys.exit(1)

    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        error_console.print(f"[red]Error:[/red] Failed to parse config: {exc}")
        sys.exit(1)

    old_provider = data.get("provider", "unknown")
    data["provider"] = provider_name
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    console.print(
        f"Switched provider from [yellow]{old_provider}[/yellow] "
        f"to [green]{provider_name}[/green] in {path}."
    )


# ---------------------------------------------------------------------------
# providers command
# ---------------------------------------------------------------------------


@cli.command()
def providers():
    """List all available providers and their capabilities."""
    registry = _get_provider_registry()
    names = registry.list_providers()

    if not names:
        console.print("[yellow]No providers registered.[/yellow]")
        return

    table = Table(title="Available Providers")
    table.add_column("Name", style="cyan")
    table.add_column("Class", style="magenta")
    table.add_column("Capabilities")

    for name in sorted(names):
        info = registry.get_provider_info(name)
        caps = ", ".join(info.get("capabilities", []))
        table.add_row(name, info.get("class", "—"), caps or "—")

    console.print(table)


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("workflow_id")
def status(workflow_id: str):
    """Display the status of a workflow.

    WORKFLOW_ID is the identifier of the workflow to check.
    """
    # In a production system this would query a persistent store.
    # For now we report that no running workflows are tracked in-memory.
    console.print(
        Panel(
            f"No active execution found for workflow [cyan]{workflow_id}[/cyan].\n"
            "Workflow status tracking requires a running Legatus server.",
            title="legatus status",
            border_style="yellow",
        )
    )


# ---------------------------------------------------------------------------
# cancel command
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("workflow_id")
def cancel(workflow_id: str):
    """Cancel a running workflow.

    WORKFLOW_ID is the identifier of the workflow to cancel.
    """
    # In a production system this would signal the running Legatus instance.
    console.print(
        Panel(
            f"No active execution found for workflow [cyan]{workflow_id}[/cyan].\n"
            "Workflow cancellation requires a running Legatus server.",
            title="legatus cancel",
            border_style="yellow",
        )
    )
