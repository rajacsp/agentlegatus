"""Unit tests for CLI commands."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from agentlegatus.cli.main import cli
from agentlegatus.core.workflow import WorkflowResult, WorkflowStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_workflow_data() -> dict:
    """Return minimal valid workflow data."""
    return {
        "workflow_id": "test-wf",
        "name": "Test Workflow",
        "description": "A test workflow",
        "version": "1.0.0",
        "provider": "mock",
        "execution_strategy": "sequential",
        "steps": [
            {
                "step_id": "step-1",
                "step_type": "agent",
                "config": {"task": "hello"},
            }
        ],
    }


def _write_workflow_file(tmp_path: Path, data: dict, ext: str = ".yaml") -> Path:
    """Write workflow data to a file and return the path."""
    filepath = tmp_path / f"workflow{ext}"
    if ext in (".yaml", ".yml"):
        filepath.write_text(yaml.dump(data, default_flow_style=False))
    elif ext == ".json":
        filepath.write_text(json.dumps(data))
    else:
        filepath.write_text("raw content")
    return filepath


def _combined_output(result) -> str:
    """Get combined stdout + stderr output from a CliRunner result."""
    parts = []
    if result.output:
        parts.append(result.output)
    if hasattr(result, "stderr") and result.stderr:
        parts.append(result.stderr)
    return "\n".join(parts)


@pytest.fixture
def runner():
    """Create a Click CliRunner."""
    return CliRunner()


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------

class TestInitCommand:
    """Tests for the 'init' command."""

    def test_init_creates_default_config(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        result = runner.invoke(cli, ["init", "--config", str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "mock"
        assert "steps" in data

    def test_init_with_custom_provider(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        result = runner.invoke(cli, ["init", "--provider", "langgraph", "--config", str(config_path)])
        assert result.exit_code == 0
        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "langgraph"

    def test_init_skips_if_config_exists(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        config_path.write_text("existing: true")
        result = runner.invoke(cli, ["init", "--config", str(config_path)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "already exists" in out
        # File should not be overwritten
        assert yaml.safe_load(config_path.read_text()) == {"existing": True}

    def test_init_unknown_provider_exits_nonzero(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        result = runner.invoke(cli, ["init", "--provider", "nonexistent", "--config", str(config_path)])
        assert result.exit_code != 0

    def test_init_output_contains_provider_name(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        result = runner.invoke(cli, ["init", "--config", str(config_path)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "mock" in out


# ---------------------------------------------------------------------------
# apply command
# ---------------------------------------------------------------------------

class TestApplyCommand:
    """Tests for the 'apply' command."""

    def test_apply_missing_file_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["apply", "/nonexistent/workflow.yaml"])
        assert result.exit_code != 0

    def test_apply_dry_run_valid_workflow(self, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath), "--dry-run"])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "dry-run" in out.lower() or "valid" in out.lower()

    def test_apply_dry_run_invalid_workflow(self, runner, tmp_path):
        data = _minimal_workflow_data()
        data["workflow_id"] = ""  # invalid
        filepath = _write_workflow_file(tmp_path, data)
        result = runner.invoke(cli, ["apply", str(filepath), "--dry-run"])
        assert result.exit_code != 0

    def test_apply_json_workflow_file(self, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data(), ext=".json")
        result = runner.invoke(cli, ["apply", str(filepath), "--dry-run"])
        assert result.exit_code == 0

    def test_apply_unsupported_file_format(self, runner, tmp_path):
        filepath = tmp_path / "workflow.txt"
        filepath.write_text("not a workflow")
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code != 0

    def test_apply_invalid_yaml_exits_nonzero(self, runner, tmp_path):
        filepath = tmp_path / "workflow.yaml"
        filepath.write_text("{{invalid yaml: [")
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code != 0

    @patch("agentlegatus.cli.main._run_workflow")
    def test_apply_executes_workflow(self, mock_run, runner, tmp_path):
        mock_run.return_value = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            output={"result": "ok"},
            metrics={"duration": 1.0},
            execution_time=1.0,
        )
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "completed" in out.lower()
        mock_run.assert_called_once()

    @patch("agentlegatus.cli.main._run_workflow")
    def test_apply_failed_workflow_exits_nonzero(self, mock_run, runner, tmp_path):
        mock_run.return_value = WorkflowResult(
            status=WorkflowStatus.FAILED,
            output=None,
            metrics={},
            execution_time=0.5,
            error=Exception("step failed"),
        )
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code != 0

    @patch("agentlegatus.cli.main._run_workflow")
    def test_apply_with_provider_override(self, mock_run, runner, tmp_path):
        mock_run.return_value = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            output="done",
            metrics={},
            execution_time=0.1,
        )
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath), "--provider", "langgraph"])
        assert result.exit_code == 0
        # The workflow definition passed to _run_workflow should use the override
        call_args = mock_run.call_args[0][0]
        assert call_args.provider == "langgraph"

    @patch("agentlegatus.cli.main._run_workflow", side_effect=RuntimeError("boom"))
    def test_apply_execution_exception_exits_nonzero(self, mock_run, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code != 0

    @patch("agentlegatus.cli.main._run_workflow")
    def test_apply_output_shows_execution_time(self, mock_run, runner, tmp_path):
        mock_run.return_value = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            output="done",
            metrics={},
            execution_time=2.345,
        )
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "2.345" in out


# ---------------------------------------------------------------------------
# plan command
# ---------------------------------------------------------------------------

class TestPlanCommand:
    """Tests for the 'plan' command."""

    def test_plan_displays_steps(self, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["plan", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "step-1" in out

    def test_plan_shows_strategy(self, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["plan", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "sequential" in out.lower()

    def test_plan_shows_provider(self, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["plan", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "mock" in out

    def test_plan_shows_step_count(self, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["plan", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "1" in out

    def test_plan_multi_step_workflow(self, runner, tmp_path):
        data = _minimal_workflow_data()
        data["steps"].append({
            "step_id": "step-2",
            "step_type": "agent",
            "config": {"task": "world"},
            "depends_on": ["step-1"],
        })
        filepath = _write_workflow_file(tmp_path, data)
        result = runner.invoke(cli, ["plan", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "step-1" in out
        assert "step-2" in out

    def test_plan_invalid_workflow_exits_nonzero(self, runner, tmp_path):
        data = _minimal_workflow_data()
        data["workflow_id"] = ""
        filepath = _write_workflow_file(tmp_path, data)
        result = runner.invoke(cli, ["plan", str(filepath)])
        assert result.exit_code != 0

    def test_plan_missing_file_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["plan", "/nonexistent/workflow.yaml"])
        assert result.exit_code != 0

    def test_plan_shows_dependencies(self, runner, tmp_path):
        data = _minimal_workflow_data()
        data["steps"].append({
            "step_id": "step-2",
            "step_type": "agent",
            "config": {},
            "depends_on": ["step-1"],
        })
        filepath = _write_workflow_file(tmp_path, data)
        result = runner.invoke(cli, ["plan", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        # step-1 appears both as a step and as a dependency for step-2
        assert "step-1" in out


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------

class TestBenchmarkCommand:
    """Tests for the 'benchmark' command."""

    def test_benchmark_missing_file_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["benchmark", "/nonexistent/workflow.yaml"])
        assert result.exit_code != 0

    def test_benchmark_invalid_workflow_exits_nonzero(self, runner, tmp_path):
        data = _minimal_workflow_data()
        data["workflow_id"] = ""
        filepath = _write_workflow_file(tmp_path, data)
        result = runner.invoke(cli, ["benchmark", str(filepath)])
        assert result.exit_code != 0

    @patch("agentlegatus.cli.main._run_benchmark")
    def test_benchmark_runs_with_defaults(self, mock_bench, runner, tmp_path):
        from agentlegatus.observability.metrics import BenchmarkMetrics

        mock_bench.return_value = {
            "mock": BenchmarkMetrics(
                provider_name="mock",
                execution_time=1.0,
                total_cost=0.01,
                token_usage={"input": 100, "output": 50, "total": 150},
                success_rate=1.0,
                error_count=0,
                latency_p50=0.5,
                latency_p95=0.8,
                latency_p99=0.9,
            )
        }
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["benchmark", str(filepath)])
        assert result.exit_code == 0
        mock_bench.assert_called_once()

    @patch("agentlegatus.cli.main._run_benchmark")
    def test_benchmark_with_providers_option(self, mock_bench, runner, tmp_path):
        from agentlegatus.observability.metrics import BenchmarkMetrics

        mock_bench.return_value = {
            "mock": BenchmarkMetrics(
                provider_name="mock",
                execution_time=1.0,
                total_cost=0.0,
                token_usage={"input": 0, "output": 0, "total": 0},
                success_rate=1.0,
                error_count=0,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
            ),
            "langgraph": BenchmarkMetrics(
                provider_name="langgraph",
                execution_time=2.0,
                total_cost=0.0,
                token_usage={"input": 0, "output": 0, "total": 0},
                success_rate=1.0,
                error_count=0,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
            ),
        }
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["benchmark", str(filepath), "--providers", "mock,langgraph"])
        assert result.exit_code == 0

    @patch("agentlegatus.cli.main._run_benchmark")
    def test_benchmark_iterations_option(self, mock_bench, runner, tmp_path):
        from agentlegatus.observability.metrics import BenchmarkMetrics

        mock_bench.return_value = {
            "mock": BenchmarkMetrics(
                provider_name="mock",
                execution_time=1.0,
                total_cost=0.0,
                token_usage={"input": 0, "output": 0, "total": 0},
                success_rate=1.0,
                error_count=0,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
            )
        }
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["benchmark", str(filepath), "--iterations", "5"])
        assert result.exit_code == 0
        # Verify _run_benchmark was called (iterations passed through)
        mock_bench.assert_called_once()

    @patch("agentlegatus.cli.main._run_benchmark")
    def test_benchmark_json_format(self, mock_bench, runner, tmp_path):
        from agentlegatus.observability.metrics import BenchmarkMetrics

        mock_bench.return_value = {
            "mock": BenchmarkMetrics(
                provider_name="mock",
                execution_time=1.0,
                total_cost=0.0,
                token_usage={"input": 0, "output": 0, "total": 0},
                success_rate=1.0,
                error_count=0,
                latency_p50=0.0,
                latency_p95=0.0,
                latency_p99=0.0,
            )
        }
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["benchmark", str(filepath), "--format", "json"])
        assert result.exit_code == 0

    @patch("agentlegatus.cli.main._run_benchmark", side_effect=RuntimeError("bench error"))
    def test_benchmark_exception_exits_nonzero(self, mock_bench, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["benchmark", str(filepath)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# switch command
# ---------------------------------------------------------------------------

class TestSwitchCommand:
    """Tests for the 'switch' command."""

    def test_switch_updates_config_file(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        config_path.write_text(yaml.dump({"provider": "mock"}))
        result = runner.invoke(cli, ["switch", "langgraph", "--config", str(config_path)])
        assert result.exit_code == 0
        data = yaml.safe_load(config_path.read_text())
        assert data["provider"] == "langgraph"

    def test_switch_output_shows_old_and_new(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        config_path.write_text(yaml.dump({"provider": "mock"}))
        result = runner.invoke(cli, ["switch", "langgraph", "--config", str(config_path)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "mock" in out
        assert "langgraph" in out

    def test_switch_unknown_provider_exits_nonzero(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        config_path.write_text(yaml.dump({"provider": "mock"}))
        result = runner.invoke(cli, ["switch", "nonexistent", "--config", str(config_path)])
        assert result.exit_code != 0

    def test_switch_missing_config_exits_nonzero(self, runner, tmp_path):
        config_path = tmp_path / "no_such_file.yaml"
        result = runner.invoke(cli, ["switch", "mock", "--config", str(config_path)])
        assert result.exit_code != 0

    def test_switch_invalid_yaml_config_exits_nonzero(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        config_path.write_text("{{bad yaml: [")
        result = runner.invoke(cli, ["switch", "mock", "--config", str(config_path)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# providers command
# ---------------------------------------------------------------------------

class TestProvidersCommand:
    """Tests for the 'providers' command."""

    def test_providers_lists_available(self, runner):
        result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "mock" in out
        assert "langgraph" in out

    def test_providers_output_is_table(self, runner):
        result = runner.invoke(cli, ["providers"])
        assert result.exit_code == 0
        out = _combined_output(result)
        # Rich table output contains column headers
        assert "Name" in out or "name" in out.lower()


# ---------------------------------------------------------------------------
# status command
# ---------------------------------------------------------------------------

class TestStatusCommand:
    """Tests for the 'status' command."""

    def test_status_with_workflow_id(self, runner):
        result = runner.invoke(cli, ["status", "my-workflow"])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "my-workflow" in out

    def test_status_shows_no_active_execution(self, runner):
        result = runner.invoke(cli, ["status", "any-id"])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "no active" in out.lower()


# ---------------------------------------------------------------------------
# cancel command
# ---------------------------------------------------------------------------

class TestCancelCommand:
    """Tests for the 'cancel' command."""

    def test_cancel_with_workflow_id(self, runner):
        result = runner.invoke(cli, ["cancel", "my-workflow"])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "my-workflow" in out

    def test_cancel_shows_no_active_execution(self, runner):
        result = runner.invoke(cli, ["cancel", "any-id"])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "no active" in out.lower()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for CLI error handling across commands."""

    def test_unknown_command_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["nonexistent-command"])
        assert result.exit_code != 0

    def test_apply_no_arguments_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["apply"])
        assert result.exit_code != 0

    def test_plan_no_arguments_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["plan"])
        assert result.exit_code != 0

    def test_benchmark_no_arguments_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["benchmark"])
        assert result.exit_code != 0

    def test_switch_no_arguments_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["switch"])
        assert result.exit_code != 0

    def test_status_no_arguments_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["status"])
        assert result.exit_code != 0

    def test_cancel_no_arguments_exits_nonzero(self, runner):
        result = runner.invoke(cli, ["cancel"])
        assert result.exit_code != 0

    def test_apply_validation_errors_cause_nonzero_exit(self, runner, tmp_path):
        """Validation errors cause non-zero exit code."""
        data = _minimal_workflow_data()
        data["steps"] = []  # no steps = invalid
        filepath = _write_workflow_file(tmp_path, data)
        result = runner.invoke(cli, ["apply", str(filepath), "--dry-run"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

class TestOutputFormatting:
    """Tests for CLI output formatting."""

    @patch("agentlegatus.cli.main._run_workflow")
    def test_apply_result_table_contains_status(self, mock_run, runner, tmp_path):
        mock_run.return_value = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            output="done",
            metrics={"tokens": 100},
            execution_time=1.5,
        )
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "completed" in out.lower()

    @patch("agentlegatus.cli.main._run_workflow")
    def test_apply_result_shows_metrics(self, mock_run, runner, tmp_path):
        mock_run.return_value = WorkflowResult(
            status=WorkflowStatus.COMPLETED,
            output="done",
            metrics={"tokens": 100},
            execution_time=1.0,
        )
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "tokens" in out.lower()

    @patch("agentlegatus.cli.main._run_workflow")
    def test_apply_result_shows_error_on_failure(self, mock_run, runner, tmp_path):
        mock_run.return_value = WorkflowResult(
            status=WorkflowStatus.FAILED,
            output=None,
            metrics={},
            execution_time=0.5,
            error=Exception("something went wrong"),
        )
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["apply", str(filepath)])
        assert result.exit_code != 0
        out = _combined_output(result)
        assert "something went wrong" in out.lower()

    def test_plan_output_contains_step_info(self, runner, tmp_path):
        filepath = _write_workflow_file(tmp_path, _minimal_workflow_data())
        result = runner.invoke(cli, ["plan", str(filepath)])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "step-1" in out
        assert "agent" in out.lower()

    def test_init_output_contains_init_info(self, runner, tmp_path):
        config_path = tmp_path / "legatus.yaml"
        result = runner.invoke(cli, ["init", "--config", str(config_path)])
        assert result.exit_code == 0
        out = _combined_output(result)
        # Should mention initialization or the config path
        assert "init" in out.lower() or str(config_path) in out or "initialized" in out.lower()

    def test_version_option(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "legatus" in out.lower() or "version" in out.lower()

    def test_help_option(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        out = _combined_output(result)
        assert "agentlegatus" in out.lower() or "terraform" in out.lower()
