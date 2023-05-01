#!/usr/bin/env python3

import json
import shutil
from collections import namedtuple
from dataclasses import astuple, dataclass
from functools import cached_property
from math import ceil
from pathlib import Path
from textwrap import dedent, indent
from typing import Generic, TypeVar

import typer

Number = TypeVar("Number", int, float)

app = typer.Typer(
    context_settings={"max_content_width": shutil.get_terminal_size().columns}
)


@app.command()
def from_json(metrics_path: Path):
    """Process the metrics JSON output from the Tower CLI.

    Here's an example Tower CLI command:

        tw --output "json" runs view -w "<org>/<workspace>" -i "<workflow-id>" metrics
    """
    json_parser = MetricsJsonParser(metrics_path)
    process_metrics = json_parser.parse()
    config_generator = NextflowConfigGenerator(process_metrics)
    config = config_generator.generate()
    print(config)


@app.command()
def from_api():
    """Process the metrics JSON output from the Tower API.

    Not implemented yet.
    """
    raise NotImplementedError()


@dataclass
class Stats(Generic[Number]):
    """Suite of statistics."""

    q1: Number
    q2: Number
    q3: Number
    min: Number
    max: Number
    mean: Number

    @cached_property
    def max_q3_ratio(self):
        """Ratio between the max and Q3 values."""
        return round(self.max / self.q3, 2)


@dataclass
class ProcessMetrics:
    """Set of metrics for a process."""

    name: str
    mem: Stats
    mem_pct: Stats
    cpu: Stats
    cpu_pct: Stats

    @staticmethod
    def bytes_to_gb(memory: int) -> int:
        """Convert memory in bytes to GB (with some buffer).

        Args:
            memory: Memory in bytes.

        Returns:
            Memory in GB (with some buffer).
        """
        memory_gb = memory / 1024**3
        return ceil(memory_gb + 0.5)

    @staticmethod
    def load_to_cores(load: float) -> int:
        """Convert CPU load to number of cores (with some buffer).

        Args:
            load: CPU load in percentage.

        Returns:
            Number of CPU cores (with some buffer).
        """
        cores = load / 100
        return ceil(cores + 0.5)

    @cached_property
    def mem_gb(self) -> Stats:
        """Retrieve memory statistics in GB.

        Returns:
            Memory statistics in GB.
        """
        values = astuple(self.mem)
        values_adj = [self.bytes_to_gb(x) for x in values]
        return Stats(*values_adj)

    @cached_property
    def cpu_cores(self) -> Stats:
        """Retrieve CPU statistics in cores.

        Returns:
            CPU statistics in cores.
        """
        values = astuple(self.cpu)
        values_adj = [self.load_to_cores(x) for x in values]
        return Stats(*values_adj)

    def optimize_mem(self) -> int:
        """Optimize memory allocation based on heuristics.

        Returns:
            Optimized memory allocation.
        """
        if self.mem.max == 1 and self.mem_pct.max <= 80:
            return self.mem_gb.max
        elif self.mem_gb.max <= 8:
            return self.mem_gb.max
        elif self.mem_gb.max_q3_ratio <= 1.5 and self.mem_gb.max <= 32:
            return self.mem_gb.max
        elif self.mem_gb.max_q3_ratio <= 1.2 and self.mem_gb.max <= 64:
            return self.mem_gb.max
        else:
            return self.mem_gb.q3

    def optimize_cpu(self) -> int:
        """Optimize CPU allocation based on heuristics.

        Returns:
            Optimized CPU allocation.
        """
        if self.cpu_cores.max_q3_ratio <= 2 and self.cpu_cores.max <= 8:
            return self.cpu_cores.max
        elif self.cpu_cores.max_q3_ratio <= 1.5 and self.cpu_cores.max <= 16:
            return self.cpu_cores.max
        else:
            return self.cpu_cores.q3


@dataclass
class MetricsJsonParser:
    """Parser for JSON file with Tower process metrics."""

    json_path: Path

    RawPctPair = namedtuple("RawPctPair", ["raw", "pct"])

    @cached_property
    def json(self):
        """De-serialized JSON contents."""
        return json.load(self.json_path.open())

    def parse(self) -> list[ProcessMetrics]:
        """Parse JSON file with Tower process metrics.

        Returns:
            List of Tower process metrics.
        """
        mem_metrics = self.parse_resource("metricsMemory", "memRaw", "memUsage")
        cpu_metrics = self.parse_resource("metricsCpu", "cpuRaw", "cpuUsage")
        processes = set(mem_metrics) & set(cpu_metrics)

        all_process_metrics = list()
        for process in processes:
            mem = mem_metrics[process]
            cpu = cpu_metrics[process]
            process_metrics = ProcessMetrics(
                process, mem.raw, mem.pct, cpu.raw, cpu.pct
            )
            all_process_metrics.append(process_metrics)
        return all_process_metrics

    def parse_resource(self, top_key, raw_key, pct_key) -> dict[str, RawPctPair]:
        """Parse the statistics for a specific resource.

        Args:
            top_key: Resource-specific top-level key.
            raw_key: Key for raw resource usage.
            pct_key: Key for percent resource usage.

        Returns:
            Resource usage statistics for each process.
        """
        metrics = dict()
        for process in self.json[top_key]:
            process_name, process_metrics = process.popitem()
            if not process_metrics:
                continue
            usage_raw = Stats(**process_metrics[raw_key])
            usage_pct = Stats(**process_metrics[pct_key])
            metrics[process_name] = self.RawPctPair(usage_raw, usage_pct)
        return metrics


@dataclass
class NextflowConfigGenerator:
    """Generator for Nextflow configuration."""

    process_metrics: list[ProcessMetrics]

    def generate(self) -> str:
        """Generate Nextflow configuration based on process metrics.

        Returns:
            Serialized Nextflow configuration.
        """
        prefix = "process { \n"
        suffix = "\n }"

        process_configs = list()
        for process in self.process_metrics:
            process_config = self.generate_single(process)
            process_config_fmt = indent(process_config, "  ")
            process_configs.append(process_config_fmt)
        process_configs_concat = "".join(process_configs)

        return prefix + process_configs_concat + suffix

    @staticmethod
    def generate_single(process_metrics: ProcessMetrics) -> str:
        """Generate Nextflow configuration for a single process.

        Args:
            process_metrics: Resource metrics for a single process.

        Returns:
            Serialized Nextflow configuration snippet.
        """
        mem_optimal = process_metrics.optimize_mem()
        cpu_optimal = process_metrics.optimize_cpu()
        config = f"""
            withName: { process_metrics.name } {{
                memory = { mem_optimal }.GB
                cpu = { cpu_optimal }
            }}
        """
        return dedent(config)


if __name__ == "__main__":
    app()
