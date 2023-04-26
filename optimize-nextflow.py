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
    task_metrics = json_parser.parse()
    config_generator = NextflowConfigGenerator(task_metrics)
    config = config_generator.generate()
    print(config)


@app.command()
def from_api():
    """Process the metrics JSON output from the Tower API.

    Not implemented yet.
    """
    ...


@dataclass
class Stats(Generic[Number]):
    q1: Number
    q2: Number
    q3: Number
    min: Number
    max: Number
    mean: Number

    @cached_property
    def max_q3_ratio(self):
        return round(self.max / self.q3, 2)


@dataclass
class TaskMetrics:
    name: str
    mem: Stats
    mem_pct: Stats
    cpu: Stats
    cpu_pct: Stats

    @staticmethod
    def bytes_to_gb(memory: int) -> int:
        memory_gb = memory / 1024**3
        return ceil(memory_gb + 0.5)

    @staticmethod
    def load_to_cores(load: float) -> int:
        cores = load / 100
        return ceil(cores + 0.5)

    @cached_property
    def mem_gb(self) -> Stats:
        values = astuple(self.mem)
        values_adj = [self.bytes_to_gb(x) for x in values]
        return Stats(*values_adj)

    @cached_property
    def cpu_cores(self) -> Stats:
        values = astuple(self.cpu)
        values_adj = [self.load_to_cores(x) for x in values]
        return Stats(*values_adj)

    def optimize_mem(self) -> int:
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
        if self.cpu_cores.max_q3_ratio <= 2 and self.cpu_cores.max <= 8:
            return self.cpu_cores.max
        elif self.cpu_cores.max_q3_ratio <= 1.5 and self.cpu_cores.max <= 16:
            return self.cpu_cores.max
        else:
            return self.cpu_cores.q3


@dataclass
class MetricsJsonParser:
    json_path: Path

    RawPctPair = namedtuple("RawPctPair", ["raw", "pct"])

    @cached_property
    def json(self):
        return json.load(self.json_path.open())

    def parse(self) -> list[TaskMetrics]:
        mem_metrics = self.parse_resource("metricsMemory", "memRaw", "memUsage")
        cpu_metrics = self.parse_resource("metricsCpu", "cpuRaw", "cpuUsage")
        tasks = set(mem_metrics) & set(cpu_metrics)

        all_task_metrics = list()
        for task in tasks:
            mem = mem_metrics[task]
            cpu = cpu_metrics[task]
            task_metrics = TaskMetrics(task, mem.raw, mem.pct, cpu.raw, cpu.pct)
            all_task_metrics.append(task_metrics)
        return all_task_metrics

    def parse_resource(self, top_key, raw_key, pct_key) -> dict[str, RawPctPair]:
        metrics = dict()
        for task in self.json[top_key]:
            task_name, task_metrics = task.popitem()
            if not task_metrics:
                continue
            usage_raw = Stats(**task_metrics[raw_key])
            usage_pct = Stats(**task_metrics[pct_key])
            metrics[task_name] = self.RawPctPair(usage_raw, usage_pct)
        return metrics


@dataclass
class NextflowConfigGenerator:
    task_metrics: list[TaskMetrics]

    def generate(self) -> str:
        prefix = "process { \n"
        suffix = "\n }"

        task_configs = list()
        for task in self.task_metrics:
            task_config = self.generate_single(task)
            task_config_fmt = indent(task_config, "  ")
            task_configs.append(task_config_fmt)
        task_configs_concat = "".join(task_configs)

        return prefix + task_configs_concat + suffix

    @staticmethod
    def generate_single(task_metrics: TaskMetrics) -> str:
        mem_optimal = task_metrics.optimize_mem()
        cpu_optimal = task_metrics.optimize_cpu()
        config = f"""
            withName: { task_metrics.name } {{
                memory = { mem_optimal }.GB
                cpu = { cpu_optimal }
            }}
        """
        return dedent(config)


if __name__ == "__main__":
    app()
