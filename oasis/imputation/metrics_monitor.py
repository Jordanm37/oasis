"""Monitoring and metrics collection for the imputation system.

This module provides real-time monitoring, metrics collection, and reporting
for the multi-dataset RAG imputation pipeline.
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import threading

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str] = None


@dataclass
class ImputationMetrics:
    """Metrics for a single imputation operation."""

    timestamp: float
    persona_id: str
    archetype: str
    retrieval_method: str
    dataset_id: Optional[str]
    latency_ms: float
    success: bool
    cache_hit: bool
    confidence: float
    tokens_used: List[str]
    text_length: int
    obfuscated: bool


class MetricsCollector:
    """Collects and aggregates metrics from the imputation system."""

    def __init__(self,
                 window_size: int = 1000,
                 time_window_minutes: int = 60):
        """Initialize the metrics collector.

        Args:
            window_size: Number of recent metrics to keep in memory
            time_window_minutes: Time window for time-based metrics
        """
        self.window_size = window_size
        self.time_window_minutes = time_window_minutes

        # Metrics storage
        self.imputation_metrics = deque(maxlen=window_size)
        self.retrieval_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.dataset_metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.archetype_metrics = defaultdict(lambda: deque(maxlen=window_size))

        # Counters
        self.total_imputations = 0
        self.total_cache_hits = 0
        self.total_failures = 0
        self.method_counts = defaultdict(int)
        self.dataset_counts = defaultdict(int)
        self.archetype_counts = defaultdict(int)

        # Performance tracking
        self.latency_percentiles = {}
        self.success_rates = {}

        # Start time
        self.start_time = time.time()

    def record_imputation(self, metrics: ImputationMetrics) -> None:
        """Record an imputation operation.

        Args:
            metrics: Imputation metrics
        """
        # Store full metrics
        self.imputation_metrics.append(metrics)

        # Update counters
        self.total_imputations += 1
        if metrics.cache_hit:
            self.total_cache_hits += 1
        if not metrics.success:
            self.total_failures += 1

        # Track by method
        self.method_counts[metrics.retrieval_method] += 1
        self.retrieval_metrics[metrics.retrieval_method].append(metrics)

        # Track by dataset
        if metrics.dataset_id:
            self.dataset_counts[metrics.dataset_id] += 1
            self.dataset_metrics[metrics.dataset_id].append(metrics)

        # Track by archetype
        self.archetype_counts[metrics.archetype] += 1
        self.archetype_metrics[metrics.archetype].append(metrics)

        # Update percentiles periodically
        if self.total_imputations % 100 == 0:
            self._update_percentiles()

    def _update_percentiles(self) -> None:
        """Update latency percentiles."""
        if not self.imputation_metrics:
            return

        latencies = [m.latency_ms for m in self.imputation_metrics if m.latency_ms > 0]
        if not latencies:
            return

        latencies.sort()
        n = len(latencies)

        self.latency_percentiles = {
            'p50': latencies[n // 2],
            'p75': latencies[int(n * 0.75)],
            'p90': latencies[int(n * 0.90)],
            'p95': latencies[int(n * 0.95)],
            'p99': latencies[int(n * 0.99)] if n > 100 else latencies[-1]
        }

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary statistics
        """
        uptime = time.time() - self.start_time

        # Calculate rates
        imputation_rate = self.total_imputations / uptime if uptime > 0 else 0
        cache_hit_rate = self.total_cache_hits / self.total_imputations if self.total_imputations > 0 else 0
        success_rate = (self.total_imputations - self.total_failures) / self.total_imputations if self.total_imputations > 0 else 0

        # Calculate average latency
        recent_latencies = [m.latency_ms for m in self.imputation_metrics if m.latency_ms > 0]
        avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0

        # Calculate method distribution
        method_distribution = {}
        if self.total_imputations > 0:
            for method, count in self.method_counts.items():
                method_distribution[method] = count / self.total_imputations

        return {
            'uptime_seconds': uptime,
            'total_imputations': self.total_imputations,
            'imputation_rate': imputation_rate,
            'cache_hit_rate': cache_hit_rate,
            'success_rate': success_rate,
            'average_latency_ms': avg_latency,
            'latency_percentiles': self.latency_percentiles,
            'method_distribution': method_distribution,
            'dataset_usage': dict(self.dataset_counts),
            'archetype_distribution': dict(self.archetype_counts),
            'total_failures': self.total_failures
        }

    def get_time_series(self,
                       metric: str,
                       minutes: int = 60) -> List[Tuple[float, float]]:
        """Get time series data for a metric.

        Args:
            metric: Metric name
            minutes: Number of minutes to look back

        Returns:
            List of (timestamp, value) tuples
        """
        cutoff_time = time.time() - (minutes * 60)
        time_series = []

        if metric == "imputation_rate":
            # Calculate rate in buckets
            buckets = defaultdict(int)
            for m in self.imputation_metrics:
                if m.timestamp > cutoff_time:
                    bucket = int(m.timestamp / 60) * 60
                    buckets[bucket] += 1

            for bucket_time, count in sorted(buckets.items()):
                time_series.append((bucket_time, count / 60.0))

        elif metric == "latency":
            for m in self.imputation_metrics:
                if m.timestamp > cutoff_time:
                    time_series.append((m.timestamp, m.latency_ms))

        elif metric == "success_rate":
            # Calculate success rate in windows
            window_size = 60  # 1-minute windows
            buckets = defaultdict(lambda: {'success': 0, 'total': 0})

            for m in self.imputation_metrics:
                if m.timestamp > cutoff_time:
                    bucket = int(m.timestamp / window_size) * window_size
                    buckets[bucket]['total'] += 1
                    if m.success:
                        buckets[bucket]['success'] += 1

            for bucket_time, counts in sorted(buckets.items()):
                if counts['total'] > 0:
                    rate = counts['success'] / counts['total']
                    time_series.append((bucket_time, rate))

        return time_series

    def get_method_comparison(self) -> Dict[str, Dict[str, float]]:
        """Compare performance across retrieval methods.

        Returns:
            Dictionary with method performance comparisons
        """
        comparison = {}

        for method, metrics_list in self.retrieval_metrics.items():
            if not metrics_list:
                continue

            latencies = [m.latency_ms for m in metrics_list]
            successes = sum(1 for m in metrics_list if m.success)
            confidences = [m.confidence for m in metrics_list if m.confidence > 0]

            comparison[method] = {
                'count': len(metrics_list),
                'avg_latency': sum(latencies) / len(latencies) if latencies else 0,
                'min_latency': min(latencies) if latencies else 0,
                'max_latency': max(latencies) if latencies else 0,
                'success_rate': successes / len(metrics_list),
                'avg_confidence': sum(confidences) / len(confidences) if confidences else 0
            }

        return comparison

    def get_archetype_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics by archetype.

        Returns:
            Dictionary with archetype performance metrics
        """
        performance = {}

        for archetype, metrics_list in self.archetype_metrics.items():
            if not metrics_list:
                continue

            # Calculate token usage
            token_counts = defaultdict(int)
            for m in metrics_list:
                for token in m.tokens_used:
                    token_counts[token] += 1

            # Get top tokens
            top_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            performance[archetype] = {
                'count': len(metrics_list),
                'avg_text_length': sum(m.text_length for m in metrics_list) / len(metrics_list),
                'obfuscation_rate': sum(1 for m in metrics_list if m.obfuscated) / len(metrics_list),
                'top_tokens': top_tokens,
                'method_distribution': self._get_method_distribution(metrics_list)
            }

        return performance

    def _get_method_distribution(self, metrics_list: List[ImputationMetrics]) -> Dict[str, float]:
        """Get distribution of retrieval methods.

        Args:
            metrics_list: List of metrics

        Returns:
            Method distribution
        """
        method_counts = defaultdict(int)
        for m in metrics_list:
            method_counts[m.retrieval_method] += 1

        total = len(metrics_list)
        return {method: count / total for method, count in method_counts.items()}

    def export_metrics(self, output_path: str) -> None:
        """Export metrics to JSON file.

        Args:
            output_path: Output file path
        """
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_summary_stats(),
            'method_comparison': self.get_method_comparison(),
            'archetype_performance': self.get_archetype_performance(),
            'recent_imputations': [
                asdict(m) for m in list(self.imputation_metrics)[-100:]
            ]
        }

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"Exported metrics to {output_path}")


class MetricsMonitor:
    """Real-time monitoring dashboard for imputation metrics."""

    def __init__(self,
                 collector: MetricsCollector,
                 update_interval: int = 5):
        """Initialize the monitor.

        Args:
            collector: MetricsCollector instance
            update_interval: Update interval in seconds
        """
        self.collector = collector
        self.update_interval = update_interval
        self.running = False
        self.monitor_thread = None

    def start(self) -> None:
        """Start the monitoring dashboard."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Metrics monitor started")

    def stop(self) -> None:
        """Stop the monitoring dashboard."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Metrics monitor stopped")

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                self._print_dashboard()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def _print_dashboard(self) -> None:
        """Print monitoring dashboard to console."""
        stats = self.collector.get_summary_stats()

        # Clear screen (Unix-like systems)
        print("\033[2J\033[H")

        print("=" * 80)
        print("IMPUTATION SYSTEM METRICS DASHBOARD")
        print("=" * 80)
        print(f"Uptime: {self._format_duration(stats['uptime_seconds'])}")
        print()

        # Overall stats
        print("📊 OVERALL STATISTICS")
        print("-" * 40)
        print(f"Total Imputations:    {stats['total_imputations']:,}")
        print(f"Imputation Rate:      {stats['imputation_rate']:.2f} req/sec")
        print(f"Success Rate:         {stats['success_rate']:.1%}")
        print(f"Cache Hit Rate:       {stats['cache_hit_rate']:.1%}")
        print(f"Average Latency:      {stats['average_latency_ms']:.1f} ms")
        print()

        # Latency percentiles
        if stats['latency_percentiles']:
            print("⏱️  LATENCY PERCENTILES (ms)")
            print("-" * 40)
            for percentile, value in stats['latency_percentiles'].items():
                print(f"{percentile.upper():5s}: {value:8.1f}")
            print()

        # Method distribution
        if stats['method_distribution']:
            print("🔄 RETRIEVAL METHOD DISTRIBUTION")
            print("-" * 40)
            for method, pct in sorted(stats['method_distribution'].items(),
                                     key=lambda x: x[1], reverse=True):
                bar = "█" * int(pct * 30)
                print(f"{method:15s} {bar:30s} {pct:.1%}")
            print()

        # Top datasets
        if stats['dataset_usage']:
            print("💾 TOP DATASETS")
            print("-" * 40)
            for dataset, count in sorted(stats['dataset_usage'].items(),
                                        key=lambda x: x[1], reverse=True)[:5]:
                print(f"{dataset:30s} {count:,} requests")
            print()

        # Top archetypes
        if stats['archetype_distribution']:
            print("🏷️  TOP ARCHETYPES")
            print("-" * 40)
            total = sum(stats['archetype_distribution'].values())
            for archetype, count in sorted(stats['archetype_distribution'].items(),
                                          key=lambda x: x[1], reverse=True)[:5]:
                pct = count / total if total > 0 else 0
                print(f"{archetype:20s} {count:6,} ({pct:.1%})")

        print("\n" + "=" * 80)
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class MetricsLogger:
    """Log metrics to file for analysis."""

    def __init__(self,
                 log_dir: str = "logs/imputation/metrics",
                 rotation_interval: str = "hourly"):
        """Initialize the metrics logger.

        Args:
            log_dir: Directory for log files
            rotation_interval: Log rotation interval
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rotation_interval = rotation_interval
        self.current_log = None
        self.last_rotation = None

    def log_metric(self, metric: ImputationMetrics) -> None:
        """Log a metric to file.

        Args:
            metric: Metric to log
        """
        # Check if rotation needed
        if self._should_rotate():
            self._rotate_log()

        # Write metric
        if self.current_log:
            metric_dict = asdict(metric)
            metric_dict['timestamp_str'] = datetime.fromtimestamp(metric.timestamp).isoformat()
            self.current_log.write(json.dumps(metric_dict) + '\n')
            self.current_log.flush()

    def _should_rotate(self) -> bool:
        """Check if log rotation is needed.

        Returns:
            True if rotation needed
        """
        if not self.last_rotation:
            return True

        now = datetime.now()
        if self.rotation_interval == "hourly":
            return now.hour != self.last_rotation.hour
        elif self.rotation_interval == "daily":
            return now.day != self.last_rotation.day

        return False

    def _rotate_log(self) -> None:
        """Rotate log file."""
        # Close current log
        if self.current_log:
            self.current_log.close()

        # Create new log file
        now = datetime.now()
        filename = f"imputation_metrics_{now.strftime('%Y%m%d_%H%M%S')}.jsonl"
        log_path = self.log_dir / filename

        self.current_log = open(log_path, 'w')
        self.last_rotation = now

        logger.info(f"Rotated log to {filename}")

    def close(self) -> None:
        """Close the logger."""
        if self.current_log:
            self.current_log.close()