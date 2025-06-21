"""
Cross-platform parallel optimization runner.
Replaces bash scripts with Python for Windows compatibility.
"""

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ParallelOptimizationRunner:
    """Cross-platform parallel optimization runner."""

    def __init__(
        self,
        num_workers: int = 4,
        n_trials_per_worker: int = 10,
        timeout_per_worker: int = 600,
        db_type: str = "sqlite",
        mysql_config: Optional[Dict[str, str]] = None,
        output_dir: str = "metrics",
    ):
        self.num_workers = num_workers
        self.n_trials_per_worker = n_trials_per_worker
        self.timeout_per_worker = timeout_per_worker
        self.db_type = db_type
        self.mysql_config = mysql_config or {}
        self.output_dir = Path(output_dir)

        # Create output directory
        self.output_dir.mkdir(exist_ok=True)

        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.output_dir / f"parallel_run_{timestamp}.log"
        self.summary_file = self.output_dir / f"run_summary_{timestamp}.json"

        # Results tracking
        self.worker_results: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None

    def log(self, message: str):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        print(full_message)

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")

    def setup_environment(self, worker_id: int) -> Dict[str, str]:
        """Setup environment variables for a worker."""
        env = os.environ.copy()

        # Add current directory to Python path so imports work
        current_dir = os.getcwd()
        python_path = env.get("PYTHONPATH", "")
        if python_path:
            env["PYTHONPATH"] = f"{python_path}{os.pathsep}{current_dir}"
        else:
            env["PYTHONPATH"] = current_dir

        env.update(
            {
                "WORKER_ID": str(worker_id),
                "N_TRIALS": str(self.n_trials_per_worker),
                "TIMEOUT": str(self.timeout_per_worker),
                "DB_TYPE": self.db_type,
            }
        )

        # Add MySQL configuration if provided
        if self.db_type == "mysql" and self.mysql_config:
            env.update(
                {
                    "MYSQL_USER": self.mysql_config.get("user", "optuna_user"),
                    "MYSQL_PASSWORD": self.mysql_config.get("password", ""),
                    "MYSQL_HOST": self.mysql_config.get("host", "localhost"),
                    "MYSQL_DATABASE": self.mysql_config.get("database", "optuna_db"),
                }
            )

        return env

    def run_single_worker(self, worker_id: int) -> Dict[str, Any]:
        """Run optimization for a single worker."""
        start_time = time.time()

        try:
            # Setup environment
            env = self.setup_environment(worker_id)

            # Run the optimization
            self.log(f"Starting worker {worker_id}")

            # Run the main optimization script
            result = subprocess.run(
                [sys.executable, "-u", "src/modeling/optimization_runner.py"],
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout_per_worker + 60,  # Add buffer for cleanup
            )

            end_time = time.time()
            duration = end_time - start_time

            if result.returncode == 0:
                self.log(
                    f"Worker {worker_id} completed successfully in {duration:.2f}s"
                )
                status = "completed"
                error = None
            else:
                self.log(
                    f"Worker {worker_id} failed with return code {result.returncode}"
                )
                self.log(f"Worker {worker_id} stderr: {result.stderr}")
                status = "failed"
                error = result.stderr

            # Try to load worker results
            worker_results_file = (
                self.output_dir / f"best_params_worker_{worker_id}.json"
            )
            worker_results = None
            if worker_results_file.exists():
                try:
                    with open(worker_results_file, "r") as f:
                        worker_results = json.load(f)
                except Exception as e:
                    self.log(f"Failed to load results for worker {worker_id}: {e}")

            return {
                "worker_id": worker_id,
                "status": status,
                "duration": duration,
                "return_code": result.returncode,
                "error": error,
                "results": worker_results,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.TimeoutExpired:
            self.log(f"Worker {worker_id} timed out after {self.timeout_per_worker}s")
            return {
                "worker_id": worker_id,
                "status": "timeout",
                "duration": self.timeout_per_worker,
                "return_code": -1,
                "error": "Timeout",
                "results": None,
                "stdout": "",
                "stderr": "",
            }
        except Exception as e:
            self.log(f"Worker {worker_id} failed with exception: {e}")
            return {
                "worker_id": worker_id,
                "status": "error",
                "duration": time.time() - start_time,
                "return_code": -1,
                "error": str(e),
                "results": None,
                "stdout": "",
                "stderr": "",
            }

    def run_parallel_optimization(self) -> Dict[str, Any]:
        """Run parallel optimization with multiple workers."""
        self.log(f"Starting parallel optimization with {self.num_workers} workers")
        self.log(
            f"Each worker will run {self.n_trials_per_worker} trials with {self.timeout_per_worker}s timeout"
        )
        self.log(f"Using database type: {self.db_type}")

        self.start_time = time.time()

        # Clean up old worker result files
        for old_file in self.output_dir.glob("best_params_worker_*.json"):
            old_file.unlink()

        # Run workers in parallel
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all workers
            future_to_worker = {
                executor.submit(self.run_single_worker, worker_id): worker_id
                for worker_id in range(1, self.num_workers + 1)
            }

            # Collect results as they complete
            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    result = future.result()
                    self.worker_results.append(result)
                except Exception as e:
                    self.log(f"Worker {worker_id} generated an exception: {e}")
                    self.worker_results.append(
                        {
                            "worker_id": worker_id,
                            "status": "exception",
                            "duration": 0,
                            "return_code": -1,
                            "error": str(e),
                            "results": None,
                            "stdout": "",
                            "stderr": "",
                        }
                    )

        self.end_time = time.time()
        total_duration = self.end_time - self.start_time

        self.log(f"All workers completed in {total_duration:.2f}s")

        # Analyze results
        summary = self.analyze_results()

        # Save summary
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.log(f"Run summary saved to {self.summary_file}")

        return summary

    def analyze_results(self) -> Dict[str, Any]:
        """Analyze results from all workers."""
        successful_workers = [
            r for r in self.worker_results if r["status"] == "completed"
        ]
        failed_workers = [r for r in self.worker_results if r["status"] != "completed"]

        self.log(
            f"Results: {len(successful_workers)} successful, {len(failed_workers)} failed"
        )

        # Find best result
        best_result = None
        best_mse = float("inf")

        for worker_result in successful_workers:
            if worker_result["results"] is not None:
                worker_mse = worker_result["results"].get("best_cv_mse")
                if worker_mse is not None and worker_mse < best_mse:
                    best_mse = worker_mse
                    best_result = worker_result

        if best_result:
            self.log(
                f"Best result from worker {best_result['worker_id']} with MSE: {best_mse:.6f}"
            )

            # Save best overall result
            best_overall_file = self.output_dir / "best_params_overall.json"
            with open(best_overall_file, "w") as f:
                json.dump(best_result["results"], f, indent=2)

            self.log(f"Best overall parameters saved to {best_overall_file}")
        else:
            self.log("No valid results found from any worker")

        summary = {
            "run_info": {
                "num_workers": self.num_workers,
                "n_trials_per_worker": self.n_trials_per_worker,
                "timeout_per_worker": self.timeout_per_worker,
                "db_type": self.db_type,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "total_duration": self.end_time - self.start_time,
            },
            "results_summary": {
                "total_workers": len(self.worker_results),
                "successful_workers": len(successful_workers),
                "failed_workers": len(failed_workers),
                "best_mse": best_mse if best_mse != float("inf") else None,
                "best_worker_id": best_result["worker_id"] if best_result else None,
            },
            "worker_results": self.worker_results,
            "best_result": best_result["results"] if best_result else None,
        }

        return summary


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run parallel hyperparameter optimization"
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of trials per worker"
    )
    parser.add_argument(
        "--timeout", type=int, default=600, help="Timeout per worker in seconds"
    )
    parser.add_argument(
        "--db-type", choices=["sqlite", "mysql"], default="sqlite", help="Database type"
    )
    parser.add_argument("--mysql-user", default="optuna_user", help="MySQL username")
    parser.add_argument("--mysql-password", default="", help="MySQL password")
    parser.add_argument("--mysql-host", default="localhost", help="MySQL host")
    parser.add_argument("--mysql-database", default="optuna_db", help="MySQL database")
    parser.add_argument("--output-dir", default="metrics", help="Output directory")

    args = parser.parse_args()

    # Setup MySQL config if needed
    mysql_config = None
    if args.db_type == "mysql":
        mysql_config = {
            "user": args.mysql_user,
            "password": args.mysql_password,
            "host": args.mysql_host,
            "database": args.mysql_database,
        }

    # Create and run the parallel optimizer
    runner = ParallelOptimizationRunner(
        num_workers=args.workers,
        n_trials_per_worker=args.trials,
        timeout_per_worker=args.timeout,
        db_type=args.db_type,
        mysql_config=mysql_config,
        output_dir=args.output_dir,
    )

    try:
        summary = runner.run_parallel_optimization()

        if summary["results_summary"]["best_mse"] is not None:
            print(f"\n✅ Optimization completed successfully!")
            print(f"Best MSE: {summary['results_summary']['best_mse']:.6f}")
            print(f"Best worker: {summary['results_summary']['best_worker_id']}")
        else:
            print(f"\n⚠️ Optimization completed but no valid results found")

        print(f"Summary saved to: {runner.summary_file}")

    except KeyboardInterrupt:
        print("\n⏹️ Optimization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
