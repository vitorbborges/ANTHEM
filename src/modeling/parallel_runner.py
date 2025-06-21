# src/modeling/parallel_runner.py
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .optimization import OptimizationRunner
from .utils import log_message


class ParallelOptimizationRunner:
    """Cross-platform parallel optimization runner."""

    def __init__(
        self,
        study_name: str = "co2_prediction",
        storage_type: str = "sqlite",
        mysql_url: Optional[str] = None,
        data_path: str = "data/processed_data/S3-coords.parquet",
    ):
        """
        Initialize parallel runner.

        Args:
            study_name: Name of the Optuna study
            storage_type: 'sqlite' for local, 'mysql' for distributed
            mysql_url: MySQL connection string (only for mysql)
            data_path: Path to the data file
        """
        self.study_name = study_name
        self.storage_type = storage_type
        self.mysql_url = mysql_url
        self.data_path = data_path

        # Create output directories
        Path("metrics").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)

        self.start_time = datetime.now()
        self.summary_file = f"metrics/parallel_run_summary_{self.start_time.strftime('%Y%m%d_%H%M%S')}.txt"

    def run_single_worker(self, worker_config: Dict) -> Dict:
        """Run optimization for a single worker."""
        worker_id = worker_config["worker_id"]
        n_trials = worker_config["n_trials"]
        timeout = worker_config.get("timeout")

        # Set environment variable for worker ID
        os.environ["WORKER_ID"] = str(worker_id)

        try:
            # Create optimization runner
            runner = OptimizationRunner(
                study_name=self.study_name,
                storage_type=self.storage_type,
                mysql_url=self.mysql_url,
                data_path=self.data_path,
            )

            # Run optimization
            study = runner.run_optimization(
                n_trials=n_trials,
                timeout=timeout,
                show_progress=False,  # Disable progress bar in parallel mode
            )

            # Return results
            result = {
                "worker_id": worker_id,
                "success": True,
                "n_trials_completed": len(
                    [t for t in study.trials if t.state.name == "COMPLETE"]
                ),
                "best_value": study.best_trial.value if study.best_trial else None,
            }

            if study.best_trial:
                result["best_params"] = study.best_trial.params
                result.update(
                    {
                        k: v
                        for k, v in study.best_trial.user_attrs.items()
                        if isinstance(v, (int, float, str, bool))
                    }
                )

            return result

        except Exception as e:
            return {"worker_id": worker_id, "success": False, "error": str(e)}

    def run_parallel(
        self,
        num_workers: int = None,
        trials_per_worker: int = 10,
        timeout_per_worker: int = 600,
    ) -> List[Dict]:
        """
        Run optimization in parallel across multiple workers.

        Args:
            num_workers: Number of parallel workers (default: CPU count)
            trials_per_worker: Number of trials per worker
            timeout_per_worker: Timeout per worker in seconds

        Returns:
            List of worker results
        """
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid overwhelming

        log_message(f"Starting parallel optimization with {num_workers} workers")
        log_message(f"Trials per worker: {trials_per_worker}")
        log_message(f"Timeout per worker: {timeout_per_worker}s")
        log_message(f"Storage type: {self.storage_type}")

        # Create worker configurations
        worker_configs = []
        for i in range(num_workers):
            worker_configs.append(
                {
                    "worker_id": i + 1,
                    "n_trials": trials_per_worker,
                    "timeout": timeout_per_worker,
                }
            )

        # Run workers in parallel
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all workers
            future_to_worker = {
                executor.submit(self.run_single_worker, config): config["worker_id"]
                for config in worker_configs
            }

            # Collect results as they complete
            for future in as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result["success"]:
                        log_message(
                            f"Worker {worker_id} completed successfully: "
                            f"{result['n_trials_completed']} trials, "
                            f"best MSE: {result.get('best_value', 'N/A')}"
                        )
                    else:
                        log_message(f"Worker {worker_id} failed: {result['error']}")

                except Exception as e:
                    log_message(
                        f"Worker {worker_id} encountered unexpected error: {str(e)}"
                    )
                    results.append(
                        {"worker_id": worker_id, "success": False, "error": str(e)}
                    )

        # Save summary
        self._save_summary(results, num_workers, trials_per_worker)

        return results

    def _save_summary(
        self, results: List[Dict], num_workers: int, trials_per_worker: int
    ):
        """Save a summary of the parallel run."""
        successful_workers = [r for r in results if r["success"]]
        failed_workers = [r for r in results if not r["success"]]

        total_time = (datetime.now() - self.start_time).total_seconds()

        summary = {
            "run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_time_seconds": total_time,
                "num_workers": num_workers,
                "trials_per_worker": trials_per_worker,
                "storage_type": self.storage_type,
                "study_name": self.study_name,
            },
            "results": {
                "successful_workers": len(successful_workers),
                "failed_workers": len(failed_workers),
                "total_trials_completed": sum(
                    r.get("n_trials_completed", 0) for r in successful_workers
                ),
            },
            "worker_results": results,
        }

        # Find best overall result
        if successful_workers:
            best_worker = min(
                successful_workers, key=lambda x: x.get("best_value", float("inf"))
            )
            summary["best_overall"] = {
                "worker_id": best_worker["worker_id"],
                "best_value": best_worker.get("best_value"),
                "best_params": best_worker.get("best_params", {}),
            }

            log_message(
                f"Best overall result from worker {best_worker['worker_id']}: "
                f"MSE = {best_worker.get('best_value', 'N/A')}"
            )

        # Save summary to file
        with open(self.summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        log_message(f"Parallel run completed in {total_time:.2f}s")
        log_message(f"Summary saved to {self.summary_file}")
        log_message(f"Successful workers: {len(successful_workers)}/{num_workers}")
        log_message(
            f"Total trials completed: {summary['results']['total_trials_completed']}"
        )

    def find_best_result(self) -> Optional[Dict]:
        """Find the best result across all worker output files."""
        best_result = None
        best_mse = float("inf")

        metrics_dir = Path("metrics")
        for result_file in metrics_dir.glob("best_params_worker_*.json"):
            try:
                with open(result_file, "r") as f:
                    result = json.load(f)

                mse = result.get("best_cv_mse") or result.get("holdout_test_mse")
                if mse and mse < best_mse:
                    best_mse = mse
                    best_result = result
                    best_result["source_file"] = str(result_file)

            except Exception as e:
                log_message(f"Error reading {result_file}: {str(e)}")

        if best_result:
            # Save best overall result
            with open("metrics/best_params_overall.json", "w") as f:
                json.dump(best_result, f, indent=2)

            log_message(f"Best overall result: MSE = {best_mse}")
            log_message(f"Best configuration saved to metrics/best_params_overall.json")

        return best_result
