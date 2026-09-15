import contextlib
import io
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from pkgs.scripts import run_experiments as runner


class RepetitionSchedulingTests(unittest.TestCase):
    def test_parallel_reps_keep_task_order_logs_and_failures_isolated(self):
        barrier = threading.Barrier(5)
        completed = {str(rep): [] for rep in range(1, 6)}

        def worker(command, *, cwd, env, stdout, stderr):
            rep = env["CKD_REP"]
            task = command[command.index("--analyses") + 1]
            self.assertEqual(command[command.index("--reps") + 1], rep)
            self.assertEqual(Path(stdout.name).parent, root / "generated_data" / f"rep{rep}")
            if task == "clinical_validity":
                # All five reps must be active before any first task finishes.
                barrier.wait(timeout=5)
                self.assertEqual(completed[rep], [])
            else:
                self.assertEqual(completed[rep], ["clinical_validity"])
            completed[rep].append(task)
            stdout.write(f"{rep}/{task}\n")
            return subprocess.CompletedProcess(command, int(rep == "2" and task == "clinical_validity"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "ROOT", root), patch.object(runner.subprocess, "run", side_effect=worker):
                with contextlib.redirect_stdout(io.StringIO()):
                    status = runner.main(["analyze", "--reps", "all", "--parallel-reps"])
            self.assertEqual(status, 1)
            for rep, tasks in completed.items():
                self.assertEqual(tasks, ["clinical_validity", "feature_importance"])
                logs = list((root / "generated_data" / f"rep{rep}").glob("*.log"))
                self.assertEqual(len(logs), 2)
                self.assertEqual({log.read_text() for log in logs}, {f"{rep}/{task}\n" for task in tasks})

    def test_default_runs_reps_sequentially(self):
        with patch.object(runner, "run_rep", return_value=[]) as run_rep:
            with patch.object(runner, "ThreadPoolExecutor") as executor:
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(runner.main(["train", "--reps", "all"]), 0)
        executor.assert_not_called()
        self.assertEqual([call.args[1] for call in run_rep.call_args_list], [1, 2, 3, 4, 5])

    def test_parallel_dry_run_creates_no_files_or_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(runner, "ROOT", root), patch.object(runner.subprocess, "run") as worker:
                with patch.object(runner, "ThreadPoolExecutor") as executor:
                    with contextlib.redirect_stdout(io.StringIO()) as output:
                        status = runner.main(["analyze", "--reps", "all", "--parallel-reps", "--dry-run"])
            self.assertEqual(status, 0)
            worker.assert_not_called()
            executor.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])
            for rep in range(1, 6):
                self.assertIn(str(root / "generated_data" / f"rep{rep}"), output.getvalue())


if __name__ == "__main__":
    unittest.main()
