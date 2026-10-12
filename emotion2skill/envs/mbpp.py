"""Fixed-manifest MBPP pass@1 in a fresh, time/resource-limited process.

Process isolation is not an OS security sandbox. Run real generated-code
benchmarks in a dedicated container/VM without credentials or mounted secrets.
"""
from __future__ import annotations
import json
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from .common import BenchmarkStep, ManifestEnv

def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S | re.I)
    return blocks[-1].strip() if blocks else text.strip()

_RUNNER = '''import json, math, sys, traceback
from pathlib import Path
payload = json.loads(sys.stdin.read())
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (payload["cpu_seconds"], payload["cpu_seconds"] + 1))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
except (ImportError, ValueError):
    pass
namespace = {"__name__": "__generated__"}
result = {"passed": False, "tests_passed": 0, "tests_total": len(payload["tests"])}
try:
    exec(compile(payload["setup"], "<test_setup>", "exec"), namespace)
    exec(compile(payload["code"], "<candidate>", "exec"), namespace)
    for test in payload["tests"]:
        exec(compile(test, "<test>", "exec"), namespace)
        result["tests_passed"] += 1
    result["passed"] = True
except BaseException as exc:
    result["error"] = type(exc).__name__ + ": " + str(exc)
Path("grading_result.json").write_text(json.dumps(result))
'''

def execute_code(code, tests, setup="", timeout=5.0):
    if timeout <= 0 or not tests:
        raise ValueError("Positive timeout and at least one test are required.")
    payload = {"code": code, "tests": tests, "setup": setup,
               "cpu_seconds": max(1, math.ceil(timeout))}
    with tempfile.TemporaryDirectory(prefix="emotion2skill-mbpp-") as directory:
        output = Path(directory) / "process_output.txt"
        process = None
        def stop_process_group():
            if process is None:
                return
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                elif process.poll() is None:
                    process.kill()
            except ProcessLookupError:
                pass
        try:
            with output.open("wb") as stream:
                process = subprocess.Popen([sys.executable, "-I", "-S", "-c", _RUNNER],
                    stdin=subprocess.PIPE, stdout=stream, stderr=subprocess.STDOUT,
                    cwd=directory, env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"},
                    start_new_session=True)
                process.communicate(json.dumps(payload).encode(), timeout=timeout)
            stop_process_group()  # Also reap children left behind by a candidate.
            grade_path = Path(directory) / "grading_result.json"
            grade = json.loads(grade_path.read_text()) if grade_path.exists() else {
                "passed": False, "error": f"Process exited without grading result ({process.returncode})"}
            if process.returncode != 0:
                grade["passed"] = False
            grade.update(returncode=process.returncode, timed_out=False,
                         stdout=output.read_text(errors="replace")[:65536])
        except subprocess.TimeoutExpired:
            stop_process_group()
            process.communicate()
            grade = {"passed": False, "timed_out": True, "error": "Execution timeout",
                     "stdout": output.read_text(errors="replace")[:65536]}
    grade["execution_mode"] = "separate-process/resource-limits"
    return grade

class MBPPEnv(ManifestEnv):
    benchmark = "mbpp"
    action_instruction = 'Write Python code that solves the task. Submit with final[Python code] or a ```python fenced block. Only one submitted candidate is graded (pass@1); no test feedback retries.'
    def __init__(self, manifest_path=None, split="test", max_steps=15, mock=False, timeout=5.0):
        self._configure(manifest_path, split, mock, max_steps)
        self.timeout = timeout
    def reset(self, task_idx=None):
        task = self._choose_task(task_idx)
        self._finished = False
        self._goal = "Synthetic task: implement add(a, b)." if self.mock else task["text"]
        self._current_obs = self._goal
        return self._current_obs, self._goal
    def step(self, action):
        if getattr(self, "_finished", False):
            raise RuntimeError("MBPP pass@1 permits one graded candidate per task; reset before a new task.")
        self._step_count += 1
        # Intermediate reasoning may be recorded, but only the first submitted code is run.
        submitted = action.startswith("final[") or "```" in action or self._step_count >= self.max_steps
        if not submitted:
            self._current_obs = "Reasoning recorded. Submit a single final[Python code] candidate."
            return BenchmarkStep(self._current_obs, 0.0, False, False, {})
        body = action[6:-1] if action.startswith("final[") and action.endswith("]") else action
        code = extract_code(body)
        tests = ["assert add(1, 2) == 3"] if self.mock else self._task["test_list"]
        setup = "" if self.mock else self._task.get("test_setup_code", "")
        grade = execute_code(code, tests, setup, self.timeout)
        self._finished = True
        grade.update(prediction=action, code=code, test_list=tests, test_setup_code=setup, synthetic=self.mock)
        passed = bool(grade["passed"])
        self._current_obs = "Candidate grading finished."
        return BenchmarkStep(self._current_obs, float(passed), True, passed, grade)
