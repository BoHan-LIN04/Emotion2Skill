"""Fixed-manifest MATH evaluation with explicit answer extraction/equivalence."""
from __future__ import annotations
import ast
import re
from fractions import Fraction
from .common import BenchmarkStep, ManifestEnv

def extract_answer(text):
    boxes = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start, depth = match.end(), 1
        for pos in range(start, len(text)):
            depth += (text[pos] == "{") - (text[pos] == "}")
            if depth == 0:
                boxes.append(text[start:pos])
                break
    if boxes:
        return boxes[-1].strip()
    matches = re.findall(r"(?:final\s+answer|answer)\s*[:=]\s*(.+)", text, re.I)
    return (matches[-1] if matches else text.strip().splitlines()[-1] if text.strip() else "").strip().rstrip(".")

def normalize_answer(text):
    text = extract_answer(str(text)).strip().strip("$")
    text = text.replace(r"\left", "").replace(r"\right", "").replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    text = re.sub(r"\\(?:text|mathrm)\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\,", "").replace(r"\!", "").replace(" ", "")
    if re.match(r"^[A-Za-z]\s*=", text):
        text = text.split("=", 1)[1]
    return text

def _expression(text):
    text = normalize_answer(text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", text)
        text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\pi", "pi")
    text = re.sub(r"\\frac(-?\d)(\d)", r"((\1)/(\2))", text)
    text = text.replace("^", "**").replace("{", "(").replace("}", ")")
    text = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", text)
    text = re.sub(r"\)(?=[A-Za-z(])", ")*", text)
    if len(text) > 1024:
        raise ValueError("Answer exceeds symbolic grading limit")
    return ast.parse(text, mode="eval").body

def _symbolic(node, sympy):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sympy.Rational(str(node.value))
    if isinstance(node, ast.Name) and node.id in {"pi", "e"}:
        return sympy.pi if node.id == "pi" else sympy.E
    if isinstance(node, ast.Name) and re.fullmatch(r"[A-Za-z]{1,16}", node.id):
        return sympy.Symbol(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _symbolic(node.operand, sympy)
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp):
        left, right = _symbolic(node.left, sympy), _symbolic(node.right, sympy)
        if isinstance(node.op, ast.Add): return left + right
        if isinstance(node.op, ast.Sub): return left - right
        if isinstance(node.op, ast.Mult): return left * right
        if isinstance(node.op, ast.Div): return left / right
        if isinstance(node.op, ast.Pow):
            if not right.is_number or abs(float(right)) > 100:
                raise ValueError("Exponent exceeds grading limit")
            return left ** right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords:
        return sympy.sqrt(_symbolic(node.args[0], sympy))
    raise ValueError("Unsupported answer expression")

def answers_equivalent(prediction, gold):
    prediction, gold = normalize_answer(prediction), normalize_answer(gold)
    if prediction == gold:
        return True
    # Ordered tuples/vectors preserve order and interval boundary symbols.
    for opening, closing in (("(", ")"), ("[", "]"), ("\\{", "\\}")):
        if prediction.startswith(opening) and prediction.endswith(closing) and gold.startswith(opening) and gold.endswith(closing):
            def parts(value):
                values, depth, start = [], 0, len(opening)
                stop = len(value) - len(closing)
                for position in range(start, stop):
                    depth += (value[position] in "{[(") - (value[position] in "}])")
                    if value[position] == "," and depth == 0:
                        values.append(value[start:position]); start = position + 1
                values.append(value[start:stop])
                return values
            left, right = parts(prediction), parts(gold)
            if opening == "\\{":
                # Unordered finite sets, with one-to-one matching.
                if len(left) != len(right): return False
                remaining = list(right)
                for item in left:
                    match = next((i for i, other in enumerate(remaining) if answers_equivalent(item, other)), None)
                    if match is None: return False
                    remaining.pop(match)
                return True
            if len(left) > 1 or len(right) > 1:
                return len(left) == len(right) and all(answers_equivalent(a, b) for a, b in zip(left, right))
    try:
        return Fraction(prediction) == Fraction(gold)
    except (ValueError, ZeroDivisionError):
        pass
    try:
        import sympy
        return bool(sympy.simplify(_symbolic(_expression(prediction), sympy) - _symbolic(_expression(gold), sympy)) == 0)
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError, ImportError):
        return False

class MATHEnv(ManifestEnv):
    benchmark = "math"
    action_instruction = 'Solve the problem. Output final[answer] when ready; put the final mathematical answer in \\boxed{...}. Other output is retained as reasoning until the step limit.'
    def __init__(self, manifest_path=None, split="test", max_steps=15, mock=False):
        self._configure(manifest_path, split, mock, max_steps)
        if not mock:
            try:
                import sympy
            except ImportError as exc:
                raise RuntimeError("MATH equivalence grading requires sympy.") from exc
    def reset(self, task_idx=None):
        task = self._choose_task(task_idx)
        self._goal = "Synthetic: compute 1+1." if self.mock else task["problem"]
        self._current_obs = self._goal
        return self._current_obs, self._goal
    def step(self, action):
        self._step_count += 1
        done = action.startswith("final[") or r"\boxed{" in action or self._step_count >= self.max_steps
        answer = action[6:-1] if action.startswith("final[") and action.endswith("]") else action
        gold = "2" if self.mock else self._task["answer"]
        success = done and answers_equivalent(answer, gold)
        info = {"prediction": action, "extracted_answer": extract_answer(answer), "gold_answer": gold,
                "grader": "exact-normalized/safe-sympy-v1", "synthetic": self.mock}
        self._current_obs = "Answer recorded." if done else "Reasoning recorded. Continue or submit final[answer]."
        return BenchmarkStep(self._current_obs, float(success), done, bool(success), info)
