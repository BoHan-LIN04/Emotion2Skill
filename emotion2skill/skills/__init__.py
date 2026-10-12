"""Hand-authored initial SOP libraries for WebShop, ALFWorld, MATH and MBPP."""

from __future__ import annotations

from emotion2skill.selection import Skill, SkillLibrary

INITIAL_LIBRARY_METADATA = {"source_type": "hand_authored",
    "historical_experiment_equivalence": "unverified", "schema_version": 1}


def _copy_library(skills):
    return SkillLibrary([Skill(item.name, item.sop, item.description) for item in skills],
                        metadata=dict(INITIAL_LIBRARY_METADATA))


WEBSHOP_SKILLS: list[Skill] = [

    Skill(
        name="ProductSearch",
        description="Search for products using keywords from the task goal.",
        sop="""\
Trigger: Start of episode or after a category is identified.
Steps:
  1. Parse the task goal to extract core product keywords, required attributes
     (color, size, material), and any budget constraint.
  2. Formulate a concise search query: 2–4 keywords, most specific first.
  3. Execute the search action.
  4. Scan the result titles and prices for relevance.
  5. Exit condition: ≥3 candidate products found that plausibly match the goal.
     If fewer than 2 results, hand off to QueryRephrase.
Recovery: If results are irrelevant, drop the least-specific keyword and retry once.
""",
    ),

    Skill(
        name="QueryRephrase",
        description="Reformulate the search query when initial results are poor.",
        sop="""\
Trigger: Fewer than 2 relevant results after ProductSearch, or explicit
         "No results found" message.
Steps:
  1. Identify which keyword caused over-filtering (usually the most specific one).
  2. Generate up to 3 alternative query formulations:
     (a) Drop the problematic keyword.
     (b) Use a synonym or category term.
     (c) Remove the price constraint and filter manually after retrieval.
  3. Execute each reformulated query in order until ≥2 results appear.
  4. Exit condition: candidate products found, or 3 attempts exhausted.
Recovery: If all 3 formulations fail, broaden to the parent category name only.
""",
    ),

    Skill(
        name="ProductCompare",
        description="Compare candidate products to select the best match for the goal.",
        sop="""\
Trigger: ≥2 candidate products identified; need to select one.
Steps:
  1. For each candidate, click through to the detail page and record:
     - Price, rating, color, size options, material, and availability.
  2. Build a comparison matrix against the task constraints (required attributes,
     budget ceiling).
  3. Eliminate candidates that violate hard constraints (wrong color, over budget).
  4. Among remaining candidates, prefer higher rating, then lower price.
  5. Exit condition: a single preferred product is identified.
""",
    ),

    Skill(
        name="PriceCompare",
        description="Find the cheapest option that satisfies all attribute constraints.",
        sop="""\
Trigger: Multiple compliant products exist; task goal specifies a budget or
         asks for the cheapest option.
Steps:
  1. Collect prices of all constraint-satisfying candidates.
  2. Sort ascending by price.
  3. Verify the cheapest item satisfies every attribute constraint.
  4. If it does, select it; otherwise move to the next cheapest.
  5. Exit condition: a single lowest-price compliant product is identified.
""",
    ),

    Skill(
        name="ReviewAnalysis",
        description="Analyse product reviews to infer quality and fit for the goal.",
        sop="""\
Trigger: Multiple products are within budget and share similar attributes;
         review sentiment can differentiate them.
Steps:
  1. For each candidate, read the top 5 reviews.
  2. Extract mentions of durability, sizing accuracy, color match, and
     any defects reported.
  3. Assign a qualitative score: positive / neutral / negative.
  4. Prefer the product with the most positive review signal.
  5. Exit condition: a single recommended product is identified based on reviews.
""",
    ),

    Skill(
        name="OptionSelect",
        description="Select the correct size, color, and quantity options on the product page.",
        sop="""\
Trigger: A product is selected; options (size, color, quantity) must be set
         before adding to cart.
Steps:
  1. Read the task goal for required size, color, and quantity.
  2. On the product page, locate the option selectors.
  3. For each attribute, click the correct option.
     - If the required option is out of stock, flag and return to ProductCompare.
  4. Verify all selected options match the task goal exactly.
  5. Exit condition: all options are correctly set and the item is ready to cart.
""",
    ),

    Skill(
        name="CartManage",
        description="Add the selected product to the cart and verify the cart state.",
        sop="""\
Trigger: A preferred product and its required options have been selected.
Steps:
  1. Remain on the chosen product page; WebShop has no separate cart workflow.
  2. Verify the selected size, color, quantity and displayed price against the task.
  3. If an option is wrong, click its displayed option label to correct it.
  4. If the product is unsuitable, use a displayed navigation button and compare alternatives.
  5. Exit condition: all required options and product constraints are verified.
""",
    ),

    Skill(
        name="PurchaseConfirm",
        description="Confirm all task constraints are met and execute the purchase.",
        sop="""\
Trigger: A product is selected and all required options are verified.
Steps:
  1. Re-read the complete task goal and check product identity, price and attributes.
  2. Correct a mismatched option or choose another product before purchasing.
  3. When every constraint is met, execute click[buy now] using the displayed button label.
  4. Exit condition: the environment terminates and supplies the final reward.
""",
    ),
]


def get_webshop_library() -> SkillLibrary:
    return _copy_library(WEBSHOP_SKILLS)


ALFWORLD_SKILLS: list[Skill] = [

    Skill(
        name="PickObject",
        description="Navigate to an object and pick it up.",
        sop="""\
Trigger: Pick one target object and place it at the requested destination.
Steps:
  1. Read the full task goal and locate likely receptacles with go to <receptacle N>.
  2. Inspect visible objects; open <container N> when necessary.
  3. Execute take <object N> from <receptacle N> using observed instance identifiers.
  4. Confirm the object is in inventory, then navigate to the task destination.
  5. Execute put <object N> in/on <destination N>.
  6. Exit condition: the goal placement is confirmed.
Recovery: Follow environment feedback and admissible action syntax; retry only after fixing the failed precondition.
""",
    ),

    Skill(
        name="LookObject",
        description="Examine an object under a light source or in a specific location.",
        sop="""\
Trigger: Examine the target object with a lamp as specified by the task.
Steps:
  1. Locate the object and take <object N> from <receptacle N>.
  2. Locate and go to the required lamp or its receptacle.
  3. While carrying the object, use <desklamp N> or use <floorlamp N> as available.
  4. Exit condition: the environment confirms the examination goal.
Recovery: Use only observed object and lamp identifiers; do not invent an examine-under command.
""",
    ),

    Skill(
        name="CleanObject",
        description="Pick up an object, clean it in a sink, and place it at the target.",
        sop="""\
Trigger: Clean a target object and place it at the requested destination.
Steps:
  1. Locate and take <object N> from <receptacle N>.
  2. Go to the observed sinkbasin while keeping the object in inventory.
  3. Execute clean <object N> with <sinkbasin N>.
  4. Confirm the cleaning result, then go to the destination.
  5. Execute put <object N> in/on <destination N>.
  6. Exit condition: the cleaned object is placed correctly.
Recovery: Inspect the environment feedback and repair the missing precondition before retrying.
""",
    ),

    Skill(
        name="HeatObject",
        description="Pick up an object, heat it in a microwave, and place it at the target.",
        sop="""\
Trigger: Heat a target object and place it at the requested destination.
Steps:
  1. Locate and take <object N> from <receptacle N>.
  2. Go to the observed microwave; inspect its state and open it when required.
  3. Execute heat <object N> with <microwave N> using the supported environment command.
  4. Confirm the heating result and retain the object in inventory.
  5. Go to the destination and put <object N> in/on <destination N>.
  6. Exit condition: the heated object is placed correctly.
Recovery: Use observed feedback and admissible commands; do not simulate waits or appliance cycles not supported by the environment.
""",
    ),

    Skill(
        name="CoolObject",
        description="Pick up an object, cool it in the fridge, and place it at the target.",
        sop="""\
Trigger: Cool a target object and place it at the requested destination.
Steps:
  1. Locate and take <object N> from <receptacle N>.
  2. Go to the observed fridge and open it when required by the environment.
  3. Execute cool <object N> with <fridge N>.
  4. Confirm the cooling result, then go to the destination.
  5. Execute put <object N> in/on <destination N>.
  6. Exit condition: the cooled object is placed correctly.
Recovery: Repair the precondition identified by the environment feedback before retrying.
""",
    ),

    Skill(
        name="PickTwoObjects",
        description="Pick up two instances of an object and place both at the target.",
        sop="""\
Trigger: Place two distinct instances of the requested object at the destination.
Steps:
  1. Locate the first instance and take <object N> from <receptacle N>.
  2. Carry it to the final destination and put <object N> in/on <destination N>.
  3. Record the placed instance ID to avoid selecting it again.
  4. Locate a second distinct instance and take it from its receptacle.
  5. Carry the second instance to the same destination and put it there.
  6. Exit condition: both distinct instances have the required placement.
Recovery: The agent carries one object at a time; place the first before taking the second.
""",
    ),
]


def get_alfworld_library() -> SkillLibrary:
    return _copy_library(ALFWORLD_SKILLS)


MATH_SKILLS: list[Skill] = [

    Skill(
        name="DirectSolve",
        description="Apply a known formula or definition directly (Level 1–2).",
        sop="""\
Trigger: Problem has a single formula or definitional step; Level 1–2.
Steps:
  1. Identify the relevant formula or theorem.
  2. Substitute problem values.
  3. Compute and simplify.
  4. State the answer with units if applicable.
""",
    ),

    Skill(
        name="AlgebraManip",
        description="Set up and manipulate equations to isolate the unknown.",
        sop="""\
Trigger: Problem requires setting up and solving one or more equations.
Steps:
  1. Define variables for all unknowns.
  2. Translate the problem statement into equations.
  3. Apply algebraic operations (substitution, elimination, factoring).
  4. Solve for the target variable.
  5. Verify by substituting back into the original equations.
""",
    ),

    Skill(
        name="Decompose",
        description="Break a complex problem into ordered sub-goals (Level 3–4).",
        sop="""\
Trigger: Problem involves multiple dependent steps or compound structures.
Steps:
  1. Read the full problem and identify the final answer type.
  2. List intermediate quantities needed to reach the answer.
  3. Order sub-goals by dependency.
  4. Solve each sub-goal in order, recording intermediate results.
  5. Combine sub-results to produce the final answer.
""",
    ),

    Skill(
        name="Casework",
        description="Exhaustively enumerate cases and analyse each separately.",
        sop="""\
Trigger: Problem has multiple distinct scenarios or conditions.
Steps:
  1. Identify the case-splitting variable or condition.
  2. List all possible cases (ensure exhaustiveness).
  3. Solve each case independently.
  4. Combine results (sum, min, or logical union as required).
  5. Check that cases are mutually exclusive and collectively exhaustive.
""",
    ),

    Skill(
        name="GeomVisualize",
        description="Build a mental geometric model and apply spatial reasoning.",
        sop="""\
Trigger: Problem involves geometric figures, coordinates, or spatial relationships.
Steps:
  1. Sketch or describe the figure based on given constraints.
  2. Label all known lengths, angles, and relationships.
  3. Identify the relevant geometric theorems (Pythagoras, similarity, etc.).
  4. Apply the theorem(s) to compute the unknown quantity.
  5. Verify the answer is geometrically consistent.
""",
    ),

    Skill(
        name="VerifyCheck",
        description="Re-derive the answer via an alternative method to verify correctness.",
        sop="""\
Trigger: Answer has been obtained; verification is needed (Level 4–5 or
         when first method was complex).
Steps:
  1. Choose an alternative approach (numerical check, symmetry, limiting case).
  2. Apply the alternative approach independently.
  3. Compare the two results.
  4. If they agree, report the answer with confidence.
  5. If they disagree, identify the error in the first approach and recompute.
""",
    ),
]


def get_math_library() -> SkillLibrary:
    return _copy_library(MATH_SKILLS)


MBPP_SKILLS: list[Skill] = [

    Skill(
        name="DirectImplement",
        description="Implement directly from docstring; straightforward translation.",
        sop="""\
Trigger: Function signature and docstring clearly specify the logic.
Steps:
  1. Read the docstring carefully; identify input types, output type, and
     any edge-case notes.
  2. Write the function body using standard library primitives.
  3. Ensure the return type matches the docstring.
  4. Run provided test cases mentally; adjust if a test would fail.
""",
    ),

    Skill(
        name="DecomposeGuard",
        description="Decompose into helper functions; add edge-case guards.",
        sop="""\
Trigger: Function logic is complex, involves multiple steps, or the
         docstring mentions edge cases (empty input, negative numbers, etc.).
Steps:
  1. Identify the main computation and any auxiliary computations.
  2. Write a helper function for each auxiliary computation.
  3. Add guards at the start of the main function for all mentioned edge cases
     (return early or raise as appropriate).
  4. Assemble the main function by calling helpers in order.
  5. Test against boundary cases (empty list, n=0, single element).
""",
    ),

    Skill(
        name="BruteForceOptimize",
        description="Write a correct brute-force first, then optimise.",
        sop="""\
Trigger: Optimal algorithm is unclear but a brute-force is easy to write.
Steps:
  1. Implement the naive O(n²) or exponential solution first.
  2. Verify correctness against sample inputs.
  3. Profile conceptually: identify the bottleneck loop.
  4. Apply a standard optimisation (memoisation, hash-set lookup, sorting
     + binary search).
  5. Replace the bottleneck; verify the optimised version matches the brute-force
     on small inputs.
""",
    ),

    Skill(
        name="PatternRecognize",
        description="Identify a standard pattern (sliding window, two-pointer, DP).",
        sop="""\
Trigger: Problem structure matches a classic algorithmic pattern.
Steps:
  1. Read the problem and classify: sequence, tree, graph, counting, optimisation.
  2. Match to a known pattern:
     - Contiguous subarray / substring → sliding window.
     - Sorted array, sum/difference targets → two-pointer.
     - Overlapping sub-problems, optimal substructure → dynamic programming.
     - Combinatorial counting → combinatorics formula or DP.
  3. Write the pattern-specific template code.
  4. Fill in problem-specific details (window condition, DP transition).
  5. Handle edge cases (empty input, size-1 input).
""",
    ),

    Skill(
        name="TestDriven",
        description="Write assert statements for boundary cases first, then implement.",
        sop="""\
Trigger: Problem has tricky boundary conditions or the docstring includes
         explicit test cases.
Steps:
  1. List all boundary cases: empty input, single element, maximum value,
     negative values, duplicates.
  2. Write assert statements for each boundary case.
  3. Implement the function body to pass all asserts.
  4. Run the provided doctest / test cases.
  5. If any assert fails, debug and fix before finalising.
""",
    ),
]


def get_mbpp_library() -> SkillLibrary:
    return _copy_library(MBPP_SKILLS)


LIBRARY_REGISTRY: dict[str, callable] = {
    "webshop":  get_webshop_library,
    "alfworld": get_alfworld_library,
    "math":     get_math_library,
    "mbpp":     get_mbpp_library,
}


def get_library(benchmark: str) -> SkillLibrary:

    benchmark = benchmark.lower()
    if benchmark not in LIBRARY_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Choose from: {list(LIBRARY_REGISTRY.keys())}"
        )
    return LIBRARY_REGISTRY[benchmark]()
