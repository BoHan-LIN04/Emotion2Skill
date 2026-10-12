"""
Skill Libraries
===============
Defines the initial skill libraries for all four benchmarks:
  - WebShop  (8 skills, §A.2)
  - ALFWorld (6 skills, §A.2)
  - MATH     (6 skills, §A.2)
  - MBPP     (5 skills, §A.2)

Each skill has a name, a one-line description, and a full SOP.
SOPs are distilled from MASA training traces (WebShop/ALFWorld) or
from standard problem-solving strategies (MATH/MBPP).
"""

from __future__ import annotations

from emotion2skill.selection import Skill, SkillLibrary


# ─────────────────────────────────────────────────────────────────────────────
# WebShop — 8 skills
# ─────────────────────────────────────────────────────────────────────────────

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
Trigger: All options are set; product is ready to be added to cart.
Steps:
  1. Click the "Add to Cart" button.
  2. Navigate to the cart view.
  3. Verify: correct product name, quantity, price, and options displayed.
  4. If cart contains an incorrect item, remove it and restart from OptionSelect.
  5. Exit condition: cart contains exactly the required product with correct attributes.
""",
    ),

    Skill(
        name="PurchaseConfirm",
        description="Confirm all task constraints are met and execute the purchase.",
        sop="""\
Trigger: Cart is verified; ready to complete purchase.
Steps:
  1. Review the full task goal one final time.
  2. Check cart: product name, price ≤ budget, all required attributes present.
  3. If any constraint is violated, abort and return to the relevant earlier skill.
  4. Click "Buy Now" / "Confirm Purchase".
  5. Record the final reward signal.
  6. Exit condition: purchase confirmed and episode ends.
""",
    ),
]


def get_webshop_library() -> SkillLibrary:
    return SkillLibrary(skills=list(WEBSHOP_SKILLS))


# ─────────────────────────────────────────────────────────────────────────────
# ALFWorld — 6 skills (one per task type)
# ─────────────────────────────────────────────────────────────────────────────

ALFWORLD_SKILLS: list[Skill] = [

    Skill(
        name="PickObject",
        description="Navigate to an object and pick it up.",
        sop="""\
Trigger: Task requires picking up an object and placing it somewhere.
Steps:
  1. Identify the target object from the task description.
  2. Navigate to the most likely receptacle locations in priority order
     (check visible surfaces first, then open containers).
  3. If the object is visible, execute 'pick up <object>'.
  4. If not found after checking 3 locations, expand search systematically
     (open drawers, check cabinets).
  5. Confirm pick-up succeeded (object now in inventory).
  6. Navigate to the destination and place the object.
  7. Exit condition: object placed at the correct destination.
Recovery: If 'pick up' fails due to blocked path, move closer and retry.
""",
    ),

    Skill(
        name="LookObject",
        description="Examine an object under a light source or in a specific location.",
        sop="""\
Trigger: Task requires examining an object under a lamp or in a lit area.
Steps:
  1. Locate the target object using the same search procedure as PickObject.
  2. Pick up the object.
  3. Identify the nearest desk lamp or floor lamp.
  4. Navigate to the lamp and turn it on if it is off.
  5. Stand adjacent to the lamp and examine the object ('examine <object>
     under <lamp>').
  6. Exit condition: examination action confirmed.
Recovery: If no lamp is found, use the overhead light receptacle.
""",
    ),

    Skill(
        name="CleanObject",
        description="Pick up an object, clean it in a sink, and place it at the target.",
        sop="""\
Trigger: Task requires cleaning an object.
Steps:
  1. Locate and pick up the target object.
  2. Navigate to the sink.
  3. Place the object in the sink.
  4. Turn the tap on ('use faucet on <object>').
  5. Pick up the cleaned object.
  6. Navigate to the destination and place it.
  7. Exit condition: cleaned object placed at the target location.
Recovery: If the sink is occupied, remove the blocking item first.
""",
    ),

    Skill(
        name="HeatObject",
        description="Pick up an object, heat it in a microwave, and place it at the target.",
        sop="""\
Trigger: Task requires heating an object.
Steps:
  1. Locate and pick up the target object.
  2. Navigate to the microwave.
  3. Open the microwave door.
  4. Place the object inside.
  5. Close the door and turn the microwave on.
  6. Wait for the heating cycle, then open the door.
  7. Pick up the heated object (it may be described as "hot").
  8. Navigate to the destination and place it.
  9. Exit condition: heated object at the target location.
Recovery: If the microwave is already running, wait one step then open.
          If object is too large, check for oven as an alternative.
""",
    ),

    Skill(
        name="CoolObject",
        description="Pick up an object, cool it in the fridge, and place it at the target.",
        sop="""\
Trigger: Task requires cooling an object.
Steps:
  1. Locate and pick up the target object.
  2. Navigate to the fridge/refrigerator.
  3. Open the fridge door.
  4. Place the object inside.
  5. Close the door and wait one step.
  6. Open the fridge and pick up the cooled object.
  7. Navigate to the destination and place it.
  8. Exit condition: cooled object at the target location.
Recovery: If fridge is full, remove one item temporarily.
""",
    ),

    Skill(
        name="PickTwoObjects",
        description="Pick up two instances of an object and place both at the target.",
        sop="""\
Trigger: Task requires delivering two objects of the same type.
Steps:
  1. Locate the first instance of the target object.
  2. Pick it up and carry it to a temporary staging location near the destination.
  3. Place the first object at the staging area.
  4. Locate the second instance (check remaining receptacles).
  5. Pick up the second object.
  6. Navigate to the final destination.
  7. Place the second object at the destination.
  8. Return for the first object and place it at the destination.
  9. Exit condition: both objects placed at the destination.
Note: The agent can only carry one object at a time; staging is required.
Recovery: If a second instance cannot be found after checking all rooms,
          re-examine containers that were only partially inspected.
""",
    ),
]


def get_alfworld_library() -> SkillLibrary:
    return SkillLibrary(skills=list(ALFWORLD_SKILLS))


# ─────────────────────────────────────────────────────────────────────────────
# MATH — 6 skills
# ─────────────────────────────────────────────────────────────────────────────

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
    return SkillLibrary(skills=list(MATH_SKILLS))


# ─────────────────────────────────────────────────────────────────────────────
# MBPP — 5 skills
# ─────────────────────────────────────────────────────────────────────────────

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
    return SkillLibrary(skills=list(MBPP_SKILLS))


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

LIBRARY_REGISTRY: dict[str, callable] = {
    "webshop":  get_webshop_library,
    "alfworld": get_alfworld_library,
    "math":     get_math_library,
    "mbpp":     get_mbpp_library,
}


def get_library(benchmark: str) -> SkillLibrary:
    """
    Convenience factory.

    Parameters
    ----------
    benchmark : "webshop" | "alfworld" | "math" | "mbpp"
    """
    benchmark = benchmark.lower()
    if benchmark not in LIBRARY_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{benchmark}'. "
            f"Choose from: {list(LIBRARY_REGISTRY.keys())}"
        )
    return LIBRARY_REGISTRY[benchmark]()
