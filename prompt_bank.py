
# -----------------------------------------------------------------------------
# Prompts (from prompt_bank)
# -----------------------------------------------------------------------------

CLASS_SIZE_PROMPT = """Goal:
Given a knowledge graph path consisting of multiple triples of the form (subject, predicate, object), estimate—FOR EACH TRIPLE—the number of members in the LARGER of two possible classes defined by the predicate.

Formal Definitions (apply to each triple (s, p, o)):
Define TWO classes.

Class A (Subject-variation class):
    {{ x | (x, p, o) is true }}
This is the set of all possible SUBJECTS x that could replace s while keeping (p, o) fixed.
The given subject s counts as ONE MEMBER of this class.

Class B (Object-variation class):
    {{ y | (s, p, y) is true }}
This is the set of all possible OBJECTS y that could replace o while keeping (s, p) fixed.
The given object o counts as ONE MEMBER of this class.

Directional Discipline (CRITICAL):
- You MUST evaluate both Class A and Class B independently for each triple.
- NEVER collapse or reinterpret the relation.
- For directional predicates (e.g., 'influenced', 'taught', 'founded', 'won', 'received'):
    * Class A asks: who/what stands in relation (predicate) TO the object?
    * Class B asks: who/what the subject stands in relation (predicate) TO?
- Always respect the original direction of the predicate.

Procedure (apply to EACH triple in the path independently):
1. Identify Class A from (p, o).
2. Identify Class B from (s, p).
3. Use factual knowledge and reasonable estimation to estimate the size of EACH class.
4. Select the LARGER of the two estimated class sizes.
5. Explain your reasoning step by step for both classes.

Output instructions:
- Return valid JSON only.
- The output must be a JSON list, with one object per triple, in the same order as the path.
- Each object must contain exactly two keys:
  - "explanation": a natural-language explanation for that triple's Class A and Class B reasoning.
  - "judgment": a single integer representing the estimated size of the LARGER class.
- Do not include any text outside the JSON list.
- Do not include markdown or extra keys.
- Each judgment must be a single integer (no commas, no extra text).

The output must exactly match this schema:
[
  {{
    "explanation": "string",
    "judgment": integer
  }},
  ...
]

==== INPUT ====
Path: {path}
"""

TRIPLE_FACTUAL_CHECKING_PROMPT = """You are a fact-checking expert evaluating a multi-hop factual path for factual accuracy and logical validity.

The path consists of an ordered list of triples.
Each triple has the form: (subject, relation, object)

Your task is to evaluate EACH triple in the given path and determine whether it is hallucinated or not. Use the path only for entity disambiguation and contextual grounding. Do not infer unsupported facts.

Evaluation criteria (apply independently to each triple):

1. Entity existence and disambiguation:
   - Verify that both the subject and object are real, identifiable entities (people, places, organizations, events, or concepts).
   - Use the path context only when necessary to disambiguate entities.
   - If either entity is fabricated or incorrectly identified, mark the triple as hallucinated.

2. Relationship validity:
   - Verify that the stated relation describes a real, historically valid, currently valid, or plausibly valid relationship between the subject and object.
   - If the relationship is plausible and could reasonably have occurred, even if not directly verifiable, mark it as not hallucinated.
   - If it is niche information about the subject and only trivia heads would know compared to common knowledge then do not mark it as hallucinated.

3. Hallucination detection:
   - Mark the triple as hallucinated if:
     - An entity is fabricated or does not exist
     - The relation is fabricated, nonsensical, or incorrect
     - The asserted connection clearly contradicts well-known facts
     - The triple introduces made-up properties or roles

4. Directionality handling:
   - If the relation is ambiguous or commonly bidirectional, evaluate both directions.
   - If either direction corresponds to a true or plausible relationship, mark the triple as not hallucinated.

Output instructions:
- Return valid JSON only.
- Include exactly two keys: "explanation" and "judgments".
- Under "explanation", provide a natural-language explanation evaluating the triples in the path.
- Under "judgments", provide the final hallucination judgment for each triple as a list, in the same order as the path.
- Do not include any text outside the JSON object.
- Do not include markdown or extra keys.

The output must exactly match this schema:

{{
  "explanation": "string",
  "judgments": ["hallucinated" | "not hallucinated", ...]
}}

Input:
Path: {path}
"""

