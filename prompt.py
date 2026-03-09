
def get_prompt(query, entity_a, entity_b, rel_b):
    prompt = f"""Query: {query}

Task: Identify how two real-world entities are connected by producing MANY connection paths. A connection path is a sequence of factual triples (head, relationship, tail) forming a continuous chain that begins with one entity and ends with a required target condition.

You MUST generate as many distinct valid paths as possible. Within each individual path, prefer STRONG connections (highly exclusive, specific relationships). Across the full set of paths, maintain DIVERSITY: include both popular/well-known connections and less well-known “trivia” connections, and avoid over-concentrating on the most obvious domain (e.g., for a celebrity, do not only use their main profession—add distinct non-professional connections when available).

Path definition:
- Every path MUST start with the head entity: '{entity_a}'
- Every path MUST end with a triple whose relationship is '{rel_b}' and whose tail entity is '{entity_b}'
- Paths may be direct or indirect and may include one or more intermediate entities

Rules and quality constraints:
- Entities must be concrete, real-world entities only (people, organizations, works, places, genes, diseases, species, etc.). No abstract concepts.
- Do not ask follow-up questions; respond using the best available factual knowledge.
- Temporal connections are allowed (relationships may span different historical periods).
- Disambiguation is required: use canonical names and qualifiers where necessary (e.g., 'Michael Jordan (basketball)').
- If multiple canonical entities share the same name, explore ALL of them explicitly where relevant.

Deduplication:
- Do not repeat the same path.
- Do not repeat the same triple within a single path.
- Prefer paths that are meaningfully different (different intermediate nodes and/or different relationships), not trivial rephrasings.

Coverage & diversity:
- Generate as many distinct valid paths as you can.
- Explore a broad range of relationship types for '{entity_a}'.
- Include BOTH:
  (a) strong/obvious connections (the first things most people would think of), AND
  (b) less well-known but still factual connections (“trivia”) that are distinct from the popular ones.
- After you have produced several paths in a dominant domain (e.g., movies/acting for an actor), actively search for other distinct domains (e.g. philanthropy) when possible.

Relationship quality guidance:
- Prefer strong, specific, and distinctive relationships.
  - Strong = highly exclusive (e.g., parent/child, founder-of, spouse, authored, CEO-of, member-of a small group).
  - Weaker = shared broad attributes (e.g., “attended”, “lives in”, “worked on” in very large productions).
- In each individual path, prioritize strong links early in the chain when possible.
- Across paths, start with strong + distinctive paths, then include progressively more general/weaker but still valid paths to maximize coverage.

Output requirements (strict):
- Return ONLY a JSON object wrapped in <answer> tags. Do not include any explanatory text.
- The JSON object must use integer keys starting from 1.
- Each triple must be of the form: (head entity, relationship, tail entity).
- Relationship strings must be 1–3 words.
- If no valid path exists, return an empty JSON object.
- Don’t worry too much if the connections are non-factual. Make your best guess and always return some paths if you can brainstorm any at all.

Enumerate all distinct valid connection paths that satisfy the above constraints.
"""
    return prompt