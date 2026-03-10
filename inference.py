from typing import Any, Callable

# TODO: Upgrade to bespoke-curator for more efficient inference (batch/dedup).


def _create_litellm_generator(
    model_name: str,
    use_vllm: bool,
    server_url: str,
    generation_params: dict,
    parser: Callable[[str], Any] | None = None,
):
    """
    Create a generator that takes (prompt, parser=None) and returns (raw_response, parsed_response).
    Uses litellm; if use_vllm, routes to server_url; else uses model name only.
    parser: callable(raw_text) -> parsed. Default _extract_answer. Strength/factuality parsers can be plugged in.
    """
    import litellm

    api_base = f"{server_url.rstrip('/')}/v1" if (use_vllm and server_url) else None
    if use_vllm and not server_url:
        raise ValueError("--server_url is required when --vllm is set")

    def generate(prompt: str, parser: Callable[[str], Any] | None = None) -> tuple[str, float]:
        parse_fn = parser
        if not prompt:
            return "", 0.0
        try:
            r = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                api_base=api_base,
                **generation_params,
            )
            raw = r.choices[0].message.content if r.choices else ""
            hidden = getattr(r, "_hidden_params", None) or {}
            cost = hidden.get("response_cost")
            cost = float(cost) if cost is not None else 0.0
        except Exception as e:
            print(f"Inference error: {e}")
            raw = ""
            cost = 0.0
        return raw, cost
    return generate
