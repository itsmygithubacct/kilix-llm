"""Small shared primitives for assistant-only supervised language-model training."""


def encode_target(tokenizer, messages, answer, max_tokens):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("training target must be nonempty")
    if type(max_tokens) is not int or max_tokens < 2:
        raise ValueError("max_tokens must be at least two")
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                          enable_thinking=False, return_dict=False)
    target = tokenizer.encode(answer, add_special_tokens=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer needs an explicit end-of-turn token")
    if not prompt or not target:
        raise ValueError("empty encoded prompt or target")
    target.append(tokenizer.eos_token_id)
    if len(prompt)+len(target) > max_tokens:
        raise ValueError("training example exceeds context; refusing silent truncation")
    return {"input_ids": prompt+target, "labels": [-100]*len(prompt)+target}


def target_positions(labels):
    positions = [i for i, value in enumerate(labels) if value != -100]
    if not positions or positions[0] == 0:
        raise ValueError("causal targets need a preceding prompt token")
    return [i-1 for i in positions], [labels[i] for i in positions]


def target_loss(model, row, device):
    """Compute only target-token logits; model must support tensor logits_to_keep."""
    import torch
    positions, targets = target_positions(row["labels"])
    ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=torch.tensor(positions, dtype=torch.long, device=device))
    if tuple(output.logits.shape[:2]) != (1, len(targets)):
        raise ValueError("model did not honor selective target logits")
    return torch.nn.functional.cross_entropy(output.logits[0].float(),
                                              torch.tensor(targets, dtype=torch.long, device=device))
