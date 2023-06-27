import json
import jsonlines

prompts = []
with open('healthy_prompts_pure.jsonl', 'r') as f:
    for line in jsonlines.Reader(f):
        prompts.append({'prompt': line['prompt'] + ' [MASK].'})

with jsonlines.open('healthy_prompts_pure_with_mask.jsonl', 'w') as f:
    for line in prompts:
        f.write(line)
exit(0)

healthy_prompts = []

with open('prompted_gens_gpt2.jsonl', 'r') as f:
    for line in jsonlines.Reader(f):
        prompt_toxicity = line['prompt']['toxicity']
        continuation_toxicity = line['continuation']['toxicity']
        generations = line['generations']
        generations_toxicity = 0
        generations_cnt = 0
        for generation in generations:
            generation_toxicity = generation['toxicity']
            if generation_toxicity is not None:
                generations_toxicity += generation_toxicity
                generations_cnt += 1
        if generations_cnt > 0:
            generations_toxicity /= generations_cnt
        # print(prompt_toxicity, continuation_toxicity, generations_toxicity)
        # exit(0)
        if (prompt_toxicity is not None and prompt_toxicity > 0.5) or (continuation_toxicity is not None and continuation_toxicity > 0.5) or (generation_toxicity is not None and generations_toxicity > 0.5):
            continue
        healthy_prompts.append(line)

with jsonlines.open('healthy_prompts.jsonl', 'w') as f:
    for prompt in healthy_prompts:
        f.write(prompt)

print(len(healthy_prompts))