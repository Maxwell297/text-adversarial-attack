from transformers import pipeline, AutoTokenizer, AutoModel
import jsonlines
import torch
import numpy as np

from toxicity import get_toxicity_score

# import sys
# sys.path.append('/cmlscratch/jryao/real-toxicity-prompts')
# from utils.perspective_api import PerspectiveWorker
# from pathlib import Path

'''
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModel.from_pretrained(model)
# input_ids = tokenizer.encode("Hello, my dog is cute", return_tensors="pt")
# outputs = model(input_ids)
raw_inputs = ['This is my cute']
inputs = tokenizer(raw_inputs, return_tensors="pt")
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)
exit(0)

# Convert input_ids to a floating point dtype
input_ids = input_ids.float()

# Compute gradients of outputs with respect to inputs
inputs = input_ids.requires_grad_(True)
grad_outputs = torch.ones_like(outputs[0])
grads = torch.autograd.grad(outputs=outputs, inputs=inputs, grad_outputs=grad_outputs)[0]

print(grads)
exit(0)
'''

model = "gpt2"
generator = pipeline("text-generation", model=model)
prompts_file = '../real-toxicity-prompts/realtoxicityprompts-data/generations/prompted/healthy_prompts.jsonl'
prompts = []
prompts_num = 100
results = []
num_return_sequences = 5
toxicities = []
# output_dir = Path('outputs')
# perspective_file = output_dir / 'perspective.jsonl'
# perspective = PerspectiveWorker(out_file=perspective_file,total=10*5,rate_limit=25)

with open(prompts_file, 'r') as f:
    prompts_cnt = 0
    for i, line in enumerate(jsonlines.Reader(f)):
        prompt = line["prompt"]["text"]
        prompts.append(prompt)
        prompts_cnt += 1
        res = generator(prompt, max_new_tokens=50, pad_token_id=50256, num_return_sequences=num_return_sequences)
        results.append(res)
        avg_toxicity = 0
        for j, item in enumerate(res):
            # perspective(f'{i}-{j}', item['generated_text'])
            toxicity = get_toxicity_score(item['generated_text'])['TOXICITY']
            avg_toxicity += toxicity
        avg_toxicity /= num_return_sequences
        toxicities.append(avg_toxicity)
        if prompts_cnt % 1 == 0:
            print(f'Step {prompts_cnt}')
        if prompts_cnt > prompts_num:
            break

# perspective.stop()

toxicity_of_clean_healthy_prompts = np.mean(toxicities)
print(toxicity_of_clean_healthy_prompts)

# prompt = "The man took off the hooker's clothers, touched"

# toxicity = get_toxicity_score(res)
# print(res)
# print(toxicity)
