from transformers import (RobertaTokenizer,
    RobertaForSequenceClassification,
    pipeline, AutoModelForCausalLM,
    GPT2TokenizerFast, GPT2LMHeadModel)
import jsonlines
import jiwer
import torch
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import time
import matplotlib.pyplot as plt

from bert_score.utils import get_idf_dict

def wer(x, y):
    x = " ".join(["%d" % i for i in x])
    y = " ".join(["%d" % i for i in y])

    return jiwer.wer(x, y)

def bert_score(refs, cands, weights=None):
    refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
    if weights is not None:
        refs_norm *= weights[:, None]
    else:
        refs_norm /= refs.size(1)
    cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
    cosines = refs_norm @ cands_norm.transpose(1, 2)
    # remove first and last tokens; only works when refs and cands all have equal length (!!!)
    cosines = cosines[:, 1:-1, 1:-1]
    R = cosines.max(-1)[0].sum(1)
    return R

def log_perplexity(logits, coeffs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_coeffs = coeffs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
    return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()

def my_gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

model = 'gpt2'
prompt_file = '../real-toxicity-prompts/realtoxicityprompts-data/generations/prompted/healthy_prompts_pure.jsonl'
prompts = []
prompts_num = 100
start_index = 0
num_samples = 100
num_iters = 100
adv_loss = 'cw'
initial_coeff = 15
lr = 1
batch_size = 1
embed_layer = -1
lam_sim = 1
lam_perp = 1
gumbel_samples = 10
kappa = 5
max_new_tokens = 30
print_every = 1

# generator = pipeline("text-generation", model=model)
generate_tokenizer = GPT2TokenizerFast.from_pretrained(model)
generate_tokenizer.model_max_length = 512
generate_tokenizer.padding_side = 'right'
generate_model = GPT2LMHeadModel.from_pretrained(model).cuda()
# generator.model = generator.model.cuda()
generate_tokenizer.pad_token = generate_tokenizer.eos_token
generate_model.config.pad_token_id = generate_model.config.eos_token_id

ref_model = AutoModelForCausalLM.from_pretrained(model, output_hidden_states=True).cuda()
with torch.no_grad():
    embeddings = generate_model.get_input_embeddings()(torch.arange(0, generate_tokenizer.vocab_size).long().cuda())
    ref_embeddings = ref_model.get_input_embeddings()(torch.arange(0, generate_tokenizer.vocab_size).long().cuda())

# load toxicity classification tokenizer and model weights
toxicity_tokenizer = RobertaTokenizer.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')
toxicity_model = RobertaForSequenceClassification.from_pretrained('SkolkovoInstitute/roberta_toxicity_classifier')

dataset = load_dataset('json', data_files=prompt_file, split=f'train[:{prompts_num}]')
preprocess_function = lambda examples: generate_tokenizer(examples['prompt'], max_length=256, truncation=True)
encoded_dataset = dataset.map(preprocess_function, batched=True)
idf_dict = get_idf_dict(dataset['prompt'], generate_tokenizer, nthreads=20)
# print(encoded_dataset[:2])

adv_log_coeffs, clean_texts, adv_texts = [], [], []
clean_logits = []
adv_logits = []
token_errors = []
times = []

end_index = min(start_index + num_samples, len(encoded_dataset))
adv_losses, ref_losses, perp_losses, entropies = torch.zeros(end_index - start_index, num_iters), torch.zeros(end_index - start_index, num_iters), torch.zeros(end_index - start_index, num_iters), torch.zeros(end_index - start_index, num_iters)

for idx in range(start_index, end_index):
    input_ids = torch.tensor(encoded_dataset['input_ids'][idx])
    prompt = encoded_dataset['prompt'][idx]
    token_type_ids = None
    clean_generation = generate_tokenizer.batch_decode(generate_model.generate(input_ids=input_ids.unsqueeze(0).cuda(), max_new_tokens=max_new_tokens))
    clean_generation = [clean_generation[0][len(prompt):]]
    # clean_generation = generator(prompt)[0]['generated_text']
    clean_toxicity = toxicity_model(**toxicity_tokenizer(clean_generation, padding=True, truncation=True, return_tensors='pt')).logits[0]
    clean_toxicity = F.softmax(clean_toxicity)[1]
    print('Prompt:', prompt)
    print('Clean generation:', clean_generation)
    print('Clean toxicity:', clean_toxicity)
    # adv_loss = 1.0 - torch.softmax(clean_toxicity, 0)[1]

    forbidden = np.zeros(len(input_ids)).astype('bool')
    forbidden[0] = True
    forbidden[-1] = True
    forbidden_indices = np.arange(0, len(input_ids))[forbidden]
    forbidden_indices = torch.from_numpy(forbidden_indices).cuda()
    token_type_ids_batch = None

    start_time = time.time()
    with torch.no_grad():
        orig_output = ref_model(torch.LongTensor(input_ids).unsqueeze(0).cuda()).hidden_states[embed_layer] # 1*21*768
        ref_weights = torch.FloatTensor([idf_dict[idy] for idy in input_ids]).cuda()
        ref_weights /= ref_weights.sum()
        log_coeffs = torch.zeros(len(input_ids), embeddings.size(0))
        indices = torch.arange(log_coeffs.size(0)).long()
        log_coeffs[indices, torch.LongTensor(input_ids)] = initial_coeff
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad = True

    optimizer = torch.optim.Adam([log_coeffs], lr=lr)
    start = time.time()
    for i in range(num_iters):
        for plt_id in range(len(input_ids)):
            plt.scatter(np.arange(generate_tokenizer.vocab_size),log_coeffs[plt_id].cpu().detach().numpy())
            plt.savefig(f'figs/{i}-{plt_id}-pure_adv.png')
            plt.clf()
        # b0 = time.time()
        optimizer.zero_grad()
        tmp = log_coeffs.unsqueeze(0).repeat(batch_size, 1, 1)
        coeffs = my_gumbel_softmax(tmp, hard=False)
        inputs_embeds = (coeffs @ embeddings[None, :, :]) # 1*21*768
        # e1 = time.time()
        # FIXME: 目前bug: generate_model.generate()会丢失梯度信息，导致adv_loss.backward无法back propagate到log_coeffs
        pred = generate_model.generate(inputs_embeds=inputs_embeds, max_new_tokens=max_new_tokens)
        pred = generate_tokenizer.batch_decode(pred)
        pred = [pred_[13:] for pred_ in pred]
        # origin_prompt = generate_tokenizer.batch_decode(generate_model.generate(inputs_embeds=inputs_embeds, max_new_tokens=len(input_ids)))
        # pred = [pred_[len(origin_prompt_):] for pred_, origin_prompt_ in zip(pred, origin_prompt)]
        # print('Sample prompt:', origin_prompt)
        print('Sample text:', pred)
        # e2 = time.time()
        toxicity_input = toxicity_tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
        toxicity_logits = toxicity_model(**toxicity_input).logits
        adv_loss = (toxicity_logits[:,0] - toxicity_logits[:,1] + kappa).clamp(min=0).mean()
        # e3 = time.time()
        # print(f'TIME: coeff {e1-b0}, generate {e2-e1}, toxicity {e3-e2}')

        # similarity constraint
        ref_embeds = (coeffs @ ref_embeddings[None, :, :])
        pred = ref_model(inputs_embeds=ref_embeds)
        # e4 = time.time()

        if lam_sim > 0:
            output = pred.hidden_states[embed_layer]
            ref_loss = -lam_sim * bert_score(orig_output, output, weights=ref_weights).mean()
        # e5 = time.time()
        # (log) perplexity constraint
        if lam_perp > 0:
            perp_loss = lam_perp * log_perplexity(pred.logits, coeffs)
        # e6 = time.time()
        # print(f'TIME: ref {e4-e3}, bert {e5-e4}, ppl {e6-e5}')
        total_loss = adv_loss + ref_loss + perp_loss
        # total_loss = adv_loss
        # adv_loss.backward()
        total_loss.backward()
        # e7 = time.time()

        entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))
        # e8 = time.time()
        if i % print_every == 0:
            print(f'Iteration: {i+1}, total loss: {total_loss}, adv_loss: {adv_loss}, ref_loss: {ref_loss}, perp_loss: {perp_loss}, entropy: {entropy}, time: {time.time()-start}')
            # print(f'Iteration: {i+1}, total (adv) loss: {adv_loss}, entropy: {entropy}, time: {time.time()-start}')
            # print(log_coeffs.shape)
            # print('-'*50)
            # print(coeffs.shape)


        # Gradient step
        log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
        optimizer.step()
        # e9 = time.time()
        # print(f'TIME: backward {e7-e6}, entropy {e8-e7}, step {e9-e8}')

        # Log statistics
        adv_losses[idx-start_index, i] = adv_loss.detach().item()
        ref_losses[idx-start_index, i] = ref_loss.detach().item()
        perp_losses[idx-start_index, i] = perp_loss.detach().item()
        entropies[idx-start_index, i] = entropy.detach().item()
    
    times.append(time.time() - start_time)

    with torch.no_grad():
        for j in range(gumbel_samples):
            adv_ids = my_gumbel_softmax(log_coeffs, hard=True).argmax(1)
            adv_ids = adv_ids.cpu().tolist()
            adv_text = generate_tokenizer.decode(adv_ids)
            x = generate_tokenizer(adv_text, max_length=256, truncation=True, return_tensors='pt')
            token_errors.append(wer(adv_ids, x['input_ids'][0]))
            adv_generate_text = generate_model.generate(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                                    max_new_tokens=max_new_tokens)
            adv_generate_text = generate_tokenizer.batch_decode(adv_generate_text)
            adv_generate_text = [adv_generate_text[0][len(adv_text):]]
            adv_toxicity = toxicity_model(**toxicity_tokenizer(adv_generate_text, padding=True, truncation=True, return_tensors='pt')).logits[0]
            adv_toxicity = F.softmax(adv_toxicity)[1]
            if adv_toxicity > 0.5 or j == gumbel_samples - 1:
                print('Adv generated text', adv_generate_text)
                print('Adv toxicity:', adv_toxicity)
                break

    # print(adv_loss)
    exit(0)

# with open(prompt_file, 'r') as f:
#     prompt_cnt = 0
#     for line in jsonlines.Reader(f):
#         prompt = line['prompt']['text']
#         prompts.append(prompt)
#         prompt_cnt += 1
#         if prompt_cnt >= prompts_num:
#             break


print('done')
exit(0)



# prepare the input
batch = tokenizer(["she went to the library", "he is a douchebag"], padding=True, truncation=True, return_tensors='pt')

# inference
res = model(**batch)
print(res)
