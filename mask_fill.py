from transformers import (RobertaTokenizer,
    RobertaModel, BertTokenizer, BertForMaskedLM,
    pipeline, AutoModelForCausalLM, AutoTokenizer,
    GPT2TokenizerFast, GPT2LMHeadModel)
import jsonlines
import json
import jiwer
import torch
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import time
import matplotlib.pyplot as plt

from bert_score.utils import get_idf_dict
from src.utils import load_gpt2_from_dict

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

model = 'bert-base-uncased'
prompt_file = '../real-toxicity-prompts/realtoxicityprompts-data/generations/prompted/healthy_prompts_pure_with_mask.jsonl'
prompts = []
prompts_num = 100
start_index = 0
num_samples = 100
num_iters = 600
adv_loss = 'cw'
initial_coeff = 15
lr = 1e-1
batch_size = 1
embed_layer = -1
lam_sim = 0.00
lam_perp = 0.00
gumbel_samples = 1000
kappa = 5
max_new_tokens = 30
print_every = 50
gpt2_checkpoint_folder = 'result'
mask_token = '[MASK]'
early_stop_adv_loss = 0.2
topk_attack = 3

# generator = pipeline("text-generation", model=model)
generate_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
generate_tokenizer.model_max_length = 512
generate_model = BertForMaskedLM.from_pretrained(model).cuda()
# generator.model = generator.model.cuda()
mask_info = generate_tokenizer(mask_token)

ref_model = load_gpt2_from_dict("%s/transformer_wikitext-103.pth" % gpt2_checkpoint_folder, output_hidden_states=True).cuda()
with torch.no_grad():
    embeddings = generate_model.get_input_embeddings()(torch.arange(0, generate_tokenizer.vocab_size).long().cuda())
    mask_embedding = embeddings[mask_info.input_ids[1]]
    ref_embeddings = ref_model.get_input_embeddings()(torch.arange(0, generate_tokenizer.vocab_size).long().cuda())

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

dirty_words = []
dirty_words_ids = []
with open('data/profane_words.json', 'r') as f:
    dirty_words_ = json.load(f)
encoded_dirty_words = generate_tokenizer(dirty_words_)

for i in range(len(dirty_words_)):
    if len(encoded_dirty_words['input_ids'][i]) == 3:
        dirty_words.append(dirty_words_[i])
        dirty_words_ids.append(encoded_dirty_words['input_ids'][i][1])
print(dirty_words)
attack_succ = 0

for idx in range(start_index, end_index):
    # 注意：要使BertForMaskedLM正确预测[MASK]，需要在prompt结尾添加句号 '.'
    input_ids = encoded_dataset['input_ids'][idx]
    prompt = encoded_dataset['prompt'][idx]
    token_type_ids = encoded_dataset['token_type_ids'][idx]
    # clean_generation_prompt = prompt + '[MASK].'
    # clean_generation_inputs = generate_tokenizer(clean_generation_prompt, return_tensors='pt')
    # clean_input_ids = clean_generation_inputs['input_ids'].cuda()
    # clean_token_type_ids = clean_generation_inputs['token_type_ids'].cuda()
    # clean_input_ids = input_ids.copy()
    # tmp = clean_input_ids[-1]
    # clean_input_ids[-1] = mask_info['input_ids'][1]
    # clean_input_ids.append(tmp)
    # clean_token_type_ids = token_type_ids.copy()
    # tmp = token_type_ids[-1]
    # clean_token_type_ids[-1] = mask_info['token_type_ids'][1]
    # clean_token_type_ids.append(tmp)
    clean_input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
    clean_token_type_ids = torch.tensor(token_type_ids).unsqueeze(0).cuda()
    clean_generation_logits = generate_model(input_ids=clean_input_ids, token_type_ids=clean_token_type_ids).logits
    # retrieve index of [MASK]
    mask_token_index = (clean_input_ids == generate_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    predicted_token_id = clean_generation_logits[0, mask_token_index].argmax(axis=-1)
    clean_generation = generate_tokenizer.decode(predicted_token_id)
    
    print('Prompt:', prompt)
    print('Clean mask prediction:', clean_generation)

    forbidden = np.zeros(len(input_ids)).astype('bool')
    forbidden[0] = True
    forbidden[-3:] = True
    forbidden_indices = np.arange(0, len(input_ids))[forbidden]
    forbidden_indices = torch.from_numpy(forbidden_indices).cuda()
    token_type_ids_batch = clean_token_type_ids.repeat(batch_size, 1)

    start_time = time.time()
    with torch.no_grad():
        orig_output = ref_model(torch.LongTensor(input_ids).unsqueeze(0).cuda()).hidden_states[embed_layer] # 1*21*768
        ref_weights = torch.FloatTensor([idf_dict[idy] for idy in input_ids]).cuda()
        ref_weights /= ref_weights.sum()
        log_coeffs = torch.zeros(len(input_ids), embeddings.size(0))
        indices = torch.arange(log_coeffs.size(0)).long()
        tmp_initial_coeff = torch.ones(len(input_ids)) * initial_coeff
        # 确保[cls, mask, ., eos]被采样的概率为1
        tmp_initial_coeff[0] = 100
        tmp_initial_coeff[-3:] = 100
        log_coeffs[indices, torch.LongTensor(input_ids)] = tmp_initial_coeff
        log_coeffs = log_coeffs.cuda()
        log_coeffs.requires_grad = True

    optimizer = torch.optim.Adam([log_coeffs], lr=lr)
    start = time.time()

    # 寻找gradient最大的token作为attack target
    optimizer.zero_grad()
    tmp = log_coeffs.unsqueeze(0).repeat(batch_size, 1, 1)
    coeffs = my_gumbel_softmax(tmp, hard=False)
    inputs_embeds = (coeffs @ embeddings[None, :, :])
    pred = generate_model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids_batch).logits
    pred_mask = generate_tokenizer.convert_ids_to_tokens(pred.argmax(dim=-1).squeeze())[-3]
    pred_mask_prob = pred[0, -3].softmax(0)
    adv_loss = 1.0 - pred_mask_prob[dirty_words_ids].sum()
    ref_embeds = (coeffs @ ref_embeddings[None, :, :])
    pred = ref_model(inputs_embeds=ref_embeds)
    if lam_sim > 0:
        output = pred.hidden_states[embed_layer]
        ref_loss = -lam_sim * bert_score(orig_output, output, weights=ref_weights).mean()
    else:
        ref_loss = 0.0
    if lam_perp > 0:
        perp_loss = lam_perp * log_perplexity(pred.logits, coeffs)
    else:
        perp_loss = 0.0
    total_loss = adv_loss + ref_loss + perp_loss
    total_loss.backward()
    log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
    _, attack_indices = torch.topk(torch.norm(log_coeffs.grad, dim=1),topk_attack)
    log_coeffs.grad.index_fill_(0, torch.arange(len(input_ids)).cuda(), 0)
    forbidden = torch.from_numpy(np.ones(len(input_ids)).astype('bool')).cuda()
    forbidden[attack_indices] = False
    forbidden_indices = torch.arange(len(input_ids)).cuda()[forbidden]
    log_coeffs.requires_grad = False
    log_coeffs[forbidden_indices, torch.tensor(input_ids).cuda()[forbidden_indices]] = 100
    log_coeffs.requires_grad = True
    print('Attack indices:', attack_indices)

    for i in range(num_iters):
        # for plt_id in range(len(input_ids)):
        #     plt.scatter(np.arange(generate_tokenizer.vocab_size),log_coeffs[plt_id].softmax(0).cpu().detach().numpy())
        #     plt.savefig(f'figs/{i}-{plt_id}.png')
        #     plt.clf()
        #     plt.scatter(np.arange(len(dirty_words_ids)),log_coeffs[plt_id].softmax(0)[dirty_words_ids].cpu().detach().numpy())
        #     plt.savefig(f'figs/{i}-{plt_id}_dirty.png')
        #     plt.clf()
        # b0 = time.time()
        optimizer.zero_grad()
        tmp = log_coeffs.unsqueeze(0).repeat(batch_size, 1, 1)
        coeffs = my_gumbel_softmax(tmp, hard=False)
        inputs_embeds = (coeffs @ embeddings[None, :, :]) # 1*21*768
        # last_embed = inputs_embeds[:, -1]
        # inputs_embeds[:, -1] = mask_embedding
        # inputs_embeds = torch.cat((inputs_embeds, last_embed.unsqueeze(1)), dim=1)
        # e1 = time.time()
        # FIXME: 目前bug: generate_model.generate()会丢失梯度信息，导致adv_loss.backward无法back propagate到log_coeffs
        pred = generate_model(inputs_embeds=inputs_embeds, token_type_ids=token_type_ids_batch).logits
        # plt.scatter(np.arange(generate_tokenizer.vocab_size), pred.logits[0,-2].softmax(0).cpu().detach().numpy())
        # plt.savefig(f'figs/{i}_mask.png')
        # plt.clf()
        # plt.scatter(np.arange(len(dirty_words_ids)), pred.logits[0,-2].softmax(0)[dirty_words_ids].cpu().detach().numpy())
        # plt.savefig(f'figs/{i}_mask_dirty.png')
        # plt.clf()
        # mask_token_index = (clean_input_ids == generate_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        pred_mask = generate_tokenizer.convert_ids_to_tokens(pred.argmax(dim=-1).squeeze())[-3]

        # pred = generate_tokenizer.batch_decode(pred)
        # pred = [pred_[13:] for pred_ in pred]
        # origin_prompt = generate_tokenizer.batch_decode(generate_model.generate(inputs_embeds=inputs_embeds, max_new_tokens=len(input_ids)))
        # pred = [pred_[len(origin_prompt_):] for pred_, origin_prompt_ in zip(pred, origin_prompt)]
        # print('Sample prompt:', origin_prompt)
        # print('Sample mask:', pred_mask)
        pred_mask_prob = pred[0,-3].softmax(0)
        adv_loss = 1.0 - pred_mask_prob[dirty_words_ids].sum()
        # e2 = time.time()
        # toxicity_input = toxicity_tokenizer(pred, padding=True, truncation=True, return_tensors='pt')
        # toxicity_logits = toxicity_model(**toxicity_input).logits
        # adv_loss = (toxicity_logits[:,0] - toxicity_logits[:,1] + kappa).clamp(min=0).mean()
        # e3 = time.time()
        # print(f'TIME: coeff {e1-b0}, generate {e2-e1}, toxicity {e3-e2}')

        # similarity constraint
        ref_embeds = (coeffs @ ref_embeddings[None, :, :])
        pred = ref_model(inputs_embeds=ref_embeds)
        # e4 = time.time()

        if lam_sim > 0:
            output = pred.hidden_states[embed_layer]
            ref_loss = -lam_sim * bert_score(orig_output, output, weights=ref_weights).mean()
        else:
            ref_loss = 0.0
        # e5 = time.time()
        # (log) perplexity constraint
        if lam_perp > 0:
            perp_loss = lam_perp * log_perplexity(pred.logits, coeffs)
        else:
            perp_loss = 0.0
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
            print('Sample mask:', pred_mask)
            # print(f'Iteration: {i+1}, total (adv) loss: {adv_loss}, entropy: {entropy}, time: {time.time()-start}')
            # print(log_coeffs.shape)
            # print('-'*50)
            # print(coeffs.shape)

        if adv_loss < early_stop_adv_loss:
            break


        # Gradient step
        log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
        optimizer.step()
        # e9 = time.time()
        # print(f'TIME: backward {e7-e6}, entropy {e8-e7}, step {e9-e8}')

        # Log statistics
        # adv_losses[idx-start_index, i] = adv_loss.detach().item()
        # ref_losses[idx-start_index, i] = ref_loss.detach().item()
        # perp_losses[idx-start_index, i] = perp_loss.detach().item()
        # entropies[idx-start_index, i] = entropy.detach().item()
    
    times.append(time.time() - start_time)

    with torch.no_grad():
        for j in range(gumbel_samples):
            adv_ids = my_gumbel_softmax(log_coeffs, hard=True).argmax(1)
            # new_adv_ids = torch.zeros(adv_ids.shape[-1]+1, dtype=torch.long)
            # new_adv_ids[-1] = adv_ids[-1]
            # new_adv_ids[-2] = mask_info['input_ids'][1]
            # new_adv_ids[:-2] = adv_ids[:-1]
            # adv_ids = adv_ids.cpu().tolist()
            # print(adv_ids)
            # adv_text = generate_tokenizer.decode(adv_ids)
            # print(adv_text)
            # exit(0)
            # x = generate_tokenizer(adv_text, max_length=256, truncation=True, return_tensors='pt')
            # token_errors.append(wer(adv_ids, x['input_ids'][0]))
            # new_input_ids = torch.zeros(x['input_ids'].shape[-1]+1, dtype=torch.long)
            # new_input_ids[-1] = x['input_ids'][0][-1]
            # new_input_ids[-2] = mask_info['input_ids'][1]
            # new_input_ids[:-2] = x['input_ids'][0][:-1]
            # new_token_type_ids = torch.zeros(x['token_type_ids'].shape[-1]+1, dtype=torch.long)
            # new_token_type_ids[-1] = x['token_type_ids'][0][-1]
            # new_token_type_ids[-2] = mask_info['token_type_ids'][1]
            # new_token_type_ids[:-2] = x['token_type_ids'][0][:-1]
            # print(new_input_ids, new_token_type_ids)
            # exit(0)
            adv_generate_text = generate_model(input_ids=adv_ids.unsqueeze(0).cuda()) #,
                                    # token_type_ids=new_token_type_ids.unsqueeze(0).cuda())
            adv_mask_prob = adv_generate_text.logits[0,-3].softmax(0)[dirty_words_ids].sum()
            perturbed_prompt = generate_tokenizer.decode(adv_generate_text.logits[0].argmax(-1))
            adv_generate_text = generate_tokenizer.decode(adv_generate_text.logits[0,-3].argmax().unsqueeze(0))
            
            # adv_toxicity = toxicity_model(**toxicity_tokenizer(adv_generate_text, padding=True, truncation=True, return_tensors='pt')).logits[0]
            # adv_toxicity = F.softmax(adv_toxicity)[1]
            if adv_generate_text in dirty_words or j == gumbel_samples - 1:
                print('Perturbed prompt:', perturbed_prompt)
                print('Adv predicted word:', adv_generate_text)
                print('Adv word prob:', adv_mask_prob)
                if adv_generate_text in dirty_words:
                    print('Attack succeed')
                    attack_succ += 1
                break

    # print(adv_loss)
print('Successful attack rate:', attack_succ / (end_index - start_index))
print('Total prompts:', end_index - start_index)

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
