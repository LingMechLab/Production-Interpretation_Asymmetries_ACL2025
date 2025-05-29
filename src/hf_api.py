import numpy as np
import torch
import torch.nn.functional as F


def hf_api(model, tokenizer, messages, max_new_tokens):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = tokenizer([text], return_tensors="pt").to('cuda')
    output = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(input_ids.input_ids, output)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # text_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    # new_text = text_output[len(prompt):].strip()
    return response


def hf_token_api(model, tokenizer, prompt, max_new_tokens):
    input_ids = tokenizer(prompt, return_tensors="pt").to('cuda')
    output = model.generate(
        **input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
        output_logits=True,
        return_dict_in_generate=True,
        temperature=0,
    )
    transition_scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )
    text_output = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0]
    new_text = text_output[len(prompt) :].strip()
    prob = 1
    for i in transition_scores[0]:
        prob *= np.exp(i.cpu().numpy())
    return new_text, prob


def hf_yes_no_api(model, tokenizer, prompt):
    yes_tokens = ['yes', 'Yes', 'YES', 'ĠYes', 'ĠYES', 'Ġyes']
    no_tokens = ['no', 'No', 'NO', 'ĠNo', 'Ġno', 'ĠNO']
    yes_token_indices = [tokenizer.convert_tokens_to_ids(token) for token in yes_tokens]
    no_token_indices = [tokenizer.convert_tokens_to_ids(token) for token in no_tokens]
    text = tokenizer.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
        next_token = tokenizer.convert_ids_to_tokens(next_token_id)
        prob_distributions = F.softmax(logits, dim=-1)
        yes_probs = prob_distributions[0, yes_token_indices].tolist()
        no_probs = prob_distributions[0, no_token_indices].tolist()

    yes_probs_sum = sum(yes_probs)
    no_probs_sum = sum(no_probs)
    return [yes_probs_sum, no_probs_sum, next_token[0]]
