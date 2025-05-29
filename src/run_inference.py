import pandas as pd
from hf_api import *
from prompt_template import *
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai_api import openai_api


def run_no_pronoun_results(data_path, model_name, output_path, use_openai=False):
    data = pd.read_csv(data_path)
    if not use_openai:
        model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        subject = row['subject']
        object = row['object']
        sentence = row['stimuli']
        no_pronoun_binary_choice_prompt = (
            no_pronoun_binary_choice_prompt_template.format(subject, object, sentence)
        )
        no_pronoun_yes_no_prompt = no_pronoun_yes_no_prompt_template.format(
            subject, sentence
        )
        no_pronoun_continuation_prompt = no_pronoun_continuation_prompt_template.format(
            sentence
        )
        if use_openai:
            binary_choice_response = openai_api(
                model_name, no_pronoun_binary_choice_prompt, use_prob=True
            )
            yes_no_response = openai_api(
                model_name, no_pronoun_yes_no_prompt, use_prob=True
            )
            continuation_response = openai_api(
                model_name, no_pronoun_continuation_prompt
            )
        else:
            binary_choice_response = hf_token_api(
                model, tokenizer, no_pronoun_binary_choice_prompt, 1
            )
            yes_no_response = hf_yes_no_api(model, tokenizer, no_pronoun_yes_no_prompt)
            continuation_response = hf_api(
                model, tokenizer, no_pronoun_continuation_prompt, 32
            )

        data.loc[i, 'binary_choice_response'] = binary_choice_response[0]
        data.loc[i, 'binary_choice_probability'] = binary_choice_response[1]
        data.loc[i, 'yes_no_response'] = yes_no_response[0]
        data.loc[i, 'yes_no_probability'] = yes_no_response[1]
        data.loc[i, 'continuation_response'] = continuation_response

    data.to_csv(output_path, index=False)


def run_pronoun_results(data_path, model_name, output_path, use_openai=False):
    data = pd.read_csv(data_path)
    if not use_openai:
        model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i, row in tqdm(data.iterrows(), total=data.shape[0]):
        subject = row['subject']
        object = row['object']
        sentence = row['stimuli']
        pronoun_binary_choice_prompt = pronoun_binary_choice_prompt_template.format(
            subject, object, sentence
        )
        pronoun_yes_no_prompt = pronoun_yes_no_prompt_template.format(subject, sentence)
        pronoun_continuation_prompt = pronoun_continuation_prompt_template.format(
            sentence
        )

        if use_openai:
            binary_choice_response = openai_api(
                model_name, pronoun_binary_choice_prompt, use_prob=True
            )
            yes_no_response = openai_api(
                model_name, pronoun_yes_no_prompt, use_prob=True
            )
            continuation_response = openai_api(model_name, pronoun_continuation_prompt)
        else:
            binary_choice_response = hf_token_api(
                model, tokenizer, pronoun_binary_choice_prompt, 1
            )
            yes_no_response = hf_yes_no_api(model, tokenizer, pronoun_yes_no_prompt)
            continuation_response = hf_api(
                model, tokenizer, pronoun_continuation_prompt, 32
            )

        data.loc[i, 'binary_choice_response'] = binary_choice_response[0]
        data.loc[i, 'binary_choice_probability'] = binary_choice_response[1]
        data.loc[i, 'yes_no_response'] = yes_no_response[0]
        data.loc[i, 'yes_no_probability'] = yes_no_response[1]
        data.loc[i, 'continuation_response'] = continuation_response

    data.to_csv(output_path, index=False)


if __name__ == '__main__':
    run_no_pronoun_results(
        'datasets/no_pronoun.csv',
        'gpt-4o-2024-11-20',
        'results/rohde_kehler_2014_no_pronoun_results_gpt4o.csv',
        use_openai=True,
    )
    run_pronoun_results(
        'datasets/pronoun.csv',
        'gpt-4o-2024-11-20',
        'results/rohde_kehler_2014_pronoun_results_gpt4o.csv',
        use_openai=True,
    )
