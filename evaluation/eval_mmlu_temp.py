import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel


# from crop import crop

openai.api_key = "INSERTYOURKEYHERE"
SUBJECTS = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics', 'clinical_knowledge', 'college_biology', 'college_chemistry', 'college_computer_science', 'college_mathematics', 'college_medicine', 'college_physics', 'computer_security', 'conceptual_physics', 'econometrics', 'electrical_engineering', 'elementary_mathematics', 'formal_logic', 'global_facts', 'high_school_biology', 'high_school_chemistry', 'high_school_computer_science'] #, 'high_school_european_history', 'high_school_geography', 'high_school_government_and_politics', 'high_school_macroeconomics', 'high_school_mathematics', 'high_school_microeconomics', 'high_school_physics', 'high_school_psychology', 'high_school_statistics', 'high_school_us_history', 'high_school_world_history', 'human_aging', 'human_sexuality', 'international_law', 'jurisprudence', 'logical_fallacies', 'machine_learning', 'management', 'marketing', 'medical_genetics', 'miscellaneous', 'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy', 'prehistory', 'professional_accounting', 'professional_law', 'professional_medicine', 'professional_psychology', 'public_relations', 'security_studies', 'sociology', 'us_foreign_policy', 'virology', 'world_religions']
choices = ["A", "B", "C", "D"]
INVALID = "[Invalid]"
default_model = "meta-llama/Llama-2-7b-hf"


def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    '''
    Add few-shot examples to the prompt.
    '''
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(choices[df.iloc[idx, k + 1]])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def extract_prediction(prompt, completion):
    if prompt not in completion:
        print(INVALID)
        return INVALID
    
    prediction = completion.split(prompt)[1].strip()
    if prediction.upper() in choices:
        return prediction
    else:
        print(INVALID + " : " + prediction)
        return INVALID + " : " + prediction
        
def get_inference(model, tokenizer, prompt):
    output = model.generate(
        **tokenizer(
            prompt,
            return_tensors='pt',
            return_token_type_ids=False,
        ).to('cuda'),
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def eval(args, model, tokenizer, subject, dev_df, test_df):
    targs = []
    preds = []
    completions = []

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        
        targ = choices[test_df.iloc[i, test_df.shape[1]-1]]
        completion = get_inference(model, tokenizer, prompt)
        pred = extract_prediction(prompt, completion)

        completions.append(completion)
        targs.append(targ)
        preds.append(pred)

    acc = np.mean(np.array(targs) == np.array(preds))
    return targs, preds, completions, acc

def main(args):
    subjects = SUBJECTS

    # create model
    model = AutoModelForCausalLM.from_pretrained(default_model, torch_dtype="auto", device_map="auto")
    if args.model_path != default_model:
        model = PeftModel.from_pretrained(model, args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token  

    all_preds = []
    all_targs = []

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"))[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"))

        targs, preds, completions, acc = eval(args, model, tokenizer, subject, dev_df, test_df)
        
        all_preds += preds
        all_targs += targs
        test_df['prediction'] = preds
        test_df['completion'] = completions
        
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        test_df.to_csv(os.path.join(args.save_dir, subject + "_result.csv"))
        print("Average accuracy {:.3f} - {}".format(acc, subject))
        
    total_acc = np.mean(np.array(all_targs) == np.array(all_preds))
    print("Average accuracy: {:.3f} - total".format(total_acc))

def get_accuracy(args):
    subjects = sorted([f.split("_result.csv")[0] for f in os.listdir(args.save_dir) if "_result.csv" in f])
    
    all_preds = []
    all_targs = []
    for subject in subjects:
        result_df = pd.read_csv(os.path.join(args.save_dir, subject + "_result.csv"))
        
        targs = [choices[i] for i in result_df['answer']]
        preds = list(result_df['prediction'])
        all_preds += preds
        all_targs += targs
        
        acc = np.mean(np.array(targs) == np.array(preds))
        print("Average accuracy {:.3f} - {}".format(acc, subject))
    
    total_acc = np.mean(np.array(all_targs) == np.array(all_preds))
    print("Average accuracy: {:.3f} - total".format(total_acc))


def save_to_csv(data_dir='data/mmlu'):
    for subject in SUBJECTS:
        dataset = load_dataset("cais/mmlu", subject)
        
        dev_df = pd.DataFrame({'question' : dataset['dev']['question'],
                               'choice0'  : [l[0] for l in dataset['dev']['choices']],
                               'choice1'  : [l[1] for l in dataset['dev']['choices']],
                               'choice2'  : [l[2] for l in dataset['dev']['choices']],
                               'choice3'  : [l[3] for l in dataset['dev']['choices']],
                               'answer' : dataset['dev']['answer'],
                              })
        
        test_df = pd.DataFrame({'question' : dataset['test']['question'],
                                'choice0'  : [l[0] for l in dataset['test']['choices']],
                                'choice1'  : [l[1] for l in dataset['test']['choices']],
                                'choice2'  : [l[2] for l in dataset['test']['choices']],
                                'choice3'  : [l[3] for l in dataset['test']['choices']],
                                'answer' : dataset['test']['answer'],
                               })
        dev_df.to_csv(os.path.join(data_dir, "dev", subject + "_dev.csv"), index=False)
        test_df.to_csv(os.path.join(data_dir, "test", subject + "_test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data/mmlu")
    parser.add_argument("--save_dir", "-s", type=str)
    parser.add_argument("--model_path", "-m", type=str, default=default_model)
    args = parser.parse_args()
    # main(args)
    get_accuracy(args)