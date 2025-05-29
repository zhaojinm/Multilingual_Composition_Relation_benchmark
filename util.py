import openai # For GPT-3 API ...
import random
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from statistics import mean
import re
import replicate
from retrying import retry
import llama_inference as llama

openai.api_key = "YOUROPENAIAPIKEYHERE"

def fix_seed(seed):
    # random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def data_reader(args):

    questions = []
    answers = []
    categoris = []
    if args.language=='chinese':
        categoris_count = {"其他推理关系":0,"人物关系":0, "identity关系":0,"数学关系":0,"位置关系":0,"比较关系":0}
    elif args.language=='english' or args.language=='korean' or args.language=='japanese' or args.language=='french':
        categoris_count = {"Other":0,"Personal":0, "Identity":0,"Mathematical":0,"Positional":0,"Comparative":0}
    else:
        raise ValueError("language is not properly defined ...")

    random_guess_prob = 0.0

    if args.dataset == "composition":
        with open(args.dataset_path, encoding='utf8') as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["question"].strip() 
                a = str(line["correct"])
                c = line["category"]
                for cat in c:
                    assert cat in categoris_count.keys()
                    categoris_count[cat]+=1
                categoris.append(c)
                choice = " Answer choices: "
                assert a in line["options"]
                assert line["options"].count(a) == 1
                random_guess_prob += 1/len(line["options"])
                for i in range(len(line["options"])):
                    choice += chr(65+i) + ") " + line["options"][i] + " "
                    if  line["options"][i] == a:
                        answers.append(chr(65+i)+") " + a)

                questions.append(q + choice)
    else:
        raise ValueError("dataset is not properly defined ...")

    q_len_list = []
    for q in questions:
        q_len_list.append(len(q.split(" ")))
    q_len_mean = mean(q_len_list)

    print("dataset : {}".format(args.dataset))
    print("data size : {}".format(len(answers)))
    print("average num of words for each sample : {}".format(q_len_mean))
    for cat in categoris_count.keys():
        print(cat," ", categoris_count[cat])
    return questions, answers, categoris


class MyDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        self.questions, self.answers, self.category= data_reader(args)
        self.len = len(self.questions)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        input = self.questions[index]
        output = self.answers[index]
        cat = self.category[index]

        return input, output, cat

def answer_cleansing(args, pred):

    if args.method in ("few_shot", "few_shot_cot"):
        preds = pred.split(args.direct_answer_trigger_for_fewshot)
        answer_flag = True if len(preds) > 1 else False
        pred = preds[-1]

    if args.dataset in ('composition'):
        pred = re.findall(r'A|B|C|D|E|F|G', pred)
    else:
        raise ValueError("dataset is not properly defined ...")

    return pred

@retry(stop_max_attempt_number=10)
# @RateLimiter(max_calls=1200, period=60)
def decoder_for_gpt3(args, input, max_length, temperature=0, cleansing=False):
    if cleansing:
        response = openai.ChatCompletion.create(model="gpt-4o-mini-2024-07-18", messages=[{"role": "user", "content": input}], temperature=temperature)
        return response["choices"][0]["message"]["content"].strip()
    elif args.model == "gpt3.5":
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo-1106", messages=[{"role": "user", "content": input}],temperature = temperature)
        return response["choices"][0]["message"]["content"].strip()
    elif args.model == "gpt4":
        response = openai.ChatCompletion.create(model="gpt-4-0613", messages=[{"role": "user", "content": input}],temperature = temperature)
        return response["choices"][0]["message"]["content"].strip()
    elif args.model == "text-davinci-002":
        response = openai.Completion.create(
        engine="davinci-002",temperature = temperature,
        max_tokens=max_length,
        prompt = input)
        return response.choices[0].text.strip()
    elif args.model == "llama2-7b":
        input = {
            "top_k": 2,
            "top_p": 1,
            "prompt": input,
            "temperature": 0,
            "max_new_tokens": 128,
            "repetition_penalty": 1
        }

        output = replicate.run(
            "meta/llama-2-7b-chat",
            input=input
        )
        return "".join(output)
    elif args.model == "llama2-13b":
        output = replicate.run(
            "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
            input={
                "debug": False,
                "top_k": 5,
                "top_p": 0.9,
                "prompt": input,
                "temperature": 0,
                "max_new_tokens": 128,
                "min_new_tokens": -1
            }
        )
        return "".join(output)
    elif args.model == "mistral7b":
        output = replicate.run(
            "mistralai/mistral-7b-instruct-v0.2:f5701ad84de5715051cb99d550539719f8a7fbcf65e0e62a3d1eb3f94720764e",
            input={
                "debug": False,
                "top_k": 5,
                "top_p": 0.9,
                "prompt": input,
                "temperature": 0,
                "max_tokens": 128,
                "min_tokens": -1,
                "seed": args.random_seed,
            }
        )
        return "".join(output)
    elif args.model == "mistral8x7b":
        output = replicate.run(
            "mistralai/mixtral-8x7b-instruct-v0.1:cf18decbf51c27fed6bbdc3492312c1c903222a56e3fe9ca02d6cbe5198afc10",
            input={
                "top_k": 50,
                "top_p": 0.9,
                "prompt": input,
                "temperature": 0,
                "max_new_tokens": 512,
                "prompt_template": "<s>[INST] {prompt} [/INST]"
            }
        )
        return "".join(output)
    else:
       raise ValueError("LLM not support")

class Decoder():
    def __init__(self):
       return

    def decode(self, args, input, max_length, temperature=0, cleansing=False):
        print("----prompt:",input)
        response = decoder_for_gpt3(args, input, max_length, temperature, cleansing)
        print("----output:", response)
        return response

def setup_data_loader(args):

    fix_seed(args.random_seed)
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed : {}".format(worker_seed))
    def seed_worker(worker_id):
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    g = torch.Generator()
    g.manual_seed(worker_seed)

    dataset = MyDataset(args)

    dataloader = torch.utils.data.DataLoader(dataset,
                                            shuffle=True,
                                            batch_size=args.minibatch_size,
                                            drop_last=False,
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            pin_memory=True)

    return dataloader