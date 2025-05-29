import argparse
import os
from util import *
import datetime
import pandas as pd

examplars = {
    "english": """
Q:Ruby is George's elder sister. Lucy's elder brother is George. What is Ruby to Lucy? A) Elder brother B) Elder sister C) Younger brother D) Younger sister E) Uncertain
A:B) Elder sister
Q:The hospital is 100 meters east of Jack's house, and the restaurant is 100 meters west of Jack's house. Where is the restaurant in relation to the hospital? Answer choices: A) East B) West C) Same location D) Uncertain
A:B) West
Q:If a region has rivers and flat terrain, then it must have a civilization. Region K is a plain between two rivers. Does Region K have a civilization? A) Yes B) No C)Not sure
A:A) Yes
Q:World of Warcraft has 500,000 more players than League of Legends. League of Legends has 500,000 fewer players than PUBG. How does the number of players of World of Warcraft compare to PUBG? Answer choices: A) 1,000,000 more players B) 1,000,000 fewer players C) Same number D) 7,100 more players E) Uncertain
A:C) Same number
Q:In a chemical reaction during an assignment, the ratio of A to B is 5:9, and the ratio of B to C is 2:7. If 63 grams of C are added, how many grams are added in total?A) 35 B) 14 C) 126 D) 91 E)Uncertain
A:D) 91
""",
    "chinese": """
Q:张冉是郭颖的姐姐，刘浩的哥哥是郭颖，张冉是刘浩的？A) 哥哥 B) 姐姐 C)弟弟 D) 妹妹 E) 不确定
A:B) 姐姐
Q:医院在小王家东方100米，饭店在小王家西方100米，饭店在医院哪里？Answer choices: A) 东方 B) 西方 C) 同一地点 D) 不确定
A:B) 西方
Q:假如一个地区有河流，且地势平坦，那么这个区域一定有文明，K地区是处于两条河流中间的平原，K地区有文明吗？A) 有 B)没有 C)不确定
A:A) 有
Q:《魔兽世界》玩家人数比《英雄联盟》多50万人，《英雄联盟》玩家人数比《绝地求生》少50万人，《魔兽世界》玩家人数比《绝地求生》？Answer choices: A) 多100万人 B) 少100万人 C) 一样多 D) 多0.71万人 E)不确定
A:C) 一样多
Q:发生作业中的化学反应，A和B的比例是5：9，B和C的比例是2：7，C加入了63克，总共加入了多少克？A) 35 B) 14 C) 126 D) 91 E)不确定
A:D) 91
""",
    "japanese": """
Q:ルビーはジョージの姉です。ルーシーの兄はジョージです。ルーシーにとってルビーはどんな存在ですか? A) 兄 B) 姉 C) 弟 D) 妹 E) 不明
A:B) 姉
Q:病院はジャックの家の東 100 メートルにあり、レストランはジャックの家の西 100 メートルにあります。病院とレストランの位置関係は? 回答の選択肢: A) 東 B) 西 C) 同じ場所 D) 不明
A:B) 西
Q:地域に川と平らな地形がある場合、そこには文明があるはずです。地域 K は 2 つの川に挟まれた平原です。地域 K には文明がありますか? A) はい B) いいえ C) わかりません
A:A) はい
Q:World of Warcraft のプレイヤー数は League of Legends より 50 万人多いです。League of Legends のプレイヤー数は PUBG より 50 万人少ないです。 World of Warcraft のプレイヤー数は PUBG と比べてどうですか? 回答の選択肢: A) 1,000,000 人多い B) 1,000,000 人少ない C) 同じ数 D) 7,100 人多い E) 不明
A:C) 同じ数
Q: 課題中の化学反応で、A と B の比率は 5:9、B と C の比率は 2:7 です。63 グラムの C を追加すると、合計で何グラム追加されますか?A) 35 B) 14 C) 126 D) 91 E) 不明
A:D) 91
""",
    "korean": """
Q: 루비는 조지의 언니입니다. 루시의 오빠는 조지입니다. 루비는 루시에게 어떤 존재입니까? A) 오빠 B) 언니 C) 남동생 D) 여동생 E) 불확실
A:B) 언니
Q: 병원은 잭의 집에서 동쪽으로 100m 떨어져 있고, 레스토랑은 잭의 집에서 서쪽으로 100m 떨어져 있습니다. 레스토랑은 병원과 어떤 관계가 있습니까? 답 선택지: A) 동쪽 B) 서쪽 C) 같은 위치 D) 불확실
A:B) 서쪽
Q: 지역에 강과 평평한 지형이 있으면 문명이 있어야 합니다. 지역 K는 두 강 사이의 평야입니다. 지역 K에 문명이 있습니까? A) 예 B) 아니요 C) 잘 모르겠습니다
A:A) 예
Q: 월드 오브 워크래프트는 리그 오브 레전드보다 50만 명 더 많은 플레이어가 있습니다. 리그 오브 레전드는 PUBG보다 50만 명 적은 플레이어가 있습니다. 월드 오브 워크래프트의 플레이어 수는 PUBG와 어떻게 비교됩니까? 답변 선택지: A) 1,000,000명 더 많은 플레이어 B) 1,000,000명 더 적은 플레이어 C) 동일한 숫자 D) 7,100명 더 많은 플레이어 E) 불확실
A:C) 동일한 숫자
Q:과제 중 화학 반응에서 A와 B의 비율은 5:9이고 B와 C의 비율은 2:7입니다. 63그램의 C를 추가하면 총 몇 그램이 추가됩니까?A) 35 B) 14 C) 126 D) 91 E) 불확실
A:D) 91
""",
    "french": """
Q:Ruby est la sœur aînée de George. Le frère aîné de Lucy s'appelle George. Que représente Ruby pour Lucy? A) Frère aîné B) Sœur aînée C) Frère cadet D) Sœur cadette E) Incertain
A:B) Sœur aînée
Q:L'hôpital est à 100 mètres à l'est de la maison de Jack et le restaurant est à 100 mètres à l'ouest de la maison de Jack. Où se trouve le restaurant par rapport à l'hôpital ? Choix de réponses : A) Est B) Ouest C) Même emplacement D) Incertain
A:B) Ouest
Q:Si une région a des rivières et un terrain plat, alors elle doit avoir une civilisation. La région K est une plaine entre deux rivières. La région K a-t-elle une civilisation ? A) Oui B) Non C) Je ne sais pas
A:A) Oui
Q:World of Warcraft compte 500 000 joueurs de plus que League of Legends. League of Legends compte 500 000 joueurs de moins que PUBG. Comment le nombre de joueurs de World of Warcraft se compare-t-il à celui de PUBG ? Choix de réponses : A) 1 000 000 de joueurs en plus B) 1 000 000 de joueurs en moins C) Même nombre D) 7 100 joueurs en plus E) Incertain
A:C) Même nombre
Q:Dans une réaction chimique au cours d'une tâche, le rapport entre A et B est de 5 : 9 et le rapport entre B et C est de 2 : 7. Si 63 grammes de C sont ajoutés, combien de grammes sont ajoutés au total ? A) 35 B) 14 C) 126 D) 91 E) Incertain
A:D) 91
""",
    "multilingual": """
Q:Ruby is George's elder sister. Lucy's elder brother is George. What is Ruby to Lucy? A) Elder brother B) Elder sister C) Younger brother D) Younger sister E) Uncertain
A:B) Elder sister
Q:医院在小王家东方100米，饭店在小王家西方100米，饭店在医院哪里？Answer choices: A) 东方 B) 西方 C) 同一地点 D) 不确定
A:B) 西方
Q:地域に川と平らな地形がある場合、そこには文明があるはずです。地域 K は 2 つの川に挟まれた平原です。地域 K には文明がありますか? A) はい B) いいえ C) わかりません
A:A) はい
Q: 월드 오브 워크래프트는 리그 오브 레전드보다 50만 명 더 많은 플레이어가 있습니다. 리그 오브 레전드는 PUBG보다 50만 명 적은 플레이어가 있습니다. 월드 오브 워크래프트의 플레이어 수는 PUBG와 어떻게 비교됩니까? 답변 선택지: A) 1,000,000명 더 많은 플레이어 B) 1,000,000명 더 적은 플레이어 C) 동일한 숫자 D) 7,100명 더 많은 플레이어 E) 불확실
A:C) 동일한 숫자
Q:Dans une réaction chimique au cours d'une tâche, le rapport entre A et B est de 5 : 9 et le rapport entre B et C est de 2 : 7. Si 63 grammes de C sont ajoutés, combien de grammes sont ajoutés au total ? A) 35 B) 14 C) 126 D) 91 E) Incertain
A:D) 91
"""
}

def parse_arguments():
    parser = argparse.ArgumentParser(description="compositional function reasoner")

    parser.add_argument(
        "--api_log_file_name", type=str, default=None, help="mandatory argument ! json['i>=1']['j==1']['k={1,2}'][{'request', response'}]"
    )

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")

    parser.add_argument(
        "--dataset", type=str, default="composition", choices=["composition"], help="dataset used for experiment"
    )

    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=1, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="llama2-13b", choices=["gpt3.5","gpt4","text-davinci-002", "llama2-7b", "llama2-13b", "mistral7b", "mistral8x7b"], help="model used for decoding. Note that 'gpt3' are the smallest models."
    )

    parser.add_argument(
        "--method", type=str, default="zero_shot", choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "few_shot_english", "few_shot_multilingual"], help="method"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=128, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help=""
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--language", type=str, default="english", choices=["chinese", "english", "japanese", "korean", "french"], help="language of dataset"
    )
    parser.add_argument(
        "--show_question_only", action="store_true", help="show question only"
    )

    args = parser.parse_args()

    if args.show_question_only:
        print("generate question only ...")
        args.dataset_path = f"./dataset/template.json"
    elif args.dataset == "composition":
        args.dataset_path = f"./dataset/composition/composition_{args.language}.json"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    args.direct_answer_trigger = "\nTherefore, among options, the answer is"
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "A:"
    args.cot_trigger = "Let's think step by step."

    return args

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')

    fix_seed(args.random_seed)
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    decoder = Decoder()
    outfilepath = './output/'+args.model+'/'+args.language+'/'+args.method + '/'
    total = 0
    correct_list = []
    output = {
            'question': [],
            'gen': [],
            'cat':[],
            'out':[],
            'out_clean':[],
            'answer':[],
            'correct':[]
            }
    if args.language=='chinese':
        acc_count = {"其他推理关系":0,"人物关系":0, "identity关系":0,"数学关系":0,"位置关系":0,"比较关系":0}
    elif args.language=='english' or args.language=='korean' or args.language=='french' or args.language=='japanese':
        acc_count = {"Other":0,"Personal":0, "Identity":0,"Mathematical":0,"Positional":0,"Comparative":0}

    for i, data in enumerate(dataloader):
        print('\n*************************')
        print("No.{} data".format(i+1))

        # Prepare question template ...
        # x, y -> question, answer
        x, y, cat = data
        question = x[0]

        x = "Q: " + x[0] + "\n" + "A:"
        y = y[0].strip()
        cat = [c[0] for c in cat]

        # for rereding paper's method just add the followingy line
        # x = x + "read this question again"+x

        # prompt -> question + "The answer is"
        if args.method == "zero_shot":
            question_prompt = x + " " + args.direct_answer_trigger_for_zeroshot
        # prompt -> question + "let us think step by step"
        elif args.method == "zero_shot_cot":
            question_prompt = x + " " + args.cot_trigger
        # prompt -> question + A:
        elif args.method == "few_shot":
            question_prompt = examplars[args.language] + x
        elif args.method == "few_shot_english":
            question_prompt = examplars["english"] + x
        elif args.method == "few_shot_multilingual":
            question_prompt = examplars["multilingual"] + x
        else:
            print(args.method)
            raise ValueError("method is not properly defined ...")
        
        if args.show_question_only:
            print("Question only ...")
            print(question_prompt)
            print("Ground True : " + y)
            continue

        max_length = args.max_length_cot if "cot" in args.method else args.max_length_direct
        out_raw = decoder.decode(args, question_prompt, max_length)
        output['gen'].append(out_raw)
        output['out'].append(out_raw)
        out_cleaned = answer_cleansing(args, out_raw)
        
        # If there are multiple answers, cleanup with gpt3.5
        if len(out_cleaned) == 0 or len(out_cleaned) > 1:
            print("multiple answers detected ... cleanup with openai gpt ...")
            print("before cleanup : {}".format(out_cleaned))
            prompt2 = "Following is a pair of a question and an answer. Only output the choice option solely based on the provided answer. Be concise. Output nothing if the questions is not answered\n\n" + x + " " + out_raw + "\n\nThe choice made by the answer is"
            max_length = args.max_length_direct
            out_cleaned = decoder.decode(args, prompt2, max_length,cleansing=True)
            out_cleaned = answer_cleansing(args, out_cleaned)
            print("after cleanup : {}".format(out_cleaned))

        # Choose the most frequent answer from the list ...
        output['question'].append(question)
        output['out_clean'].append(out_cleaned)
        output['answer'].append(y)
        output['cat'].append(cat)
        print("clean pred : {}".format(out_cleaned))
        print("Ground True : " + y)
        print("Category:", cat)
        print('*************************')

        # Checking answer ...
        if len(out_cleaned)>0 and y[0] == out_cleaned[0]:
            correct=1
        else:
            correct=0
        output['correct'].append(correct)
        for c in cat:
            print(c)
            acc_count[c]+=correct

        correct_list.append(correct)
        total += 1

        if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
            print("Stop !! limit_dataset_size applied ...")
            break

    if args.show_question_only:
        return

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    print(acc_count)
    print(sum(correct_list))
    time_tag = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(f"accuracy_{time_tag}.log", 'a') as f:
        f.write(f"{args.method} {args.model} {args.language} {accuracy}\n")
        f.write(str(acc_count))
        f.write("\n")
        f.write(str(sum(correct_list)))
        f.write("\n")

    # Save results ...
    if not os.path.exists(outfilepath):
        os.makedirs(outfilepath)
    
    file_name = f"out_{args.model}_{args.method}_{args.language}.csv"
    filepath = os.path.join(outfilepath, file_name)

    if os.path.exists(filepath):
        filepath = filepath.replace(".csv", "_{}.csv".format(time_tag))

    df = pd.DataFrame.from_dict(output)
    df.to_csv(filepath)

    print("output saved to {}".format(filepath))

if __name__ == "__main__":
    main()
