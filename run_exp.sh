# french
python main.py --model text-davinci-002 --method zero_shot      --limit_dataset_size 0 --language french &
python main.py --model gpt3.5           --method zero_shot      --limit_dataset_size 0 --language french &
python main.py --model llama2-7b        --method zero_shot      --limit_dataset_size 0 --language french &
python main.py --model llama2-13b       --method zero_shot      --limit_dataset_size 0 --language french &
python main.py --model mistral7b        --method zero_shot      --limit_dataset_size 0 --language french &
python main.py --model text-davinci-002 --method zero_shot_cot  --limit_dataset_size 0 --language french &
python main.py --model gpt3.5           --method zero_shot_cot  --limit_dataset_size 0 --language french &
python main.py --model llama2-7b        --method zero_shot_cot  --limit_dataset_size 0 --language french &
python main.py --model llama2-13b       --method zero_shot_cot  --limit_dataset_size 0 --language french &
python main.py --model mistral7b        --method zero_shot_cot  --limit_dataset_size 0 --language french &
python main.py --model text-davinci-002 --method few_shot       --limit_dataset_size 0 --language french &
python main.py --model gpt3.5           --method few_shot       --limit_dataset_size 0 --language french &
python main.py --model llama2-7b        --method few_shot       --limit_dataset_size 0 --language french &
python main.py --model llama2-13b       --method few_shot       --limit_dataset_size 0 --language french &
python main.py --model mistral7b        --method few_shot       --limit_dataset_size 0 --language french &

# gpt4
python main.py --model gpt4             --method zero_shot      --limit_dataset_size 0 --language english &
python main.py --model gpt4             --method zero_shot_cot  --limit_dataset_size 0 --language english &
python main.py --model gpt4             --method few_shot       --limit_dataset_size 0 --language english &
python main.py --model gpt4             --method zero_shot      --limit_dataset_size 0 --language chinese &
python main.py --model gpt4             --method zero_shot_cot  --limit_dataset_size 0 --language chinese &
python main.py --model gpt4             --method few_shot       --limit_dataset_size 0 --language chinese &
python main.py --model gpt4             --method zero_shot      --limit_dataset_size 0 --language french &
python main.py --model gpt4             --method zero_shot_cot  --limit_dataset_size 0 --language french &
python main.py --model gpt4             --method few_shot       --limit_dataset_size 0 --language french &
python main.py --model gpt4             --method zero_shot      --limit_dataset_size 0 --language japanese &
python main.py --model gpt4             --method zero_shot_cot  --limit_dataset_size 0 --language japanese &
python main.py --model gpt4             --method few_shot       --limit_dataset_size 0 --language japanese &
python main.py --model gpt4             --method zero_shot      --limit_dataset_size 0 --language korean &
python main.py --model gpt4             --method zero_shot_cot  --limit_dataset_size 0 --language korean &
python main.py --model gpt4             --method few_shot       --limit_dataset_size 0 --language korean &

# few_shot_english
python main.py --model gpt3.5           --method few_shot_english       --limit_dataset_size 0 --language english &
python main.py --model gpt3.5           --method few_shot_english       --limit_dataset_size 0 --language chinese &
python main.py --model gpt3.5           --method few_shot_english       --limit_dataset_size 0 --language french &
python main.py --model gpt3.5           --method few_shot_english       --limit_dataset_size 0 --language japanese &
python main.py --model gpt3.5           --method few_shot_english       --limit_dataset_size 0 --language korean &

# few_shot_multilingual
python main.py --model gpt3.5           --method few_shot_multilingual       --limit_dataset_size 0 --language english &
python main.py --model gpt3.5           --method few_shot_multilingual       --limit_dataset_size 0 --language chinese &
python main.py --model gpt3.5           --method few_shot_multilingual       --limit_dataset_size 0 --language french &
python main.py --model gpt3.5           --method few_shot_multilingual       --limit_dataset_size 0 --language japanese &
python main.py --model gpt3.5           --method few_shot_multilingual       --limit_dataset_size 0 --language korean &
