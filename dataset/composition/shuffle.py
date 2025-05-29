# python3 dataset/composition/shuffle.py
import os
import json
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
random.seed(42)

with open(current_dir + '/composition_chinese.json', 'r', encoding='utf-8') as file:
    chinese_data_old = json.load(file)
with open(current_dir + '/composition_chinese_new.json', 'r', encoding='utf-8') as file:
    chinese_data_new = json.load(file)
with open(current_dir + '/composition_english.json', 'r', encoding='utf-8') as file:
    english_data_old = json.load(file)
with open(current_dir + '/composition_english_new.json', 'r', encoding='utf-8') as file:
    english_data_new = json.load(file)


output_chinese_data = chinese_data_new
output_english_data = english_data_new

if len(output_english_data) != len(output_chinese_data):
    raise ValueError("English and Chinese JSON Mismatch")

combined_data = list(zip(output_english_data, output_chinese_data))
random.shuffle(combined_data)
shuffled_english_data, shuffled_chinese_data = zip(*combined_data)
shuffled_english_data = list(shuffled_english_data)
shuffled_chinese_data = list(shuffled_chinese_data)

with open(current_dir + '/shuffled_chinese_new.json', 'w', encoding='utf-8') as f:
     json.dump(shuffled_chinese_data, f, ensure_ascii=False, indent=2)

with open(current_dir + '/shuffled_english_new.json', 'w', encoding='utf-8') as f:
     json.dump(shuffled_english_data, f, ensure_ascii=False, indent=2)