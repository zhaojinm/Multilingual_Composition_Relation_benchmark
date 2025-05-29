import json
import requests

API_KEY = 'YOURGOOGLEAPIKEY'

def translate_text(text, target_lang='zh-CN'):
    url = "https://translation.googleapis.com/language/translate/v2"
    data = {
        'q': text,
        'target': target_lang,
        'key': API_KEY
    }
    response = requests.post(url, data=data)
    print(response.json())
    if response.status_code != 200:
        return None
    return response.json()['data']['translations'][0]['translatedText']


def save_data(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def translate_json(input_file, output_file, target_lang='zh-CN'):
    '''
    example input:
    [
        {
            "question": "小明有一只猫，小红有一只狗，小明的猫是小红的？",
            "correct" : "不确定",
            "options": ["猫", "狗", "朋友", "家人", "不确定"],
            "category": ["人物关系"]
        },
        {
            "question": "小明有一只猫，小红有一只狗，小明的猫是小红的？",
            "correct" : "不确定",
            "options": ["猫", "狗", "朋友", "家人", "不确定"],
            "category": ["人物关系", "位置关系"]
        }
    ]
    '''
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for item in data:
        item['question'] = translate_text(item['question'], target_lang)
        item['correct'] = translate_text(item['correct'], target_lang)
        item['options'] = [translate_text(option, target_lang) for option in item['options']]
        
        if None in [item['question'], item['correct']] + item['options']:
            print("Translation failed.")
            save_data(data, output_file)
            return
    
    save_data(data, output_file)


# target_languages = ['ja']
target_languages = ['ko']
# target_languages = ['fr']
for lang in target_languages:
    # input_file = './dataset/template.json'
    input_file = './dataset/composition/composition_english.json'
    translate_json(input_file, f'{input_file}.translated_output.{lang}.json', lang)