from analysis import analysis
from tqdm import tqdm
import json
from deep_translator import GoogleTranslator
import time
import re

def split_paragraph(paragraph, max_length=4000):
    """
    Split a paragraph into multiple complete sentences, each not exceeding max_length.
    
    Args:
        paragraph (str): The input paragraph.
        max_length (int, optional): The maximum length of each split. Defaults to 4000.
        
    Returns:
        list: A list of strings, each representing a part of the original paragraph.
              If the original paragraph is shorter than max_length, it is returned
              as the only element in the list.
    """
    
    # Define a regular expression pattern to match sentence boundaries
    # sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
    # Split the paragraph into sentences
    # sentences = re.split(sentence_pattern, paragraph)

    sentence_pattern = re.compile(r'([^.!?\n]+[.!?\n])')
    sentences = sentence_pattern.findall(paragraph)
    
    # If the paragraph is shorter than max_length, return it as a single-element list
    if len(paragraph) <= max_length:
        return [paragraph]
    
    # Initialize a list to store the parts
    parts = []
    current_part = ''
    
    # Iterate over sentences and accumulate them into parts
    for sentence in sentences:
        if len(current_part) + len(sentence) <= max_length:
            current_part += sentence
        else:
            parts.append(current_part)
            current_part = sentence
    
    # Append the last part
    if current_part:
        parts.append(current_part)
    
    return parts

def trans_api(translator, text):
        if text != '':
            try:
                trans_text = translator.translate(text)
                time.sleep(0.5)
            except Exception as e:
                print(e)
                #### text too long #####
                error = str(e)
                if 'Text length need to be between 0 and 5000 characters' in error:
                    parts = split_paragraph(text, max_length=3000)
                    trans_parts = []
                    for part in parts:
                        trans_part = translator.translate(part)
                        time.sleep(0.5)
                        trans_parts.append(trans_part)
                    af_text = ''
                    for trans_part in trans_parts:
                        af_text += trans_part
                    return af_text
                else:
                    trans_text = ''
                
                time.sleep(0.5)
            return trans_text
        else:
            return ''

translator_zh = GoogleTranslator(source='zh-CN', target='en')
translator_fr = GoogleTranslator(source='fr', target='en')

a = analysis('./log')
dataset = a._dataset_loader_()
context_idx = a._dataset_context_id_(dataset)

n = len(dataset)
output_zh = []
output_fr = []

with tqdm(total=n) as pbar:
    for i in range(n):
        data = dataset[i]
        idx = data['idx']
        question_id = data['question_id']
        if data['context'] == 'nan':
            context = dataset[context_idx[f'{data["language"]}_{data["main_question_id"]}']]['context']
        else:
            context = data['context']
        question = data['question']
        options = data['options']
        image_type = data['image_type']
        answers = data['answers']
        topic_difficulty = data['topic_difficulty']
        question_type = data['question_type']
        subfield = data['subfield']
        language = data['language']
        if language == 'english':
            language = 'English'
            translate_question = ''
            translate_context = ''
        elif language == 'chinese':
            language = 'Chinese'
            translate_question = trans_api(translator_zh, question)
            translate_context = trans_api(translator_zh, context)
        elif language == 'french':
            language = 'French'
            translate_question = trans_api(translator_fr, question)
            translate_context = trans_api(translator_fr, context)
        is_arithmetic = data['is_arithmetic']

        if question_type == 'open question':
            input = f'context: {context}\n\nquestion: {question}'
        elif question_type == 'multiple-choice':
            input = f'context: {context}\n\nquestion: {question}\n\noptions: {options}'
        log = {
            "idx": idx,
            "question_id": question_id,
            "context": context,
            "question": question,
            "options": options,
            "image_type": image_type,
            "answers": answers,
            "topic_difficulty": topic_difficulty,
            "question_type": question_type,
            "subfield": subfield,
            "language": language,
            "is_arithmetic": is_arithmetic,
            "translate_question": translate_question,
            "translate_context": translate_context,
        }
        if language == 'Chinese':
            output_zh.append(log)
        elif language == 'French':
            output_fr.append(log)
        pbar.update(1)
        
logs_zh = json.dumps(output_zh, ensure_ascii=False)
logs_fr = json.dumps(output_fr, ensure_ascii=False)

log_file_zh = open(f'./data/basic_zh.json', 'w', encoding='utf-8')
log_file_fr = open(f'./data/basic_fr.json', 'w', encoding='utf-8')

log_file_zh.write(logs_zh)
log_file_zh.close()

log_file_fr.write(logs_fr)
log_file_fr.close()