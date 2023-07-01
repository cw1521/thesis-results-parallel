from multiprocessing import Pool
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.translate.bleu_score import sentence_bleu
from math import sqrt
from re import compile
from json import dump, load



def get_token():
   with open ('auth_token.json', 'r') as f:
      token = load(f)
   return token

AUTH_TOKEN = get_token()["token"]

BATCH_SIZE = 100
NUM_OF_PROCESSES = 30


ST_EN_CHECKPOINT = 'cw1521/st-en-10'
EN_ST_CHECKPOINT = 'cw1521/en-st-10'

DATASET_NAME = 'cw1521/en-st-small'

OUTPUT_FILE_PATH = 'output\\results-sm-10.json'






def perform_translation(batch_texts, model, tokenizer):
  # Generate translation using models
  translated = model.generate(**tokenizer(batch_texts, return_tensors="pt", padding=True))
  # Convert the generated tokens indices back into text
  translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

  return translated_texts



def text_to_vector(text):
  WORD = compile(r"\w+")
  words = WORD.findall(text)
  return Counter(words)

def get_cosine(vec1, vec2):
  vec1 = text_to_vector(vec1)
  vec2 = text_to_vector(vec2)
  intersection = set(vec1.keys()) & set(vec2.keys())
  numerator = sum([vec1[x] * vec2[x] for x in intersection])

  sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
  sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
  denominator = sqrt(sum1) * sqrt(sum2)

  if not denominator:
    return 0.0
  else:
    return float(numerator) / denominator




def jaccard_similarity(str1, str2):
    str1 = set(str1.split())
    str2 = set(str2.split())
    return float(len(str1 & str2)) / len(str1 | str2)




"""Calculate the Loss of State Information"""

def get_state_obj(state):
  state_obj = {}
  percepts = ['is_demoed', 'on_ground', 'ball_touched', 'boost_amount', 'position', 'direction',
          'speed', 'throttle', 'steer', 'jump', 'boost', 'handbrake']
  for percept in percepts:
    state = state.replace(' '+percept, '/r'+percept)
  state = state.split('/r')

  for elem in state:
    elem_list = elem.split(' ')
    if len(elem_list) == 2:
        state_obj[elem_list[0]] = elem_list[1]
    else:
        state_obj[elem_list[0]] = elem_list[1:]

  return state_obj


def calc_loss(p1, p2):
    p1 = get_state_obj(p1)
    p2 = get_state_obj(p2)
    bool_percepts = ['is_demoed', 'on_ground', 'ball_touched', 'throttle',
                    'steer', 'jump', 'boost', 'handbrake']
    num_percepts = ['boost_amount', 'speed']
    loss = 0.0
    for percept in bool_percepts:
        try:
            if p1[percept] != p2[percept]:
                loss += 1.0
        except:
            loss += 1.0
        for percept in num_percepts:
            try:
                diff = int(p1[percept])-int(p2[percept])
                loss += abs(diff)*0.01
            except:
                loss += 1
        for i in range(2):
            try:
                diff = int(p1['position'][i])-int(p2['position'][i])
                loss += abs(diff)*0.001
            except:
                loss += 1
    return loss



"""Combine the Text and Calculate the Scores"""

def combine_texts(original_texts, back_translated_batch):
    text = [
       {
      'target': x[0],
      'predicted': x[1],
      'cosine': get_cosine(x[0], x[1]),
      'loss': calc_loss(x[0], x[1]),
      'jaccard': jaccard_similarity(x[0], x[1]),
      'bleu': sentence_bleu(
                [x[0].replace('.', '').split(' ')],
                x[1].replace('.', '').split(' ')
           )
        }
    for x in zip(original_texts, back_translated_batch)
    ]
    return text



def get_model(checkpoint):
   return AutoModelForSeq2SeqLM.from_pretrained(
      checkpoint,
      use_auth_token=AUTH_TOKEN
   )


def get_tokenizer(checkpoint):
   return AutoTokenizer.from_pretrained(
      checkpoint,
      use_auth_token=AUTH_TOKEN
    )




def perform_back_translation_with_augmentation(batch_texts):

  en_st_model = get_model(EN_ST_CHECKPOINT)
  en_st_tokenizer = get_tokenizer(EN_ST_CHECKPOINT)

  st_en_model = get_model(ST_EN_CHECKPOINT)
  st_en_tokenizer = get_tokenizer(ST_EN_CHECKPOINT)

  # Translate from State to English
  tmp_translated_batch = perform_translation(
    batch_texts,
    st_en_model,
    st_en_tokenizer
  )

  # Translate Back to State
  back_translated_batch = perform_translation(
     tmp_translated_batch,
     en_st_model,
     en_st_tokenizer
  )

  # Return The Final Result
  return combine_texts(batch_texts, back_translated_batch)




def export_results(results, file_name):
    with open(f'{file_name}.json', 'w') as f:
        dump(results, f)




def perform_batch_translation(args):
  print(f'Working on file: {args[1]}')
  results = perform_back_translation_with_augmentation(args[0])
  return results





def get_batch_dataset(texts, batch_size):
    dataset = []
    file_number = 0
    for i in range(0, len(texts), batch_size):
        temp_text = []
        file_number = file_number + 1
        # SET temp_text equal to the batch size IF enough elements exist in list
        # ELSE SET equal to elements left in list
        if len(texts[i:i+batch_size]) == batch_size:
            temp_text = texts[i:i+batch_size]
        else:
            temp_text = texts[i:]

        dataset.append((temp_text, file_number))
    return dataset
        
    



def pooled_batch_translation(texts, batch_size):
    dataset = get_batch_dataset(texts, batch_size)
    results = {}
    with Pool(processes=NUM_OF_PROCESSES) as pool:
        results['translations'] = pool.map(perform_batch_translation, dataset)
    export_results(results, OUTPUT_FILE_PATH)





if __name__ == '__main__':

  data = ['oracle-test.json']

  raw_data = load_dataset(
      DATASET_NAME,
      data_files={'test':data},
      use_auth_token=AUTH_TOKEN,
      field='data'
      )

  raw_states = raw_data['test']['input']

  pooled_batch_translation(raw_states, BATCH_SIZE)





