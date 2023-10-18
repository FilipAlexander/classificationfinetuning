import openai
import dotenv
import os
import json
import tqdm

dotenv.load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

base_prompt = """
We are retraining a classifier. It is a multi class classifier. The classes are: anger, fear, joy, love, sadness, surprise.
After training we measured that in lot of cases it confuses %ACTUAL% as %PREDICTED%. Please generate some more text of class %ACTUAL%. You
can take inspiration in style and lenght from %TEXT%. Output only text without additional information. It should be one sentence in lenght.
"""

def chat_with_gpt3_5_turbo(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message['content']

with open('confusion_dict.json', 'r') as fh:
    confusion_dict = json.load(fh)

with open('./data/emotion_dataset_generated_2.jsonl', 'a') as fh:
    for i in range(3):
        to_generate = [('fear','surprise'),('anger','fear'),('sadness','fear'),('sadness','anger'),('anger','sadness'),('joy','surprise')]
        for actual, predicted in to_generate:
            for text in tqdm.tqdm(confusion_dict[actual + '_confused_as_' + predicted]):
                promt = base_prompt.replace('%ACTUAL%', actual).replace('%PREDICTED%', predicted).replace('%TEXT%', text)
                new_text = chat_with_gpt3_5_turbo(promt)
                dct = {'label': actual, 'text': new_text}
                fh.write(json.dumps(dct) + '\n')
