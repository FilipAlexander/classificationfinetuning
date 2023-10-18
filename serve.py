from transformers import pipeline, AutoTokenizer, RobertaForSequenceClassification
from sanic import Sanic, response
from sanic.response import json
VERSION = 'labcafe_1'

p = pipeline('text-classification',
             model=RobertaForSequenceClassification.from_pretrained(
                 './finished_models/xlm-roberta-base-lr_1e-05-bs_16-epochs_5-grad_acc_1-weight_decay_0.3_weighted'),
             tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-large'), truncation=True,
             max_length=512, device='cpu')

app = Sanic("Labcafe-demo")


@app.route('/version', methods=['POST'])
async def version(request):
    return json({'version':VERSION})

@app.route('/classify', methods=['POST'])
async def test(request):
    result = {'version':VERSION}
    try:
        text = request.json['text']
    except KeyError:
        return json({'status': 'bad_request', 'message': 'No text provided.','version':VERSION}, 400)  # Bad Request

    try:
        cats = p(text, top_k=5)
        result['result'] = cats
    except Exception as e:
        return json({'status': 'error', 'message': str(e),'version':VERSION}, 500)  # Internal Server Error

    return json(cats)


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8534, access_log=False)