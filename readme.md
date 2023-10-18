# Fine tuning LLMs for classification

Dataset used: 
https://huggingface.co/datasets/dair-ai/emotion
(Transformed in such a way that label is word not int, can be found compressed in the repo)

Order of operations:
1. split_dataset.py DATASET_PATH TRAIN_TEST_RATIO
2. (Optional) balance_dataset.py TRAIN_DATASET_PATH
3. create_transformers_dataset.py
4. train.py
5. serve.py
6. evaluate.py
7. generate.py
8. python -m wandb sweep sweep.yml

Some of these python files need to be edited for your use. This code is
a supplement of this workshop: https://www.youtube.com/watch?v=fyydvBcJTn8
(In slovak language)



