from transformers import pipeline
from transformers import AutoTokenizer , AutoModelForSequenceClassification

model_name = 'distilbert=base-uncased-finetuned-sst-2-english'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline('sentiment-analysis',
                      model = model,
                      tokenizer = tokenizer)

sequence = 'Using a Transformer network is neat'
print(tokenizer(sequence))
tokens = tokenizer.tokenize(sequence)
print(tokens)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
decoded_string = tokenizer.decode(ids)
print(decoded_string)
