from transformers import pipeline

classifier = pipeline('sentiment-analysis')
res = classifier('There is weather today')
print(res)

generator = pipeline('text-generation' , model = 'distilgpt2')
res2 = generator(
    'In this course, we will teach you how to',
    max_length = 30,
    num_return_sequences = 2
)
print(res2)

classifier2 = pipeline('zero-shot-classification')
res3 = classifier2(
    'This is a course about Python list comprehension',
    candidate_labels = ['education' , 'politics' , 'business']
)
print(res3)