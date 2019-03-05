from utils import *

new_text = "It is important to by very pythonly while you are pythoning with python. \
                All pythoners have pythoned poorly at least once."
print(getStemmedDocuments(new_text))

x = json_reader("train.json")

y = next(x)
print(y['stars'])
print(y['text'])

print(getStemmedDocuments(y['text']))
