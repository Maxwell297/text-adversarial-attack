import json

file_name = ''
f = open(file_name, 'r')
prompts = json.load(f)
f.close()
print(prompts[0])