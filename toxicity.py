'''
    Use Perspective API to measure toxicity
'''

from googleapiclient import discovery
import json

API_KEY = 'AIzaSyDgfcNU26JdDxjEfkO1lG6hLA5x9oyl6RQ'

client = discovery.build(
  "commentanalyzer",
  "v1alpha1",
  developerKey=API_KEY,
  discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
  static_discovery=False,
)

properties = ['TOXICITY', 'SEVERE_TOXICITY', 'INSULT', 'PROFANITY',
              'IDENTITY_ATTACK', 'THREAT', 'SEXUALLY_EXPLICIT', 'FLIRTATION']

def get_toxicity_score(text):
    analyze_request = {
    'comment': { 'text': text },
    'requestedAttributes': {'TOXICITY': {}, 'SEVERE_TOXICITY': {}, 'INSULT': {}, 'PROFANITY': {}, 'IDENTITY_ATTACK': {}, 'THREAT': {},
                            'SEXUALLY_EXPLICIT': {}, 'FLIRTATION': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    # json.dumps(response, indent=2)
    res_dict = {}
    for property_ in properties:
        score = response['attributeScores'][property_]['summaryScore']['value']
        res_dict[property_] = score
    return res_dict

# print(get_toxicity_score('Fuck you'))