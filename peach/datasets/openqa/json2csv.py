import logging
import os
import json
import csv

'''
json data
instance = {
                'dataset': str,
                'question': str,
                'answers': List[str],
                'positive_ctxs': Dict,
                'negative_ctxs': Dict,
                'hard_negative_ctxs': Dict
            }
positive_psg = {
                 'title': str, 
                 'text': str, 
                 'score': float,
                 'title_score': 0,
                 'passage_id' if args.task=='nq' else 'psg_id': passage['id'],
                 'has_answer': passage['has_answer']
                }
csv data
two columns of string
first column is question
second column is a string of list of answers (string)

'''


logger = logging.getLogger()

data_dir = '../data/openqa'
path = os.path.join(data_dir,'biencoder-squad1-train.json')
with open(path, "r", encoding="utf-8") as f:
	logger.info("Reading file %s" % path)
	data = json.load(f)
	logger.info("Aggregated data size: {}".format(len(data)))

with open(os.path.join(data_dir,'squad1-train.qa.csv'),'w') as file:
	writer = csv.writer(file,delimiter="\t")
	for item in data:
		row = (item['question'],str(item['answers']))
		writer.writerow(row)

logger.info('Done')