import os
import json
from jiwer import wer
import re
from tqdm import tqdm
import sys

def compute_normalized_wer(results_data_path, output_wer_path, data_name):

    norm_ground_truth = []
    norm_hypo = []

    utts = json.load(open(results_data_path, 'r', encoding='utf8'))['utts']
    #outcomes = dict()
    for key, values in tqdm(utts.items()):
        #key_utts = []
        for index ,i in enumerate(values['output']):
            hyp_text = i['rec_text'].replace('▁',' ').replace('<eos>','').replace('<UNK>','').replace('unk','').replace('<','').replace('>','')
            ref_text = i['text'].replace('ـ',' ')
            #hyp_score =  i['score']
            ref_text = ref_text.strip().replace('i','j').replace('I','g').replace('E','G').replace('B','G').replace('C','G').replace('*','')
            hyp_text = hyp_text.strip().replace('i','j').replace('I','g').replace('E','G').replace('B','G').replace('C','G').replace('*','')
                
            #Remove Multiple Spaces
            ref_text = re.sub(' +',' ',ref_text).strip()
            hyp_text = re.sub(' +',' ',hyp_text).strip()

            norm_ground_truth.append(ref_text)
            norm_hypo.append(hyp_text)

    error = wer(norm_ground_truth, norm_hypo)
    print(error)
    with open(output_wer_path, 'w') as out:
        out.write(data_name + ' normalized WER: ' + str(error) + '\n')



def main():
    results_data_path = sys.argv[1]
    output_wer_path = sys.argv[2]
    data_name = sys.argv[3]
    compute_normalized_wer(results_data_path, output_wer_path, data_name)

if __name__ == '__main__':
    main()