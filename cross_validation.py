import numpy as np 


input_file = 'complex_10fold.txt'

scores = open(input_file,'r',encoding='utf-8').read().split('\n')
np_scores = np.array(scores).astype('float')
print("Accuracy: %0.2f (+/- %0.2f)" % (np_scores.mean(), np_scores.std() * 2))