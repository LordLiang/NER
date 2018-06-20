"""
Tagging example.
"""
import argparse
import os
from pprint import pprint

from bilstm_crf.tagger import Tagger
from bilstm_crf.models import BiLSTMCRF
from bilstm_crf.preprocessing import IndexTransformer
from bilstm_crf.utils import load_data


def main(args):

    print('Loading data...')
    x_test = load_data(args.test_data)
    print(len(x_test), 'test sequences')

    print('Loading model...')
    model = BiLSTMCRF.load(args.weights_file, args.params_file)
    p = IndexTransformer.load(args.preprocessor_file)
    tagger = Tagger(model, preprocessor=p)

    print('Tagging...')
    tagger.analyze_all(x_test, args.save_prediction)
    #res = tagger.analyze(args.sent)
    #pprint(res)



if __name__ == '__main__':
    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/medical_NER')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')

    parser = argparse.ArgumentParser(description='Tagging.')
    parser.add_argument('--test_data', default=os.path.join(DATA_DIR, 'test.eval'), help='testing data')
    parser.add_argument('--save_prediction', default=os.path.join(SAVE_DIR, 'test_prediction.eval'))
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'model_weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocessor.json'))
    args = parser.parse_args()
    main(args)