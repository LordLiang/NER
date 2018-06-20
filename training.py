import argparse
import os

import bilstm_crf
from bilstm_crf.utils import load_data_and_labels, load_glove, filter_embeddings
from bilstm_crf.models import BiLSTMCRF
from bilstm_crf.preprocessing import IndexTransformer
from bilstm_crf.trainer import Trainer

def main(args):

    print('Loading data...')
    x_train, y_train = load_data_and_labels(args.train_data)
    x_valid, y_valid = load_data_and_labels(args.valid_data)
    print(len(x_train), 'train sequences')
    print(len(x_valid), 'valid sequences')
    embeddings = load_glove(EMBEDDING_PATH)

    print('Transforming datasets...')
    p = IndexTransformer()
    p.fit(x_train, y_train)
    embeddings = filter_embeddings(embeddings, p._word_vocab.vocab, args.word_emb_size)

    print('Building a model.')
    model = BiLSTMCRF(num_labels=p.label_size,
                      word_vocab_size=p.word_vocab_size,
                      char_vocab_size=p.char_vocab_size,
                      word_embedding_dim=args.word_emb_size,
                      char_embedding_dim=args.char_emb_size,
                      word_lstm_size=args.word_lstm_units,
                      char_lstm_size=args.char_lstm_units,
                      dropout=args.dropout,
                      embeddings=embeddings,
                      use_char=True,
                      use_crf=True
                      )

    model.build()
    model.compile(loss=model.get_loss(), optimizer=args.optimizer)

    print('Training the model...')
    trainer = Trainer(model, preprocessor=p)
    trainer.train(x_train, y_train, x_valid, y_valid, epochs=args.max_epoch)

    print('Saving the model...')
    model.save(args.weights_file, args.params_file)
    p.save(args.preprocessor_file)


if __name__ == '__main__':

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data/medical_NER')
    SAVE_DIR = os.path.join(os.path.dirname(__file__), 'models')
    EMBEDDING_PATH = os.path.join(os.path.dirname(__file__), 'data/glove.6B/glove.6B.100d.txt')

    parser = argparse.ArgumentParser(description='Training a model')
    parser.add_argument('--train_data', default=os.path.join(DATA_DIR, 'train.eval'), help='training data')
    parser.add_argument('--valid_data', default=os.path.join(DATA_DIR, 'dev.eval'), help='validation data')
    parser.add_argument('--weights_file', default=os.path.join(SAVE_DIR, 'model_weights.h5'))
    parser.add_argument('--params_file', default=os.path.join(SAVE_DIR, 'params.json'))
    parser.add_argument('--preprocessor_file', default=os.path.join(SAVE_DIR, 'preprocessor.json'))

    # Training parameters
    parser.add_argument('--loss', default='categorical_crossentropy', help='loss')
    parser.add_argument('--optimizer', default='adam', help='optimizer')
    parser.add_argument('--max_epoch', type=int, default=50, help='max epoch')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--checkpoint_path', default=None, help='checkpoint path')
    parser.add_argument('--log_dir', default=None, help='log directory')
    parser.add_argument('--early_stopping', action='store_true', help='early stopping')

    # Model parameters
    parser.add_argument('--char_emb_size', type=int, default=25, help='character embedding size')
    parser.add_argument('--word_emb_size', type=int, default=100, help='word embedding size')
    parser.add_argument('--char_lstm_units', type=int, default=25, help='num of character lstm units')
    parser.add_argument('--word_lstm_units', type=int, default=100, help='num of word lstm units')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')

    args = parser.parse_args()
    main(args)
