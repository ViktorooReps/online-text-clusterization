from pathlib import Path

from encoder import Encoder

if __name__ == '__main__':
    model_path = Path('main.pkl')
    model_name = 'cointegrated/rubert-tiny2'
    model = Encoder(model_name)
    model.save(model_path)