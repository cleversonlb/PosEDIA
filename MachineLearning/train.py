# CARREGANDO AS BIBLIOTECAS QUE SERAO UTILIZADAS

from __future__ import print_function
import argparse
import os
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


# UTILIZANDO A BIBLIOTECA ARGPARSE PARA OBTER OS PARAMETROS PASSADOS QUANDO CHAMAR TRAIN.PY

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
   
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

   
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # CARREGANDO O DATASET ATRAVES DO PATH PASSADO PELO --train (O DATASET ESTA NO AWS S3)
    file = os.path.join(args.train, "train_data.csv")
    data = pd.read_csv(file, engine="python")

    # SEPARANDO AS VARIAVEIS PREDITORAS DA VARIAVEL TARGET
    X = data.drop(['target', 'id'], axis =1)
    y = data['target']
    
    # SEPARANDO O DATASET EM DADOS DE TREINO E DE TESTE
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
   
    # CRIANDO O MODELO DE REGRESSAO E FAZENDO O FIT DO MODELO COM AS VARIAVEIS DE TESTE
    model = LogisticRegression()
    model.fit(X, y)

    # SALVANDO O MODELO TREINADO
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    
    

# CRIANDO METODO PARA CARREGAR O MODELO TREINADO                       
def model_fn(model_dir):
    
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    
    return model
