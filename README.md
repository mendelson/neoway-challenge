# Desafio Data Scientist - Mateus Mendelson

Este notebook foi desenvolvido como parte do teste técnico para a Neoway para a vaga de Cientista de Dados Sênior. Este material foi disponibilizado de forma pública sob autorização da empresa.

Para realizar a instalação dos pacotes necessários, basta executar o arquivo requirements.txt.

`$ pip install -r requirements.txt`

## Sistema de arquivos

    .
    └── data                                        # pasta para dados
          ├── conexoes.csv                          # possui todos os valores das conexões após predição
          ├── conexoes_espec.csv                    # arquivo original das conexões, ainda com valores faltantes
          ├── individuos_espec.csv                  # arquivo original com dados dos indivíduos, ainda com valores faltantes
          └── processed_dataset.pkl                 # arquivo com o dataset completo, sem nenhum valor faltante
    ├── best_model.pickle                           # arquivo com os parâmetros da rede neural treinada
    ├── Desafio notebook.zip                        # arquivo com o notebook (compactado)
    ├── Desafio notebook.html                       # arquivo com o notebook exportado para HTML
    ├── desafio-ds.zip                              # arquivo original com o dataset (deve ser descompactado para a pasta "data")
    ├── Product Data Science.pdf                    # arquivo com o enunciado do desafio
    ├── requirements.txt                            # arquivo com os pacotes a serem instalados via pip
    └── customLib                                   # pasta com classes próprias desenvolvidas para este desafio
          ├── DataImputer.py                        # módulo Python para realização de imputação de dados
          └── FeatureMaker.py                       # módulo Python para montagem das features para o model de ML