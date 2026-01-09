# 🍎🍌 Classificador de Frutas com rede neural convolucional

Este projeto implementa uma **Rede Neural Convolucional** capaz de classificar imagens de frutas, atualmente distinguindo entre **banana** e **maçã**.

O objetivo do projeto é aplicar, na prática, conceitos fundamentais de **Visão Computacional** e **Deep Learning**, com foco em aprendizado manual e compreensão do funcionamento interno de um modelo de convolução.

---

## 🧠 Visão Geral

A rede recebe uma imagem como entrada e retorna uma predição simples diretamente no terminal, por exemplo:

Isso é uma banana.  
Isso é uma maçã.

O projeto prioriza clareza, controle do processo e construção consciente do modelo, mesmo utilizando um dataset pequeno.

---

## 📂 Estrutura do Projeto

Classificador-de-Frutas/  
│  
├── dataset/  
│   ├── treino/  
│   │   ├── banana/  
│   │   └── maca/  
│   └── validacao/  
│       ├── banana/  
│       └── maca/  
│  
├── model.py  
├── train.py  
├── predict.py  
└── README.md  

---

## 🖼️ Dataset

- Dataset criado **manualmente**
- Aproximadamente:
  - 20 imagens por classe para **treinamento**
  - 15 imagens por classe para **validação**
- Total aproximado: **70 imagens**
- As imagens apresentam variações naturais de:
  - iluminação
  - tamanho
  - textura
  - pequenas imperfeições

Mesmo com um conjunto reduzido de dados, o modelo consegue aprender padrões relevantes entre as classes.

---

## 🧪 Modelo

O modelo é baseado em uma **Rede Neural Convolucional**, composta por:

- Camadas convolucionais com filtros 3x3
- Funções de ativação
- Camadas de pooling
- Camadas densas para classificação final

A arquitetura foi escolhida para equilibrar simplicidade e capacidade de aprendizado, considerando o tamanho do dataset.

---

## ⚙️ Funcionamento

1. As imagens são carregadas e pré-processadas
2. O modelo é treinado utilizando o conjunto de treino
3. A validação é feita com imagens nunca vistas pela rede
4. Uma imagem externa pode ser passada ao modelo para classificação
5. O resultado é exibido no terminal em forma de texto

---

## 🚀 Objetivo do Projeto

- Consolidar o entendimento de redes neurais convolucionais
- Trabalhar com visão computacional em um cenário real
- Criar um projeto prático e didático para portfólio
- Demonstrar domínio do pipeline completo:
  - dados → modelo → treino → validação → predição

---

## 🛠️ Tecnologias Utilizadas

- Python
- PyTorch
- PIL

---

## 📌 Observações

Este projeto foi desenvolvido com foco educacional e experimental.  
Melhorias futuras podem incluir:
- aumento do dataset
- mais classes de frutas
- visualização da imagem no momento da predição
- ajustes finos na arquitetura do modelo
