# Detecção de Phishing com Machine Learning

Projeto desenvolvido como Trabalho de Conclusão de Curso (TCC) do curso de **Segurança da Informação** da **Universidade Federal do Ceará (UFC)**.

Autor: **Tiago Andrade**
Orientador: **Prof. Israel Eduardo**
Ano: **2025**

---

## Objetivo do Projeto

Detecção automática de e-mails de phishing, utilizando técnicas de **Processamento de Linguagem Natural (NLP)** e **Aprendizado de Máquina**, com comparação entre diferentes algoritmos de classificação.

---

## Técnicas Utilizadas

* Processamento de Linguagem Natural (NLP)
* TF-IDF
* Validação Cruzada Estratificada
* Análise Exploratória de Dados
* Métricas de desempenho (Acurácia, Precisão, Recall e F1-Score)
* Análise de desempenho computacional (tempo e memória)

---

## Modelos Avaliados

* Multinomial Naive Bayes (MultinomialNB)
* Support Vector Machine (LinearSVC)
* Random Forest Classifier

---

## Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-learn
* SpaCy
* Matplotlib
* Seaborn
* TQDM
* PSUtil
* SciPy

---

## Como Executar o Projeto

### Clonar o repositório

```bash
git clone https://github.com/TiagoACTR/tcc-deteccao-phishing-machine-learning.git
cd tcc-deteccao-phishing-machine-learning
```

---

### Instalar as dependências

```bash
pip install -r requirements.txt
```

---

### Baixar o modelo de linguagem do SpaCy

```bash
python -m spacy download en_core_web_sm
```

---

### Executar o projeto

```bash
python main.py
```

---

## Resultados

Os modelos foram avaliados por meio de validação cruzada estratificada, sendo comparados pelas métricas:

* Acurácia
* Precisão
* Recall
* F1-Score

Também foram analisados:

* Tempo de treinamento
* Tempo de predição
* Consumo de memória

Os resultados demonstram a viabilidade do uso de Machine Learning para detecção automática de e-mails de phishing.

---

## Dataset



---

## Licença

Este projeto está licenciado sob a licença **MIT**.
