### Desafio 1

Considerando os dados do arquivo dados_Q1.csv, treine o modelo “Árvore de Classificação” (função de custo: entropia, sem podas) e utilize como metodologia de reamostragem a técnica validação cruzada sequencial (com 10 folds). 

a) Qual é a média da medida acurácia nas partições de treino? 

[1.00]

b) Qual é a média da medida acurácia nas partições de validação?

[0.76]

### Desafio 2

Considerando os dados do arquivo dados_Q2.csv, treine o modelo SVM (kernel: linear, fator de regularização: 0.1) e utilize como metodologia de reamostragem a técnica Leave-One-Out.

a) Qual é a média da medida MAE nas partições de treino?

[76.91]

b) Qual é a média da medida MAE nas partições de validação?

[78.39]

### Desafio 3

Considerando os dados do arquivo dados_Q3.csv, treine o modelo Regressão Logística (tipo de regularização: L1, fator de regularização: 0.1) utilizando a metodologia de validação holdout. Use para treino a primeira metade da base de dados, e as demais linhas para validação. Para fins de aproximação, considere que números menores que 0.001 são iguais a zero.

a) Qual é o valor da medida F1 Score para a partição de validação?

[0.61]

b) Quantas variáveis influenciam as predições geradas pelo modelo treinado?

['v_11', 'v_10', 'v_3', 'v_16', 'v_12']

### Desafio 4

Considerando os dados do arquivo dados_Q4.csv, aplique a técnica PCA (Principal Component Analysis)

a) Qual é a fração da variabilidade total explicada pelo segundo componente principal?

[33.73]

b) Qual é o número mínimo de componentes necessários para explicar 99% da variabilidade dos dados?

[8]

### Desafio 5

A) Considerando os dados do arquivo dados_Q5.csv, execute o algoritmo Complete-Linkage e corte o dendrograma na altura 2,3. Qual a posição do centroide de cada grupo formado? 

[[-0.01067531, 0.04420797, -0.04634375, 0.88997719, 0.06051477]
 [1.02705476, 0.02465894, -0.00805571, -0.0338859, -0.07312192]
 [-0.0343138, 1.11416901, 0.00533692, 0.03115574, 0.03287099]
 [-0.02287197, -0.01309619, 0.98219399, 0.07248944, 0.06414487]
 [0.04243589, 0.13170493, 0.12078695, 0.01712429, -0.02424086]]

B) Considerando os dados do arquivo dados_Q5.csv, execute o algoritmo Kmeans (com 10 iterações no máximo). Utilize como posição inicial dos centroides a posição calculada na questão anterior. Qual a posição final dos centroides?

[[6.63286061e-02, 3.61618954e-02, -3.61446512e-02, 1.03064205e+00, 3.19803265e-02]
 [1.01085001e+00, 1.41187280e-02, -1.52992669e-02, -2.02804403e-02, -5.56874047e-02]
 [-3.18851265e-02, 1.02291983e+00, 6.40807866e-02, 9.06345310e-03, 4.93184937e-02]
 [2.40435798e-03, 1.50286066e-02, 1.02150177e+00, 4.09546617e-02, 3.78730831e-02]
 [-7.54894132e-02, -3.42770553e-02, 1.41084792e-04, 1.07591849e-01, -2.13484714e-03]]


Bloco de Perguntas 1

Assinale as alternativas com V ou F para Verdadeiro ou Falso respectivamente.

Atente para o fato que uma questão errada anula uma certa.
Em caso de dúvidas deixe em branco.

3 - É possível calcular eficientemente as distâncias entre as instâncias mais próximas utilizando KD-Trees, que funciona melhor quando a dimensão das instâncias é pequena.

[V]

4 - Quando falamos de similaridade cosseno em variáveis independentes positivas (como TF-IDF, por exemplo) sabemos que deve estar no intervalo [0, 1], em que 0 significa a maior distância.

[V]

5 - O objetivo do algoritmo k-means é minimizar a soma dos quadrados das distâncias dos pontos aos centroides.

[F]

6 - Diferente do k-means com distância euclidiana, o GMM (Gaussian Mixture Models) pode encontrar grupos com formato elipsoidal.

[V]

7 - As técnicas PCA e T-SNE são amplamente utilizadas porque conseguem reduzir o número de dimensões de uma base de dados, sem que haja perda de informações.

[V]

8 - Silhueta é um índice que mensura a qualidade do agrupamento a qual favorece uma menor distância intracluster e uma grande distância extracluster.

[V]

9 - O dendrograma permite visualizar os grupos formados em diferentes granularidades, sem ser necessário executar novamente o algoritmo.

[V]

10 - Há dois tipos principais de algoritmos para Agrupamento hierárquico: Divisivos (ou top-down) e Aglomerativos (ou botton-up).

[V]

11 - O método SGD (Stochastic Gradient Descendent) é utilizado para treinar diversos tipos de modelos, tais como redes neurais, regressão logística e SVM.

[V]

12 - No paradigma Map/Reduce, remover caracteres especiais de cada documento de um corpus (coleção de documentos) é um exemplo de Reduce.

[F]


Bloco de Perguntas 2

Assinale as alternativas com V ou F para Verdadeiro ou Falso respectivamente.
Atente para o fato que uma questão errada anula uma certa.
Em caso de dúvidas deixe em branco.


3 -  A regressão logística utiliza o método de Mínimos Quadrados Ordinários na estimação dos parâmetros.

[F]

4 - Na regressão logística, a variável dependente é contínua no intervalo entre [0,1].

[V]

5 - Na regressão logística as variáveis independentes podem ser numéricas ou categóricas. O coeficiente associado a cada variável representa a variação do valor estimado quando esta aumenta em 1 unidade.

[V]

6 - Em uma árvore de decisão pode-se avaliar a taxa de crescimento do erro de classificação para paralisar a divisão dos ramos.

[V]

7 - No classificador SVM, pode-se alterar a função objetivo para permitir que alguns pontos caiam dentro da margem durante o treinamento.

[V]

8 - O precision é uma medida de qualidade calculada pela fração de quantas predições positivas estão corretas, podendo ser utilizada tanto para algoritmos de classificação quanto de regressão.

[F]

9 - O KNN, com distância euclidiana, é sensível à escala dos atributos usado para treiná-lo.

[V]

10 - No processo de criação de ensembles, quanto mais correlacionadas forem as predições individuais de cada classificador, o ganho de desempenho do modelo final combinado tende a ser menor.

[F]

11 - O algoritmo Naive Bayes assume dependência entre cada par de característica.

[F]

12 - Combinar diversas árvores de decisão usando a técnica Bagging é o mesmo que treinar uma Random Forest.

[V]

13 - Os métodos de ensemble com boosting combinam vários classificadores para produzir um classificador mais robusto.

[V]

14 - Uma das formas de tratar problemas de classes desbalanceadas é incluir a informação a priori do número de amostras na função de custo a ser otimizada.

[Pula]

15 - Não é aconselhado a utilização da acurácia para problemas com classes desbalanceadas.

[V]


Bloco de Perguntas 3

2 - Para problemas de regressão com duas variáveis independentes altamente correlacionadas, enquanto o Lasso tem maior chance de escolher aleatoriamente uma delas, o método Elastic Net pode selecionar ambas.

[Pula]

3 - L1 e L2 são ambos componentes de regularização cujo objetivo é tratar o overfitting exclusivamente para problemas de regressão.

[F]

4 - Em um problema de regressão utilizando KNN, quanto maior for o número de vizinhos (K), menor a probabilidade de overfitting.

[V]

5 - O bootstrap pode ser usado para estimar os erros dos coeficientes de uma regressão Linear.

[V]

6 - Grid Search é uma técnica estocástica para encontrar os melhores hiperparâmetros para um modelo.

[V]

Estatistica

a) Sob o contexto de inferência estatística e teste de hipóteses, calcule um intervalo de confiança para a média do vetor [4, 13, 9, 9, 8, 8, 11, 10, 12, 7, 20, 11, 7, 16, 13] utilizando 1.000 reamostras de bootstrap. Considere 𝑧(1−𝛼 2 ⁄ ) = 1,96. 

[8.53333333, 12.53333333]

b) Uma gerente deve decidir sobre a concessão de empréstimos aos seus clientes. Para tal, ela deve analisar diversas informações para definir a chance do cliente ficar inadimplente. Com base em dados passados, ela estima em 15% a taxa de inadimplência. Dentre os inadimplentes, ela tem 90% de chance de tomar a decisão certa, enquanto essa chance aumenta para 95% entre os clientes adimplentes. Essa gerente acaba de recusar um empréstimo, qual é a probabilidade da decisão tomada estar correta? 

15% taxa de inadimplência
85% taxa de adimplência

Inadimplentes 90% de chance decisão correta
Adimplentes 95% de chance

0,15 * 0,9 + 0,85 * 0,95 = 0,9425 = 94,25%

2 - Considerando três eventos quaisquer A, B e C, com probabilidade de ocorrência maior que zero, pode-se afirmar que se os eventos forem independentes então P (A ∩ B ∩ C) = P(A ∩ B) + P(C).

A afirmação não é correta,

Tendo A, B e C como independentes:

P(A ∩ B) = P(A)P(B), P(B ∩ C) = P(B)P(C), P(A ∩ C) = P(A)P(C) 
P(A ∩ B ∩ C) = P(A)P(B)P(C)

Então, P(A ∩ B ∩ C) = P(A)P(B)P(C) = P(A ∩ B)P(C).

3 -  Se dois eventos são disjuntos, então P(A ∩ B) = P(A) x P(B).

Afirmação não é correta, para que P(A ∩ B) seja igual P(A) x P(B) os elementos A e B precisam ser independentes.



