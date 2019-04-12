'''

PROBLEMA 1

Dado um vetor de inteiros positivos, verificar se existe algum valor presente
em mais da metade (maioria absoluta) dos elementos do vetor.
    * Caso exista, retorna-lo.
    * Caso nao exista, retornar -1.

Exemplos:
    1. [] -> -1
    2. [13] -> 13
    3. [4, 7] -> -1
    4. [5, 1, 5] -> 5
    5. [2, 3, 1] -> -1
    6. [2, 1, 7, 2, 2, 8, 2] -> 2
    7. [2, 1, 7, 2, 2, 8, 2, 1] -> -1

'''


def problema1(vector):

    # Count the number os occurences of each number
    counts = dict((number, vector.count(number)) for number in set(vector))

    # Loop over the dictionary and return the number if greater than the half
    # of the length of the vector
    for number, count in counts.items():
        if count > len(vector)/2:
            # Return the number if the counts is greater than half of vector
            # length
            return number

    # Return -1 if didn't find any match
    return -1


'''

PROBLEMA 2

Dada uma string representando um numero em algarismos romanos, converte-la
para um inteiro convencional.

Considere somente numeros cuja saida correspondente seja menor ou igual a
3999.

Observacao: a representacao da entrada sera sempre em letras maiusculas.

Seguem os valores de cada algarismo romano:
    * I = 1
    * V = 5
    * X = 10
    * L = 50
    * C = 100
    * D = 500
    * M = 1000

Exemplos:
    1. "I" -> 1
    2. "III" -> 3
    3. "XVII" -> 17
    4. "CMXCII" -> 992
    5. "MMMCIV" -> 3104

'''


def problema2(romano):

    # Create a dictionary of the Roman numbers and correspond decimal number
    dictionary = {'M': 1000,
                  'D': 500,
                  'C': 100,
                  'L': 50,
                  'X': 10,
                  'V': 5,
                  'I': 1}

    # Initializa the variable
    number = 0

    # Loop over the string and make a vertical lookup of each Roman number and
    # substitute by the decimal number and sum the result
    for i in range(len(romano)):
        value = dictionary[romano[i]]
        if i+1 < len(romano) and dictionary[romano[i+1]] > value:
            number -= value
        else:
            number += value

    # Return the decimal number
    return(number)


'''
PROBLEMA 3

O dígito verificador é um mecanismo de autenticação utilizado para verificar a
validade e a autenticidade de um valor numérico, evitando dessa forma fraudes
ou erros de transmissão ou digitação. Consiste em um ou mais dígitos
acrescentados ao valor original e calculados a partir deste através de um
determinado algoritmo. Números de documentos de identificação, de matrícula,
cartões de crédito e quaisquer outros códigos numéricos, que necessitem de
maior segurança, utilizam dígitos verificadores (Wikipedia).

Uma das rotinas mais tradicionais para cálculo do dígito verificador é
denominada Módulo 11, que funciona da seguinte forma: cada dígito do número,
começando da direita  para a esquerda (menos significativo para o mais
significativo) é multiplicado, na ordem, por 2, depois 3, depois 4 e assim
sucessivamente, até o limite de multiplicação escolhido. Então novamente
multiplica-se o número por 2, 3, etc. O resultado de cada uma dessas
multiplicações é somado e depois multiplicado por 10 e por fim dividido pelo
módulo escolhido, o digito será o resto dessa divisão inteira.

Exemplo para numero 2615338 (digito verificador = 8) com limite de
multiplicação 5 em módulo 11:

2    6    1    5    3    3    -    8
x3   x2   x5   x4   x3   x2
6    12   5    20   9    6

6 + 12 + 5 + 20 + 9 + 6 = 58
58 x 10 = 580
580 / 11 = 52, resto 8 => DV = 8

Escreva um programa que receba um número inteiro, juntamente com um digito
verificador. Calcule o dígito verificador do número usando a técnica descrita
acima, considerando que o limite de multiplicação (após a multiplicação pelo
limite, a multiplicação retorna a 2) e o módulo são passados como parametro.
O algoritmo deve retornar 1, se o número é válido ou 0 caso não seja.

'''


def problema3(numero, limMultiplicacao, modulo):

    # Extract the Digito Verificado and Numero
    number = [int(d) for d in str(numero)][:-1]
    dv = [int(d) for d in str(numero)][-1]

    # Itinialize the multiplication list and initial value mul list
    mul_list = []
    init = 1

    # Loop over the lenght of the number and create multiplication list
    for _ in range(len(number)):
        if init == limMultiplicacao:
            init = 1
        init += 1
        mul_list.append(init)

    # Reverse the order of the list
    mul_list.reverse()

    # Multiply the list lists
    result = [a*b for a, b in zip(number, mul_list)]

    # Calculate the module 11
    result = (sum(result) * 10) % 11

    # return the 1 if the digite is valid and 0 if invalid
    if result == dv:
        return 1
    else:
        return 0


def main():
    # Teste básico do problema 1
    vetorP1 = [2, 1, 7, 2, 2, 8, 2]
    print("Saida P1 =", problema1(vetorP1))

    # Teste básico do problema 2
    stringP2 = "MCMLXXXV"
    print("Saida P2 =", problema2(stringP2))

    # Teste básico do problema 3
    numero = 2615338
    limMult = 5
    modulo = 11
    print("Saida P3 = ", end='')
    if problema3(numero, limMult, modulo):
        print("Digito Valido")
    else:
        print("Digito invalido")
