# Anotações do Projeto CNN - Classificador de Imagens

Este arquivo serve como um diário de bordo e referência para entender as diferentes partes do projeto.


## 1. Saída do Terminal (Ao executar `main.py`)

Quando o script é executado, o terminal exibe várias mensagens. Nenhuma delas é um erro; são informações e confirmações.

### Mensagens Informativas do TensorFlow
- **Exemplos:**
  - `... oneDNN custom operations are on ...`
  - `... This TensorFlow binary is optimized to use available CPU instructions ...`
- **O que significa:** O TensorFlow está simplesmente confirmando que está ativo e usando otimizações de hardware (CPU) para rodar de forma mais eficiente.
- **Ação Necessária:** Nenhuma. **Isto é normal e positivo.**

### Confirmação do Carregamento dos Dados
- **Exemplo:**
  - `Found 1730 images belonging to 11 classes.`
  - `Found 1960 images belonging to 11 classes.`
- **O que significa:** Esta é a confirmação mais importante. Mostra que o Keras conseguiu encontrar e carregar com sucesso as imagens das pastas `data/train` e `data/test`.
- **Ação Necessária:** Apenas verificar se os números de imagens e classes correspondem ao esperado.

### Dicionário de Classes
- **Exemplo:** `Classes encontradas: {'0': 0, '1': 1, '10': 2, '2': 3, ...}`
- **O que significa:** É o resultado do nosso comando `print()`. Mostra o mapeamento que o Keras criou entre o **nome da pasta** (a classe, que é um texto) e o **índice numérico** que o modelo realmente usará para os cálculos.
- **Ação Necessária:** Nenhuma, apenas observar. Lembre-se que a ordem é **alfabética** (`'10'` vem antes de `'2'`), e não numérica.

### Aviso de Usuário (`UserWarning`)
- **Exemplo:** `... UserWarning: Do not pass an input_shape ...`
- **O que significa:** É apenas um **aviso**, não um erro. A biblioteca Keras está sugerindo uma maneira mais moderna de escrever a primeira camada do modelo. A forma como escrevemos (`input_shape=...`) é totalmente válida, funcional e muito comum.
- **Ação Necessária:** Nenhuma. **Pode ser ignorado com segurança.**

## 2. Lembretes Importantes do Projeto

- **Estrutura de Pastas:** O `ImageDataGenerator` exige que as imagens estejam organizadas em `data/train/NOME_DA_CLASSE` e `data/test/NOME_DA_CLASSE`.

- **Ordenação das Classes:** Para evitar confusão com a ordenação alfabética (ex: "10" vindo antes de "2"), a melhor prática é renomear as pastas de classe com zeros à esquerda (ex: `00, 01, 02, ..., 09, 10`).

- **Data Augmentation:** As técnicas de aumento de dados (rotação, zoom, etc.) são aplicadas **apenas** no conjunto de treinamento para ajudar o modelo a generalizar. O conjunto de validação/teste deve permanecer com as imagens originais.

## 3. Definição do Modelo:

- **1. Compilação do Modelo:** A linha `modelo.compile(...)` configura o otimizador `('adam')`, a função de perda `('categorical_crossentropy')` e as métricas `('accuracy')` que o modelo usará durante o treinamento.

