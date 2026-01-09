# 👁️ Tech Challenge: Análise de Vídeo com Visão Computacional

## 📋 Sobre o Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge** da Pós-Graduação em Inteligência Artificial. O objetivo é criar uma aplicação de visão computacional capaz de analisar vídeos para extrair informações relevantes sobre o comportamento humano.

A aplicação processa um arquivo de vídeo frame a frame e realiza simultaneamente:
1.  **Reconhecimento Facial:** Identificação da presença humana.
2.  **Análise Sentimental:** Classificação de emoções (feliz, triste, neutro, surpresa, etc.) em tempo real.
3.  **Detecção de Atividades/Anomalias:** Monitoramento de movimentos bruscos baseados na diferença de pixels entre frames.
4.  **Geração de Relatório:** Exportação automática de um resumo estatístico ao final da execução.

---

## 🚀 Funcionalidades

* **Detecção Facial:** Utiliza algoritmos de *Deep Learning* para localizar rostos na cena.
* **Classificação de Emoções:** Exibe a emoção predominante sobre a face detectada.
* **Monitoramento de Movimento:**
    * 🟢 **Status Normal:** Movimentação padrão.
    * 🔴 **Alerta de Anomalia:** Movimentos bruscos ou mudanças repentinas de cena.
* **Relatório Automático:** Gera um arquivo `.txt` contendo o total de frames analisados, contagem de anomalias e estatísticas das emoções.

---

## 🛠️ Tecnologias Utilizadas

* **[Python](https://www.python.org/):** Linguagem principal.
* **[OpenCV](https://opencv.org/):** Manipulação de vídeo e processamento de imagem (cálculo de diferenças, desenho de retângulos).
* **[DeepFace](https://github.com/serengil/deepface):** Framework de reconhecimento facial e análise de atributos faciais.
* **NumPy:** Cálculos matemáticos de matrizes para análise de pixels.

---

## 📦 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o **Python** instalado em sua máquina.

### 1. Instalação das Dependências

Abra o terminal na pasta do projeto e execute:

```bash
pip install opencv-python deepface tf-keras numpy