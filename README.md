> ## 📌 Autoria e divisão do trabalho
>
> Este repositório tem duas partes, separadas também no histórico do git:
>
> | Parte | Autor | Commits |
> |---|---|---|
> | **Base do projeto** (pré-existente) | Igor Gonçalves ([@Bolodarie](https://github.com/Bolodarie/traffic-lights-pro)) | os 9 primeiros commits, de 04/12/2025 |
> | **Melhorias e continuidade** (trabalho avaliado) | João Lucas Morais Ortiz | commits de 2026 em diante |
>
> Detalhes na seção [Contribuições de João Lucas Morais Ortiz](#-contribuições-de-joão-lucas-morais-ortiz) no fim deste arquivo.

# 🚦 Traffic Light Detection com RT-DETR (Transformer)

Este projeto implementa um sistema de detecção e classificação de semáforos em tempo real utilizando **Deep Learning** e **Computer Vision**. 

Diferente de abordagens clássicas de classificação (que apenas dizem a cor de uma imagem recortada), este projeto utiliza um **Vision Transformer (RT-DETR)** para localizar múltiplos semáforos na cena completa e identificar seus estados (Verde, Vermelho, Amarelo, Off) simultaneamente.

## 🎯 Objetivo
O projeto foi desenvolvido como parte de um estudo prático sobre **Transformers Visuais**, migrando de arquiteturas CNN tradicionais (como ResNet) para modelos de detecção de objetos de última geração (SOTA).

**Principais Funcionalidades:**
- 🧹 **Pipeline de Engenharia de Dados:** Conversão automática do dataset LISA (CSV/Anotações complexas) para o formato padrão YOLO/Detection.
- 🧠 **Treinamento com RT-DETR:** Utilização da biblioteca Ultralytics para Fine-Tuning do modelo `rtdetr-l` (Large).
- 🎥 **Inferência em Vídeo:** Script para processamento de vídeos reais com visualização de bounding boxes em tempo real.

## 🛠️ Stack Tecnológico
* **Linguagem:** Python 3.10
* **Core:** [Ultralytics](https://github.com/ultralytics/ultralytics) (RT-DETR)
* **Visão Computacional:** OpenCV
* **Manipulação de Dados:** Pandas, Scikit-Learn
* **Dataset:** LISA Traffic Light Dataset

## 📂 Estrutura do Projeto

```text
├── convert_lisa_to_detection.py  # Script de ETL: Converte anotações do LISA para .txt (YOLO format)
├── train_detection.py            # Script de configuração e treinamento do modelo
├── run_video_detection.py        # Script de inferência (aplica o modelo em vídeos)
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação
```

🚀 Como Rodar o Projeto
1. Instalação
Clone o repositório e instale as dependências:

Bash

pip install -r requirements.txt
2. Preparação dos Dados
Este projeto utiliza o LISA Traffic Light Dataset.

Baixe o dataset e extraia para a pasta data/archive.

Execute o script de conversão para organizar as pastas e gerar os labels:

Bash

python convert_lisa_to_detection.py
Isso criará a pasta data/detection_dataset pronta para o treino.

3. Treinamento
Para iniciar o treinamento do RT-DETR (Transfer Learning):

Bash

python train_detection.py
Nota: O script está configurado para detectar GPU automaticamente. Se não houver, rodará em CPU (mais lento).

4. Teste / Inferência
Para testar o modelo treinado em um vídeo MP4:

Bash

python run_video_detection.py --video seu_video_teste.mp4
📊 Resultados e Métricas
O modelo é treinado para detectar 3 classes principais:

0: Green (Verde)

1: Red (Vermelho)

2: Yellow (Amarelo)

---

## 🧩 Contribuições de João Lucas Morais Ortiz

### O que já existia (base de Igor Gonçalves)
- `convert_lisa_to_detection.py`: ETL do dataset LISA para o formato YOLO/Detection.
- `train_detection.py`: fine-tuning do RT-DETR-L com Ultralytics.
- `run_video_detection.py` (versão original, ~36 linhas): inferência frame a frame com `model.predict` e `results[0].plot()`, com caminho de vídeo fixo no código.
- `requirements.txt` e a primeira parte deste README.

### O que foi adicionado neste trabalho

| Arquivo | Situação | O que faz |
|---|---|---|
| `lane_detection.py` | **novo** | Detecção da faixa do próprio veículo (ego-lane) com Canny + Transformada de Hough. |
| `run_video_detection.py` | **reescrito** | Tracking com ByteTrack (ID por semáforo), histórico de estado dos últimos 5 frames contra *flickering*, cor do semáforo via HSV, escolha do semáforo **TARGET** alinhado à faixa, frame skipping com cache das detecções, `--imgsz`, `--save` (MP4 com fallback AVI, salva mesmo com CTRL+C), `--display-scale`, `--video`, e escolha automática do modelo mais rápido disponível (OpenVINO INT8 → OpenVINO → ONNX → `.pt`). |
| `export_model.py` | **novo** | Exporta o modelo para ONNX ou OpenVINO, com quantização INT8 e resolução de entrada configurável. |
| `cpp_runner/` (`main.cpp`, `CMakeLists.txt`) | **novo** | Esqueleto de inferência em C++ com OpenCV DNN para hardware embarcado. |
| `DOCUMENTACAO_MELHORIAS.md` | **novo** | Relatório das três frentes de melhoria: tracking temporal, associação ego-lane + cor, e otimização para CPU. |
| `.gitignore` | **novo** | Exclui pesos, vídeos e datasets do versionamento. |

Para ver exatamente o que mudou em relação à base: `git diff a271f84 HEAD`.

### Como rodar a versão melhorada
```bash
pip install -r requirements.txt
python export_model.py --format openvino --int8 --imgsz 320   # opcional, acelera na CPU
python run_video_detection.py --video seu_video.mp4 --imgsz 320 --skip-frames 2 --save
```
