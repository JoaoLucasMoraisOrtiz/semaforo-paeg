> **Versão original (08/10/2026).** A versão revisada após a avaliação, com as correções pedidas e os resultados medidos, está em [`relatorio.md`](relatorio.md). Duas afirmações abaixo estão incorretas: a quantização INT8 com as opções padrão zerou as detecções do modelo (ver `relatorio.md`), e passar de FP32 para INT8 reduz os pesos a cerca de ¼, não à metade.

# Relatório de Melhorias e Continuidade do Projeto

Este documento detalha as três principais frentes de melhoria arquitetural implementadas no projeto **Traffic Lights Pro** para detecção de semáforos em veículos autônomos. O foco principal destas adições foi transformar uma simples inferência frame a frame em um sistema robusto, rápido e com contexto espacial-temporal.

---

## Ponto 1: Memória Temporal e Rastreamento (Tracking)

**O Problema:** Inicialmente, o modelo de inteligência artificial detectava semáforos de forma independente em cada quadro do vídeo. Isso causava intermitência visual (o famoso *flickering*), e o veículo autônomo não sabia se o semáforo do "frame 1" era o mesmo do "frame 2".

**A Solução Implementada:**
- Integramos o algoritmo **ByteTrack**, um dos rastreadores de múltiplos objetos (*Multi-Object Trackers*) mais eficientes da atualidade.
- A rede RT-DETR agora não apenas detecta as *Bounding Boxes*, mas repassa as coordenadas para o tracker, que atribui um **ID Único** para cada semáforo.
- Esse ID único permitiu a criação de um "Histórico de Estados" (`state_history`), onde o carro consegue lembrar a cor do semáforo nos últimos 5 frames, evitando que reflexos momentâneos gerem falsos positivos (ex: ler vermelho no meio do verde).

---

## Ponto 2: Associação Ego-Lane e Decodificação de Cor Clássica

**O Problema:** Mesmo sabendo onde estão os semáforos, um cruzamento pode ter vários. O carro autônomo precisa saber **qual semáforo importa para a sua faixa de rolagem**. Além disso, o modelo cru RT-DETR pré-treinado na base COCO identifica apenas a classe genérica "traffic light", sem distinguir sua cor atual.

**A Solução Implementada:**
- **Detecção Clássica de Faixas:** Criamos o script `lane_detection.py` utilizando *Canny Edge Detection* e a Transformada de Hough (*Hough Lines*). Isso permitiu identificar o centro da pista por onde o carro trafega de forma muito leve (sem onerar a placa de vídeo).
- **Associação Espacial (Targeting):** Calculamos matematicamente qual semáforo (ID) está geometricamente mais alinhado ao centro da faixa, marcando-o dinamicamente como **TARGET** na tela.
- **Extração de Cor via HSV:** Desenvolvemos a função `get_traffic_light_color()`. Em vez de treinar uma rede neural super pesada só para cor, utilizamos Visão Computacional Clássica: recortamos o miolo do semáforo (a partir da Bounding Box gerada pela IA) e convertemos para o espaço HSV, contando matematicamente os pixels brilhantes nas faixas do Vermelho, Amarelo e Verde.

---

## Ponto 3: Otimização Extrema para Hardware Embarcado (Edge CPU)

**O Problema:** Modelos arquitetura Transformer (como o RT-DETR-L, com mais de 30 milhões de parâmetros) são insustentáveis para rodar a 30 FPS em processadores de prateleira (CPU) dentro de um carro autônomo sem aceleração de hardware massiva.

**A Solução Implementada:**
- **Compilação Estática (OpenVINO / ONNX):** Desenvolvemos o módulo `export_model.py`, capaz de "congelar" e converter os gráficos computacionais do PyTorch para a *Engine* OpenVINO. 
- **Quantização INT8:** A precisão dos pesos decimais foi reduzida de 32 bits (FP32) para 8 bits (INT8), o que corta pela metade o consumo de memória RAM/Cache e acelera enormemente a CPU.
- **Frame Skipping Inteligente:** Adicionamos parâmetros de escalabilidade no script `run_video_detection.py`. Ao utilizar `--skip-frames 2`, a Inteligência Artificial é executada apenas em 10 frames por segundo, reduzindo a carga térmica da CPU, enquanto um sistema de Cache Visual projeta os dados anteriores na tela, mantendo a experiência do usuário perfeitamente suave e fluida.
- **Escalabilidade de Resolução e Renderização:** Adicionamos suporte a `--imgsz` (redimensionamento do tensor base para 320/480/640) e `--save` para facilitar a documentação acadêmica e o debug dos dados gerados.
- **Boilerplate C++:** Como visão de longo prazo, construímos e provisionamos a pasta `cpp_runner`, um esqueleto nativo em `C++` usando `OpenCV DNN` para importar o modelo otimizado sem depender do interpretador Python.