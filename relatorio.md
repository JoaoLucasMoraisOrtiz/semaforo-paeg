# Relatório de Melhorias e Continuidade do Projeto

Este documento detalha as três frentes de melhoria implementadas no projeto **Traffic Lights Pro** para detecção de semáforos em veículos autônomos, as correções feitas após a primeira avaliação e o que foi efetivamente medido. O objetivo das adições foi transformar uma inferência frame a frame em um sistema com contexto espacial e temporal, e avaliar o quanto ele se aproxima de rodar em CPU embarcada.

> **Revisão de 13/10/2026.** Esta versão incorpora os ajustes pedidos na avaliação: TARGET restrito a semáforos e com casos indeterminados; ausência de faixa reconhecida pelo programa; histórico de cor sem votos repetidos nem cor "presa"; registro de versões, pesos e comandos; e separação entre o que foi implementado e o que foi validado (seção 4). A versão anterior continua em `DOCUMENTACAO_MELHORIAS.md`.

---

## 1. Memória temporal e rastreamento (tracking)

**O problema.** O modelo detectava semáforos de forma independente em cada quadro. Isso causava intermitência visual (*flickering*), e o sistema não sabia se o semáforo do quadro 1 era o mesmo do quadro 2.

**O que foi implementado.**
- **ByteTrack** (via Ultralytics) recebe as caixas do RT-DETR e atribui um **ID** a cada semáforo.
- Cada ID tem um **histórico das últimas 5 leituras de cor**, e o estado exibido é o voto majoritário dessas leituras.

**Correções desta revisão.**
- **Leituras desconhecidas também votam.** Antes, um "Unknown" era descartado, e uma cor antiga podia ficar presa no histórico indefinidamente. Agora, se as leituras novas forem desconhecidas, o estado passa a "Unknown". Em caso de empate, vence a leitura mais recente.
- **Só leituras novas entram no histórico.** Com `--skip-frames`, os quadros pulados reaproveitam a detecção anterior. Antes eles votavam de novo com a mesma leitura, e a janela de "5 quadros" representava na prática 2 ou 3 inferências. Agora o histórico recebe uma leitura por inferência.
- IDs que não aparecem por 30 inferências têm o histórico descartado.

## 2. Associação ego-lane e decodificação de cor clássica

**O problema.** Um cruzamento pode ter vários semáforos, e o veículo precisa saber qual vale para a sua faixa. Além disso, o RT-DETR pré-treinado no COCO identifica só a classe genérica "traffic light", sem a cor.

**O que foi implementado.**
- **Detecção clássica de faixa** (`lane_detection.py`): Canny + Transformada de Hough, separando bordas esquerda e direita pela inclinação. O centro da faixa é a média das duas bordas numa altura de referência.
- **Associação espacial (TARGET):** o semáforo cujo centro horizontal está mais próximo do centro da faixa é marcado como alvo.
- **Cor via HSV** (`get_traffic_light_color()`): recorta a caixa do semáforo, converte para HSV e conta os pixels nas faixas de vermelho, amarelo e verde. Isso não seria necessário se utilizássemos o modelo já treinado pela equipe do PAEG, mas eu não tinha os checkpoints, nem hardware para treinar o modelo a partir do modelo base.

**Correções desta revisão.**
- **TARGET restrito a semáforos.** A versão anterior considerava todas as classes do COCO, e nos vídeos o próprio motorista (classe *person*) aparecia como alvo. Agora o detector já é chamado só com as classes de semáforo (`traffic light` no COCO; `Green/Red/Yellow` no modelo treinado no LISA). Carros e pessoas não são rastreados nem desenhados.
- **Ausência de faixa reconhecida.** `detect_ego_lane()` retornava o centro da imagem quando não encontrava linhas. Agora retorna `None` e um status: `ok` (duas bordas), `partial` (uma borda só, sem como saber a largura da faixa), `none` ou `invalid` (bordas geometricamente inconsistentes). O status aparece na tela.
- **Casos em que não dá para escolher o alvo.** O programa mostra "Alvo: indeterminado" com o motivo em quatro situações:
  - faixa não detectada;
  - nenhum semáforo na cena;
  - nenhum semáforo a menos de 15% da largura da imagem do centro da faixa (`--target-max-dist`);
  - ambiguidade: os dois candidatos mais próximos estão a menos de 5% da largura um do outro (`--target-margin`).
- `--lane-roi-bottom` permite excluir o capô/painel da busca de faixa (0,72 no vídeo de demonstração).

## 3. Otimização para hardware embarcado (CPU)

**O problema.** Modelos Transformer como o RT-DETR-L (cerca de 32 milhões de parâmetros) são pesados para 30 FPS em CPU.

**O que foi implementado.**
- **Exportação ONNX/OpenVINO** (`export_model.py`), com opção de quantização INT8 e resolução configurável. Nesta revisão foi adicionado `--data`, para calibrar o INT8 com o próprio dataset em vez do coco8 padrão.
- **Frame skipping:** com `--skip-frames 1`, a inferência roda em 1 de cada 2 quadros, e os quadros intermediários reaproveitam a última detecção.
- **`--imgsz`** para reduzir a resolução de entrada.
- **Medição de desempenho:** o script agora mede o tempo de cada quadro e de cada inferência e grava um registro JSON da execução (seção 7).
- **`cpp_runner/`:** esqueleto em C++ com OpenCV DNN. Ele carrega um modelo ONNX e executa o *forward*, mas **ainda não extrai as caixas** nem tem tracking, faixa ou cor.

**Resultado sobre o INT8.** A quantização INT8 com as opções padrão do Ultralytics/NNCF (calibração no coco8) **destruiu o modelo**: no vídeo de teste, os modelos INT8 detectaram 0 semáforos, contra 360 do modelo FP32. Num quadro em que o FP32 faz 10 detecções (conf 0,3), o INT8 a 640 não devolveu nenhuma caixa, mesmo com confiança mínima de 0,05; o de 480 devolveu uma, com confiança 0,07. Por isso o INT8 saiu da escolha automática de modelo. Ainda não foram testadas calibração com imagens do domínio (LISA ou dashcam) e quantização mista que preserve o decoder do Transformer.

---

## 4. Implementado × validado

| Item | Situação | Evidência |
|---|---|---|
| Detecção de semáforos (RT-DETR-L COCO) | **Validado parcialmente** | 360 detecções em 118 de 357 inferências no vídeo de teste. Sem anotações, não há precisão/recall medidos. Inspeção visual: semáforos vermelhos detectados; verdes (que aparecem azulados à noite) muitas vezes não; uma placa azul detectada como semáforo. |
| Tracking ByteTrack (IDs) | **Implementado, não medido** | IDs são atribuídos, mas não houve medição de trocas de ID; nos vídeos há trocas visíveis (o mesmo semáforo recebe IDs diferentes). |
| Histórico de cor (correções desta revisão) | **Implementado, verificado no código** | Uma leitura por inferência; "Unknown" vota. Não há medição do efeito sobre o *flickering*. |
| Cor via HSV | **Implementado, não medido** | Sem rótulos de cor por quadro. Postes de luz alaranjados e reflexos na chuva são fontes conhecidas de erro. |
| Detecção de faixa | **Validado como limitada** | No vídeo noturno com chuva: duas bordas em 15% das inferências (53/357), uma borda em 38%, nenhuma em 47%. Quando há duas bordas, a correção do centro não foi verificada contra referência. |
| TARGET só com semáforos e com casos indeterminados | **Validado no vídeo de teste** | Nenhuma pessoa ou carro foi marcado como alvo. Alvo decidido em 7 de 357 inferências; nas demais o programa informa o motivo: faixa não detectada 304, nenhum semáforo 44, ambíguo 1, nenhum alinhado 1. |
| Exportação OpenVINO FP32 | **Validado** | Mesmas 360 detecções do PyTorch, com inferência ~29% mais rápida (253 vs. 355 ms). |
| Exportação OpenVINO INT8 | **Não funciona como está** | 0 detecções (ver seção 3). |
| Tempo real em CPU | **Não atingido** | Melhor configuração útil: 7,7 FPS de processamento contra 30 FPS do vídeo (seção 5). |
| Runner C++ | **Esqueleto inicial** | Carrega o modelo e executa o *forward*; sem pós-processamento. Não compilado nem testado nesta revisão. |

**Sobre a fluidez dos vídeos.** Os vídeos salvos com `--save` são gravados a 30 FPS, independentemente do tempo de processamento. Por isso um vídeo fluido **não comprova tempo real**. A medida correta é o tempo de processamento por quadro, que agora aparece na tela ("Proc: X FPS") e no registro JSON.

---

## 5. Resultados medidos

Vídeo: dashcam noturna com chuva, 720×1280, 30 FPS, 714 quadros (23,8 s). Todas as execuções em CPU (AMD Ryzen 7 5700G, 16 threads), com `--skip-frames 1` (357 inferências) e `--lane-roi-bottom 0.72`, sem janela de exibição.

O tempo por quadro inclui faixa, inferência + tracking, cor e desenho. Não inclui leitura e gravação do vídeo. Tempo real exigiria no máximo **33,3 ms por quadro**.

| Modelo | imgsz | Inferência média (p95) | Tempo médio por quadro (p95) | FPS de processamento | Semáforos detectados |
|---|---|---|---|---|---|
| PyTorch `rtdetr-l.pt` | 640 | 355 ms (395) | 181 ms (386) | 5,5 | 360 |
| OpenVINO FP32 | 640 | 253 ms (296) | 130 ms (288) | 7,7 | 360 |
| OpenVINO INT8 | 640 | 156 ms (200) | 81 ms (184) | 12,3 | **0** |
| OpenVINO INT8 | 480 | 94 ms (101) | 50 ms (101) | 20,1 | **0** |

O INT8 seria o mais rápido, mas não detecta nada. Entre as configurações que funcionam, o melhor resultado é o **OpenVINO FP32 a 640**, que processa cerca de **26% da velocidade de tempo real**. O tempo por quadro varia muito, porque metade dos quadros não roda inferência; o p95 reflete os quadros com inferência.

A GPU integrada da máquina de testes não é suportada pelo ROCm, então não houve medição em GPU.

Observação anterior, qualitativa e não medida nesta revisão: o modelo roda satisfatoriamente com 320, mas tem desempenho claramente superior com 480.

---

## 6. Limitações que permanecem

- **Detector genérico.** O RT-DETR COCO não foi ajustado para semáforos brasileiros nem para imagens noturnas. Ele perde muitos semáforos verdes e gera falsos positivos (placas azuis). O modelo treinado no LISA (`train_detection.py`) não foi treinado neste trabalho, por falta de hardware.
- **Faixa por Canny + Hough** falha com chuva, reflexos e pouca iluminação. Na maior parte do vídeo de teste, o alvo fica indeterminado por falta de faixa.
- **Associação faixa–semáforo por distância horizontal** é uma heurística: compara o centro da faixa, medido perto do veículo, com a posição do semáforo no alto da imagem. Ela não considera perspectiva, curvas nem semáforos laterais.
- **Sem anotações de referência**, não há precisão, recall, acurácia de cor nem trocas de ID medidas. Os números da seção 5 medem desempenho computacional e contagens, não acerto.
- **Sem tempo real em CPU**, e o INT8 precisa ser refeito antes de ser útil.
- **Runner C++** é só um esqueleto.

---

## 7. Reprodutibilidade

**Ambiente usado nas medições** (também registrado em cada JSON de `resultados/`):
- Python 3.13.12, ultralytics 8.4.75, torch 2.11.0+rocm7.2, torchvision 0.26.0+rocm7.2, opencv-python 4.13.0.92, numpy 2.4.3, openvino 2026.2.1, nncf 3.2.0, onnx 1.22.0, onnxruntime 1.27.0, lap 0.5.13
- Linux 7.0.0-38, AMD Ryzen 7 5700G, inferência em CPU
- Versões fixadas em `requirements-testado.txt`

**Pesos e vídeo (SHA-256):**

| Arquivo | SHA-256 |
|---|---|
| `rtdetr-l.pt` ([Ultralytics assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/rtdetr-l.pt)) | `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f` |
| `rtdetr-l_openvino_model_640/` | `16c8b0c4ea85035ea867eb7a701c1f2481d1e70970f0c57b2b4f8498429c0b1b` |
| `rtdetr-l_int8_openvino_model_640/` | `6e75fea4667a0992c831d266b34bd7a62194c861e3b11711a9afac95b213a819` |
| `rtdetr-l_int8_openvino_model_480/` | `62cb37e692e6d2f3f935c7bff84bc959152847ba0023c115f4c865cbc961c2e8` |
| `videCarr.mp4` (vídeo de teste) | `3e07f4c05e28ee896b3a4b353bbfaa3c76af365474d8b9be83240d1ffd5aa3f1` |

O hash de uma pasta exportada é calculado sobre todos os arquivos dela, em ordem alfabética (`file_sha256()` em `run_video_detection.py`). O INT8 foi calibrado com o coco8, padrão do Ultralytics.

**Comandos:**

```bash
# Exportação. O export sempre grava em rtdetr-l_[int8_]openvino_model; as pastas foram
# renomeadas com o sufixo da resolução para manter as três versões lado a lado.
python export_model.py --format openvino --imgsz 640          && mv rtdetr-l_openvino_model rtdetr-l_openvino_model_640
python export_model.py --format openvino --int8 --imgsz 640   && mv rtdetr-l_int8_openvino_model rtdetr-l_int8_openvino_model_640
python export_model.py --format openvino --int8 --imgsz 480   && mv rtdetr-l_int8_openvino_model rtdetr-l_int8_openvino_model_480

# Demonstrações e medições (uma por linha da tabela da seção 5)
python run_video_detection.py --video videCarr.mp4 --model rtdetr-l.pt --no-display --skip-frames 1 --imgsz 640 --lane-roi-bottom 0.72 --save --output demos/demo_pt_cpu_640.mp4 --log resultados/demo_pt_cpu_640.json
python run_video_detection.py --video videCarr.mp4 --model rtdetr-l_openvino_model_640 --no-display --skip-frames 1 --imgsz 640 --lane-roi-bottom 0.72 --save --output demos/demo_openvino_fp32_640.mp4 --log resultados/demo_openvino_fp32_640.json
python run_video_detection.py --video videCarr.mp4 --model rtdetr-l_int8_openvino_model_640 --no-display --skip-frames 1 --imgsz 640 --lane-roi-bottom 0.72 --save --output demos/demo_openvino_int8_640.mp4 --log resultados/demo_openvino_int8_640.json
python run_video_detection.py --video videCarr.mp4 --model rtdetr-l_int8_openvino_model_480 --no-display --skip-frames 1 --imgsz 480 --lane-roi-bottom 0.72 --save --output demos/demo_openvino_int8_480.mp4 --log resultados/demo_openvino_int8_480.json
```

A demonstração curta (`demos/demo_curta_openvino_fp32_640.mp4`) é o vídeo da configuração OpenVINO FP32 recomprimido em H.264 (`ffmpeg -c:v libx264 -crf 26`). A tela mostra o status da faixa, o alvo ou o motivo de ele estar indeterminado, o FPS de processamento e se o quadro teve inferência ou reaproveitou a anterior.

Os vídeos de demonstração da primeira entrega não tiveram comando, pesos nem versões registrados, e foram substituídos por estes.

---

## Conclusão

Implementamos sobre a base do projeto um pipeline que rastreia semáforos entre quadros, estima a cor sem treinar uma rede extra e tenta associar o semáforo à faixa do veículo, reconhecendo quando não consegue fazer essa associação.

As medições mostram os limites da abordagem atual. Em CPU, o RT-DETR-L chega a 7,7 FPS na melhor configuração funcional, longe dos 30 FPS do vídeo. A quantização INT8 padrão, que seria o caminho para ganhar velocidade, inutilizou o modelo. E a detecção clássica de faixa raramente funciona à noite com chuva.

Esses resultados reforçam o ponto de partida do trabalho: para dispositivos embarcados, é preciso reduzir bastante o custo de inferência, seja com modelos menores, seja com uma quantização calibrada no domínio, sem perder detecções. Como continuidade, os próximos passos são:
- refazer o INT8 calibrando com imagens do domínio;
- treinar ou obter um detector de semáforos com classes de cor;
- substituir a faixa por Hough por um método mais robusto;
- anotar alguns trechos de vídeo para medir acerto, além de velocidade.
