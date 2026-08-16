# TRINITY

Sistema de análise automatizada de molhabilidade, ângulo de contato e dinâmica de gotas para estudos de superfícies e interfaces líquido-sólido.

## Visão geral

O projeto implementa uma interface gráfica desktop para processamento de imagens e vídeos com foco em:

- medição de ângulo de contato estático
- análise de histerese de contato
- ajuste por curvas e métodos robustos
- análise de impacto de queda de gota
- visualização de resultados, overlays e gráficos

A aplicação é construída em Python com Tkinter/CustomTkinter e integra processamento de imagem com OpenCV, NumPy, SciPy, Pandas e Matplotlib.

## Arquitetura atual

O projeto já está parcialmente refatorado em módulos, com o arquivo principal funcionando como ponto de entrada e orquestrador, e as funcionalidades distribuídas em pacotes especializados.

### Estrutura principal

```text
Lucas/
├── trinity_gotas_modelo.py        # ponto de entrada da aplicação
├── trinity/
│   ├── analysis/                  # processamento científico e detecção
│   │   ├── drop_impact.py
│   │   ├── edge_detection.py
│   │   ├── hysteresis.py
│   │   ├── info_overlay.py
│   │   ├── plot_service.py
│   │   ├── static_angle.py
│   │   └── __init__.py
│   ├── ui/                       # interface, menus, controles e selectors
│   │   ├── analysis_menu.py
│   │   ├── app_controls.py
│   │   ├── app_shell.py
│   │   ├── dashboard.py
│   │   ├── drop_impact_parameters.py
│   │   ├── drop_impact_results.py
│   │   ├── drop_impact_selector.py
│   │   ├── drop_impact_workflow.py
│   │   ├── manual_analysis.py
│   │   ├── static_angle_selector.py
│   │   ├── tooltip.py
│   │   ├── video_player.py
│   │   ├── view_controls.py
│   │   ├── visual_tools.py
│   │   └── __init__.py
│   ├── utils/                    # geometria, desenho e cálculo de ângulos
│   │   ├── contact_angles.py
│   │   ├── drawing.py
│   │   ├── geometry.py
│   │   └── __init__.py
│   └── __init__.py
├── vids/                         # vídeos de exemplo / teste
├── .venv/                        # ambiente virtual do projeto
├── README.md
└── .gitignore
```

## Módulos e responsabilidades

### [trinity_gotas_modelo.py](trinity_gotas_modelo.py)
Arquivo principal da aplicação. Responsável pela criação da interface e pela orquestração das operações da análise.

### [trinity/analysis](trinity/analysis)
Contém as rotinas científicas e de processamento:

- detecção de bordas e segmentação
- cálculo de ângulos estáticos e dinâmicos
- ajuste polinomial e WLS
- análise de histerese
- processamento de impacto de gotas
- geração de overlays e gráficos

### [trinity/ui](trinity/ui)
Responsável pela camada visual do sistema:

- menus e opções de análise
- controles de reprodução de vídeo
- zoom e navegação por frame
- seletores manuais de pontos e linhas
- dashboard e painel de resultados
- workflow de análise interativa

### [trinity/utils](trinity/utils)
Funções auxiliares de geometria e desenho:

- cálculo de diâmetro e ângulos
- separação e seleção de pontos
- desenho de tangentes, arcos e curvas de ajuste
- conversão e pós-processamento geométrico

## Principais funcionalidades

### 1. Análise de ângulo de contato estático
- detecção automática de contorno
- ajuste de perfil da gota
- cálculo de ângulo de contato por diferentes métodos
- suporte a abordagem manual e avançada

### 2. Histerese
- análise de avanço e recuo
- métodos baseados em ajuste polinomial e WLS
- processamento de sequência de frames
- acompanhamento do comportamento dinâmico da linha de contato

### 3. Impacto de gota
- segmentação da gota em queda
- análise de velocidade e impacto
- extração de parâmetros temporais e geométricos
- visualização de resultados dimensionais

### 4. Visualização e exploração de dados
- reprodução de vídeo
- navegação por frames
- zoom interativo
- apresentação de gráficos e tabelas
- exportação de dados da análise

## Requisitos

- Python 3.10+
- OpenCV
- NumPy
- SciPy
- Pandas
- Matplotlib
- Tkinter
- CustomTkinter

## Como executar

No diretório do projeto:

```powershell
# ativar ambiente virtual
.\.venv\Scripts\Activate.ps1

# executar a aplicação
.\.venv\Scripts\python.exe trinity_gotas_modelo.py
```

## Fluxo de uso típico

1. abrir um vídeo ou sequência de imagens
2. configurar o tipo de análise desejada
3. selecionar parâmetros de detecção ou pontos de referência
4. executar a rotina de processamento
5. validar o resultado na interface e/ou exportar os dados

## Estado do projeto

O projeto encontra-se em processo de modularização. A lógica foi separada em módulos especializados sob o pacote [trinity](trinity), reduzindo o acoplamento do arquivo principal e organizando melhor as responsabilidades por domínio funcional.

## Licença

A licença do projeto pode ser ajustada conforme a política do laboratório ou instituição responsável. Se não houver uma definição formal, esse campo deve ser preenchido antes da publicação ou distribuição.

## Observação

Este README foi atualizado para refletir a estrutura atual do código e a distribuição funcional real do projeto, em vez de documentação de uma etapa anterior da refatoração.
