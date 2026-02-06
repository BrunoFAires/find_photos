# Find Photos Pipeline

Este projeto implementa um pipeline completo para:

1. **Baixar fotos em lote** a partir de um arquivo JSON (`images_source.json`)
2. **Gerar embeddings faciais** para todas as faces detectadas nas imagens usando **FaceNet (facenet-pytorch)**

O pipeline foi projetado para:

- Grandes volumes de imagens
- Execução incremental (retomar de onde parou)
- Múltiplas faces por imagem
- Uso por várias pessoas/máquinas

## Requisitos

- Python **3.10+** (testado em 3.12)
- GPU opcional

Instale as dependências:

```bash
pip install -r requirements.txt
```

## 1. Baixar fotos

### Arquivo obrigatório: `images_source.json`

Este arquivo não é gerado automaticamente. Ele deve ser fornecido previamente e conter as URLs das imagens.

### Formato esperado do `images_source.json`

```json
[
  {
    "filename": "CCF252_07937",
    "original": "https://example.com/images/CCF252_07937.jpg",
    "thumb": "https://example.com/thumbs/CCF252_07937.jpg"
  },
  {
    "filename": "CCF252_07938",
    "original": "https://example.com/images/CCF252_07938.jpg",
    "thumb": "https://example.com/thumbs/CCF252_07938.jpg"
  }
]
```

### Campos obrigatórios

| Campo | Descrição |
|-------|-----------|
| `filename` | Nome base do arquivo (sem extensão) |
| `original` | URL da imagem em alta resolução |
| `thumb` | URL da miniatura (opcional) |

O script não assume que `original` e `thumb` existam ao mesmo tempo. Você deve escolher qual usar via argumento.

### Executar download das imagens

```bash
python download_images.py \
  --input images_source.json \
  --output_dir imagens \
  --start_name CCF252_07937 \
  --type original \
  --workers 8
```

### Argumentos

| Argumento | Descrição |
|-----------|-----------|
| `--input` | Caminho para o `images_source.json` |
| `--output_dir` | Pasta onde as imagens serão salvas |
| `--start_name` | Nome do arquivo a partir do qual o download começa |
| `--type` | `original` ou `thumb` |
| `--workers` | Quantidade de downloads paralelos |

### Comportamento importante

- Imagens já existentes são ignoradas
- O download pode ser interrompido e retomado
- Ordem baseada no `filename`

## 2. Criar embeddings faciais

Este passo:

- Detecta todas as faces em cada imagem
- Gera um embedding por face
- Salva tudo em um arquivo JSON incremental

### Executar geração de embeddings

```bash
python generate_embeddings.py \
  --input_dir imagens \
  --output embeddings.json \
  --start_name CCF252_07937 \
  --max_faces 15
```

### Argumentos

| Argumento | Descrição |
|-----------|-----------|
| `--input_dir` | Pasta com as imagens baixadas |
| `--output` | Arquivo JSON de saída |
| `--start_name` | Nome da imagem inicial (ordem alfabética) |
| `--max_faces` | Número máximo de faces por imagem |

### Formato do `embeddings.json`

```json
[
  {
    "image": "CCF252_07937.jpg",
    "face_index": 0,
    "embedding": [0.0123, -0.9981, ...]
  },
  {
    "image": "CCF252_07937.jpg",
    "face_index": 1,
    "embedding": [0.2211, 0.0042, ...]
  }
]
```

- Um registro por face
- `face_index` indica a ordem da face detectada
- Cada embedding possui 512 dimensões

O arquivo é salvo incrementalmente a cada imagem processada, evitando corrupção em caso de interrupção.

## Fluxo recomendado

```
images_source.json
        ↓
download_images.py
        ↓
pasta imagens/
        ↓
generate_embeddings.py
        ↓
embeddings.json
```
