# Find Photos Pipeline

Este projeto implementa um pipeline completo para **processamento, indexação e recuperação de fotos com base em reconhecimento facial**, usando **FaceNet (facenet-pytorch)**.

Funcionalidades principais:

1. **Baixar fotos em lote** a partir de um arquivo JSON (`images_source.json`)
2. **Gerar embeddings faciais** para todas as faces detectadas nas imagens
3. **Encontrar fotos de uma pessoa específica** usando imagens de referência
4. **Baixar automaticamente apenas as fotos selecionadas**

O pipeline foi projetado para:

- Grandes volumes de imagens
- Múltiplas faces por imagem
- Uso em CPU (com paralelismo) ou GPU
- Reutilização dos embeddings sem recomputar

## Requisitos

- Python **3.10+** (testado em 3.12)
- GPU opcional (CUDA acelera bastante, mas CPU funciona)

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

Downloads já existentes são ignorados e o processo pode ser retomado.

## 2. Criar embeddings faciais

Este passo:

- Detecta todas as faces em cada imagem
- Gera um embedding por face
- Salva tudo em um arquivo JSON incremental

### Executar geração de embeddings (CPU em paralelo)

```bash
python generate_embeddings_parallel.py \
  --input_dir imagens \
  --output embeddings.json \
  --start_name CCF252_07937.jpg \
  --max_faces 15 \
  --workers 6
```

### Argumentos

| Argumento | Descrição |
|-----------|-----------|
| `--input_dir` | Pasta com as imagens baixadas |
| `--output` | Arquivo JSON de saída |
| `--start_name` | Nome da imagem inicial (ordem alfabética) |
| `--max_faces` | Número máximo de faces por imagem |
| `--workers` | Número de processos paralelos (recomendado: 4-8) |

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

## 3. Encontrar e baixar fotos de uma pessoa

Este passo permite:

- Usar uma ou mais imagens de referência
- Encontrar todas as fotos onde a pessoa aparece
- Baixar somente as fotos correspondentes

Nenhum embedding é recalculado.

### Estrutura esperada

```
reference/
  ├─ ref1.jpg
  ├─ ref2.jpg

embeddings.json
images_source.json
```

### Executar busca e download das fotos correspondentes

```bash
python find_images.py \
  --reference_dir reference \
  --embeddings embeddings.json \
  --images_source images_source.json \
  --output_dir matched \
  --threshold 0.8 \
  --type original \
  --workers 8
```

### O que o script faz

1. Gera um embedding médio a partir das imagens em `reference/`
2. Compara esse embedding com todas as faces em `embeddings.json`
3. Seleciona as imagens cuja similaridade ≥ threshold
4. Baixa automaticamente apenas essas imagens

### Argumentos principais

| Argumento | Descrição |
|-----------|-----------|
| `--reference_dir` | Pasta com imagens da pessoa de referência |
| `--embeddings` | Arquivo com embeddings pré-gerados |
| `--images_source` | Fonte das URLs |
| `--output_dir` | Pasta de saída das imagens encontradas |
| `--threshold` | Similaridade mínima (ex: 0.75–0.85) |
| `--type` | `original` ou `thumb` |
| `--workers` | Downloads paralelos |

## Fluxo completo recomendado

```
images_source.json
        ↓
download_images.py
        ↓
pasta imagens/
        ↓
generate_embeddings_parallel.py
        ↓
embeddings.json
        ↓
find_images.py
        ↓
matched/
```
