# Otimização de Memória via Batch Embedding

**Squad 02 — SB100 | Nicolas Alves Witzel da Silva | 29 de março de 2026**

---

## Problema identificado

Durante a execução do pipeline v1 de ingestão de PDFs no Google Colab, o processo de geração de embeddings densos era realizado de forma sequencial — um chunk por vez, dentro do loop principal de upsert. Cada chamada individual ao modelo `Qwen3-Embedding-0.6B` carregava o tensor de entrada, executava a inferência e retornava o vetor, mas **não liberava o grafo computacional do PyTorch entre chamadas**.

O resultado era acúmulo progressivo de tensores alocados na RAM ao longo do processamento de um PDF com muitos chunks, frequentemente causando:

- crash da sessão Colab por esgotamento de memória (OOM)
- lentidão crescente conforme o número de chunks aumentava

O problema era mais severo em PDFs grandes — artigos com muitas páginas e tabelas podiam gerar 60 a 80 chunks — onde o acúmulo de memória não liberada tornava o processamento instável a partir da metade do documento.

---

## Solução implementada

A solução foi separar completamente a etapa de embedding denso do loop de upsert, introduzindo uma função dedicada de encoding em batch:

```python
def gerar_vetores_densos_batch(textos: list, batch_size: int = 32) -> list:
    embeddings = dense_model.encode(
        textos,
        batch_size=batch_size,
        show_progress_bar=True
    )
    return [emb.tolist() for emb in embeddings]
```

No loop principal, todos os textos de um PDF são coletados primeiro e vetorizados de uma só vez **antes** do início do upsert:

```python
# Vetorizar todos os chunks do PDF em batch único
all_dense = gerar_vetores_densos_batch(
    [c["texto"] for c in chunks],
    batch_size=32
)

# Só então iterar para upsert (sem mais chamadas ao modelo denso)
for idx, chunk in enumerate(chunks):
    dense_vec = all_dense[idx]   # acesso direto à lista já computada
    ...
```

Complementarmente, ao final de cada PDF são liberadas explicitamente as variáveis de maior peso em memória:

```python
del pdf_bytes, blocos, docling_doc, resultado
gc.collect()
```

---

## Por que o batch reduz o uso de memória

O `SentenceTransformer.encode()` com `batch_size=32` processa os textos em grupos de 32, reutilizando o mesmo buffer de alocação de tensor a cada grupo e descartando-o ao final de cada sub-batch. Quando a chamada encerra, o PyTorch libera o grafo computacional de uma vez.

Isso contrasta com 60 a 80 chamadas individuais, cada uma alocando e nunca descartando seu próprio grafo durante o loop. Em termos práticos:

| Abordagem | Memória alocada pelo modelo |
|---|---|
| Sequencial (v1) | Cresce proporcionalmente ao número de chunks |
| Batch (v2) | Constante — determinada apenas pelo `batch_size` |

---

## Impacto observado

Após a implementação do batch, o uso de RAM durante a etapa de embedding denso caiu visivelmente. PDFs com 60 a 80 chunks, que anteriormente causavam instabilidade ou crash, passaram a ser processados de forma estável até o final.

A sessão Colab não encerrou por OOM em nenhum dos **mais de 150 PDFs** processados após a mudança.

A separação entre embedding e upsert também trouxe benefício secundário de organização: o loop de upsert passou a ser responsável apenas por sparse embedding, montagem de payload e envio ao Qdrant — tornando o fluxo mais legível e cada etapa mais fácil de depurar isoladamente.

---

## Limitação remanescente

O vetor esparso (`gerar_vetor_esparso`) ainda é gerado sequencialmente, chunk por chunk, dentro do loop de upsert. O modelo SPLADE utilizado (`opensearch-neural-sparse-encoding-doc-v2-distill`) não foi incluído no batch porque seu padrão de saída — índices e valores de ativação esparsos — é mais complexo de consolidar em batch sem perda de rastreabilidade por chunk.

Isso representa uma oportunidade de otimização adicional em versões futuras do pipeline.
