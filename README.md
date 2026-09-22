# extrator-texto-ocr

Script em Python que lê texto de uma imagem — foto, print, documento escaneado — usando o Tesseract. Tem uma etapa de pré-processamento em OpenCV pra melhorar imagem ruim antes de mandar pro OCR.

Instalei o Tesseract, testei em português de verdade (acento, cedilha, R$, número de nota) e cheguei a mudar o jeito que pré-processo a imagem porque a primeira versão que escrevi piorava o resultado em vez de ajudar — os detalhes estão lá embaixo.

## Instalando

Precisa do Tesseract instalado à parte (não é biblioteca Python, é um programa):

```bash
winget install --id UB-Mannheim.TesseractOCR
```

E do pacote de idioma português, que não vem por padrão — baixa o `por.traineddata` em https://github.com/tesseract-ocr/tessdata e joga na pasta `tessdata` da instalação do Tesseract (ou aponte a variável `TESSDATA_PREFIX` pra uma pasta sua, caso não tenha permissão de admin pra escrever em `Program Files`, que foi o meu caso).

Depois:

```bash
git clone https://github.com/Thi4goVcs/extrator-texto-ocr.git
cd extrator-texto-ocr
pip install -r requirements.txt
```

## Usando

```bash
python ocr.py exemplos/lista_compras.png
```

Outras opções:

```bash
python ocr.py imagem.png --idioma eng              # outro idioma
python ocr.py imagem.png --sem-preprocessamento     # OCR direto, sem tratar a imagem
python ocr.py imagem.png --comparar                 # mostra os dois resultados lado a lado
python ocr.py imagem.png --saida resultado.json      # salva em arquivo (.txt ou .json)
```

## Testando

```bash
pytest
```

Os testes geram a própria imagem na hora (com PIL), então o texto "esperado" é sempre exatamente o que foi desenhado — não depende de nenhum arquivo de imagem salvo no repositório.

## A parte que valeu a pena registrar

Fiz uma imagem de teste limpa e outra "suja" (metade escurecida, simulando sombra de foto de celular, mais ruído gaussiano por cima) pra comparar com e sem pré-processamento.

Minha primeira versão do pré-processamento fazia CLAHE (contraste) e threshold adaptativo direto em cima do ruído. Resultado: na imagem suja, o OCR foi de "quase certo com um pouco de lixo no final" pra **vazio**. Ou seja, meu "melhoramento" piorou.

O que funcionou: tirar o ruído primeiro (`fastNlMeansDenoising`), depois achatar a sombra dividindo a imagem por uma versão bem borrada dela mesma, e só então binarizar com Otsu (que escolhe o limiar sozinho, em vez de um valor fixo). Com isso a imagem suja passou a ler perfeito, igual à limpa.

A lição: pré-processamento de imagem não é "sempre ajuda" — dá pra piorar fácil se aplicar a técnica errada pro tipo de ruído que você tem. Por isso deixei o `--comparar` no script: em vez de confiar cegamente, dá pra ver o antes e depois em qualquer imagem real.

## Limites que conheço

- Sem GPU, então imagem grande demora. Não testei com arquivo de vários MB.
- OCR erra bem mais em letra manuscrita do que em texto digitado — não testei manuscrito aqui.
- O pré-processamento foi ajustado pro tipo de sombra/ruído que simulei. Foto com desfoque de movimento ou perspectiva torta (documento fotografado em ângulo) é caso que ainda não tratei.
