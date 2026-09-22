"""Extrai texto de imagens com Tesseract OCR, com pré-processamento opcional em OpenCV."""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# No Windows o instalador do Tesseract nem sempre entra no PATH. Se não achar
# o binário, tenta o caminho padrão do instalador antes de desistir.
_CAMINHO_PADRAO_WINDOWS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if shutil.which("tesseract") is None and os.path.isfile(_CAMINHO_PADRAO_WINDOWS):
    pytesseract.pytesseract.tesseract_cmd = _CAMINHO_PADRAO_WINDOWS


def preprocessar(imagem: np.ndarray) -> np.ndarray:
    """Remove ruído, corrige sombra/luz desigual e binariza.

    A primeira versão disso usava CLAHE + threshold adaptativo, e piorava o
    resultado em imagem com ruído (zerava o texto lido). O que funcionou de
    verdade foi denoise, depois dividir a imagem por uma versão bem borrada
    dela mesma (achata a sombra) e só então binarizar com Otsu.
    """
    cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    sem_ruido = cv2.fastNlMeansDenoising(cinza, h=15)
    fundo = cv2.GaussianBlur(sem_ruido, (0, 0), sigmaX=25)
    normalizada = cv2.divide(sem_ruido, fundo, scale=255)
    _, binarizada = cv2.threshold(normalizada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binarizada


def extrair_texto(imagem: np.ndarray, idioma: str = "por") -> str:
    return pytesseract.image_to_string(imagem, lang=idioma).strip()


def processar_arquivo(caminho: Path, idioma: str = "por", usar_preprocessamento: bool = True) -> str:
    imagem = cv2.imread(str(caminho))
    if imagem is None:
        raise FileNotFoundError(f"não consegui abrir a imagem: {caminho}")
    if usar_preprocessamento:
        imagem = preprocessar(imagem)
    return extrair_texto(imagem, idioma)


def salvar_resultado(caminho_saida: Path, caminho_imagem: Path, texto: str) -> None:
    if caminho_saida.suffix == ".json":
        conteudo = json.dumps({"arquivo": str(caminho_imagem), "texto": texto}, ensure_ascii=False, indent=2)
    else:
        conteudo = texto
    caminho_saida.write_text(conteudo, encoding="utf-8")


def main() -> None:
    # no Windows, quando a saída é redirecionada (pipe, arquivo), o Python às
    # vezes escolhe a codepage do sistema em vez de UTF-8 e come acento
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Extrai texto de uma imagem com OCR.")
    parser.add_argument("imagem", type=Path, help="Caminho da imagem (foto, print, documento escaneado)")
    parser.add_argument("--idioma", default="por", help="Idioma do Tesseract (padrão: por)")
    parser.add_argument(
        "--sem-preprocessamento", action="store_true", help="Roda o OCR direto na imagem, sem melhorar o contraste antes"
    )
    parser.add_argument(
        "--comparar", action="store_true", help="Mostra o resultado com e sem pré-processamento, lado a lado"
    )
    parser.add_argument("--saida", type=Path, help="Salva o texto num arquivo (.txt ou .json) em vez de mostrar na tela")
    args = parser.parse_args()

    if not args.imagem.is_file():
        raise SystemExit(f"arquivo não encontrado: {args.imagem}")

    if args.comparar:
        sem = processar_arquivo(args.imagem, args.idioma, usar_preprocessamento=False)
        com = processar_arquivo(args.imagem, args.idioma, usar_preprocessamento=True)
        print("=== sem pré-processamento ===")
        print(sem or "(vazio)")
        print("\n=== com pré-processamento ===")
        print(com or "(vazio)")
        return

    texto = processar_arquivo(args.imagem, args.idioma, usar_preprocessamento=not args.sem_preprocessamento)

    if args.saida is None:
        print(texto)
        return

    salvar_resultado(args.saida, args.imagem, texto)
    print(f"salvo em {args.saida}")


if __name__ == "__main__":
    main()
