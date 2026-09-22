"""Testes do extrator de OCR. Rodar com: pytest

As imagens de teste são geradas na hora (PIL), não ficam salvas no repositório -
isso garante que o texto "esperado" é sempre exatamente o que foi desenhado.
"""
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr import extrair_texto, preprocessar, salvar_resultado

FONTE = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)


def imagem_com_texto(texto: str, largura: int = 700, altura: int = 90) -> np.ndarray:
    img = Image.new("RGB", (largura, altura), "white")
    ImageDraw.Draw(img).text((15, 15), texto, fill="black", font=FONTE)
    return np.array(img)[:, :, ::-1].copy()  # RGB (PIL) -> BGR (OpenCV)


def sujar_imagem(imagem: np.ndarray, semente: int = 42) -> np.ndarray:
    """Simula foto de celular ruim: metade da imagem escurecida (sombra) + ruído."""
    rng = np.random.default_rng(semente)
    ruido = rng.normal(0, 35, imagem.shape).astype(np.int16)
    sombreada = imagem.astype(np.int16)
    sombreada[:, imagem.shape[1] // 2 :] = (sombreada[:, imagem.shape[1] // 2 :] * 0.4).astype(np.int16)
    return np.clip(sombreada + ruido, 0, 255).astype(np.uint8)


def test_le_imagem_limpa_sem_preprocessamento():
    imagem = imagem_com_texto("Texto de teste 123")
    assert extrair_texto(imagem, "por") == "Texto de teste 123"


def test_le_acentuacao_e_pontuacao():
    imagem = imagem_com_texto("Código não é mágica, R$ 97,50.", largura=900)
    assert extrair_texto(imagem, "por") == "Código não é mágica, R$ 97,50."


def test_preprocessamento_recupera_leitura_em_imagem_com_ruido_e_sombra():
    limpa = imagem_com_texto("Texto de teste 123")
    suja = sujar_imagem(limpa)

    sem_preproc = extrair_texto(suja, "por")
    com_preproc = extrair_texto(preprocessar(suja), "por")

    # sem tratar a imagem, sobra lixo de ruído grudado no texto
    assert sem_preproc != "Texto de teste 123"
    # com o tratamento (denoise + normalização de sombra + Otsu), volta a ler certo
    assert com_preproc == "Texto de teste 123"


def test_preprocessamento_devolve_imagem_de_1_canal_binaria():
    imagem = imagem_com_texto("qualquer coisa")
    resultado = preprocessar(imagem)

    assert resultado.ndim == 2  # 1 canal (cinza/binário), não 3 (BGR)
    assert set(np.unique(resultado)) <= {0, 255}  # só preto e branco


def test_salvar_resultado_txt(tmp_path):
    destino = tmp_path / "saida.txt"
    salvar_resultado(destino, tmp_path / "foto.png", "texto extraido")
    assert destino.read_text(encoding="utf-8") == "texto extraido"


def test_salvar_resultado_json(tmp_path):
    destino = tmp_path / "saida.json"
    salvar_resultado(destino, tmp_path / "foto.png", "texto extraido")

    conteudo = destino.read_text(encoding="utf-8")
    assert '"texto": "texto extraido"' in conteudo
    assert "foto.png" in conteudo
