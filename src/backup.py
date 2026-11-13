"""Cópia de segurança dos resultados do M5 (ex.: para o Google Drive no Colab)."""
import shutil
from pathlib import Path

from src import config as C


def salvar_backup(destino):
    """Copia {PREFIX}* de TAB_DIR e pred_{PREFIX}* de PROCESSED_DIR para `destino`."""
    if not destino:
        return
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    arquivos = list(C.TAB_DIR.glob(f"{C.PREFIX}*")) + list(C.PROCESSED_DIR.glob(f"pred_{C.PREFIX}*"))
    for f in arquivos:
        shutil.copy2(f, destino / f.name)
    print(f"backup: {len(arquivos)} arquivos -> {destino}")
