from pathlib import Path

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    ImageFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

def doc2text(dossier: Path):
    """
    Convertit les fichiers PDF et images dans le dossier spécifié en fichiers Markdown.
    Les fichiers Markdown existants ne sont pas remplacés.
    """
    pipeline_options = PdfPipelineOptions()

    # PDF
    pipeline_options.layout_options.engine_options.compile_model = False

    # Classification d'images
    pipeline_options.picture_classification_options.engine_options.compile_model = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            ),
            InputFormat.IMAGE: ImageFormatOption(
                pipeline_options=pipeline_options
            ),
        }
    )

    print(f"Start scan : {dossier}")
    for element in dossier.rglob("*"):
        if not element.is_file():
            continue

        if element.suffix.lower() == ".md":
            continue

        if element.with_suffix(".md").exists():
            print(f"Fichier : {element.name} (déjà traité)", flush=True)
            continue

        print(f"Fichier : {element.name}", flush=True)
        try:
            print(str(element.resolve()))
            result = converter.convert(str(element.resolve()))

            sortie = element.with_suffix(".md")
            print(f"Sortie : {sortie}")
            sortie.write_text(
                result.document.export_to_markdown(),
                encoding="utf-8"
            )
        except Exception as e:
            print(f"Erreur lors de la conversion de {element}: {e}")
            continue