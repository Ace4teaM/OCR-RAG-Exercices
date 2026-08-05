import re
from pathlib import Path

def split_markdown_by_headers(file: Path):
    """
    Découpe un document Markdown en blocs selon les titres.

    Retourne une liste de dictionnaires :
    {
        "title": "Automate > Démarrage > Vérification",
        "headers": {
            "h1": "Automate",
            "h2": "Démarrage",
            "h3": "Vérification"
        },
        "filename": "nom du fichier"
        "content": "..."
    }
    """
    filename = str(file.resolve())

    with open(filename, "r", encoding="utf-8") as f:
        markdown = f.read()

        header_regex = re.compile(r"^(#{1,6})\s+(.*)$")

        current_headers = {}
        current_content = []

        chunks = []

        def save_chunk():
            """Sauvegarde le bloc courant."""
            content = "\n".join(current_content).strip()

            if content:
                ordered_headers = [
                    current_headers[f"h{i}"]
                    for i in range(1, 7)
                    if f"h{i}" in current_headers
                ]

                chunks.append({
                    "title": " > ".join(ordered_headers),
                    "headers": current_headers.copy(),
                    "filename": file.name,
                    "path": filename,
                    "content": content
                })

        for line in markdown.splitlines():

            match = header_regex.match(line)

            if match:
                # Nouveau titre rencontré

                save_chunk()

                level = len(match.group(1))
                title = match.group(2).strip()

                # Supprime les niveaux inférieurs
                for i in range(level, 7):
                    current_headers.pop(f"h{i}", None)

                current_headers[f"h{level}"] = title

                current_content = []

            else:
                current_content.append(line)

        save_chunk()

        return chunks

