import os
import httpx
import numpy as np
import os
import time
import httpx
import numpy as np
import os
import time
import httpx
import numpy as np


def make_embeddings(
    text: str
) -> np.ndarray:

    max_retries = 5

    url = "https://api.mistral.ai/v1/embeddings"

    for attempt in range(max_retries):
        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"
                },
                json={
                    "model": "mistral-embed",
                    "input": text
                },
                timeout=30,
            )

            response.raise_for_status()

            data = response.json()

            return np.array(
                data["data"][0]["embedding"],
                dtype=np.float32
            )

        except httpx.HTTPStatusError as e:

            if e.response.status_code != 429:
                raise

            delay = 2 ** attempt

            print(
                f"Erreur 429 : nouvelle tentative dans {delay}s "
                f"({attempt + 1}/{max_retries})",
                flush=True
            )

            time.sleep(delay)

    raise RuntimeError(
        f"Échec après {max_retries} tentatives"
    )