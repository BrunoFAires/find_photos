import json
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from numpy.linalg import norm

# ================== CONFIG ==================
REFERENCE_IMAGE = "reference/1.png"
EMBEDDINGS_JSON = "embeddings.json"
INPUT_IMAGES_DIR = "imagens"
OUTPUT_DIR = "output"
SIMILARITY_THRESHOLD = 0.7

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== MODELS ==================
mtcnn = MTCNN(
    image_size=160,
    margin=40,
    keep_all=True,
    device=device
)

facenet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)

# ================== HELPERS ==================
def load_image(path, max_size=1600):
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))

    return img


def get_embeddings(image_path):
    img = load_image(image_path)
    faces = mtcnn(img)

    if faces is None:
        return np.empty((0, 512))

    faces = faces.to(device)

    with torch.no_grad():
        embeddings = facenet(faces)

    return embeddings.cpu().numpy()


def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


# ================== LOAD EMBEDDINGS ==================
with open(EMBEDDINGS_JSON, "r", encoding="utf-8") as f:
    stored = json.load(f)

stored_embeddings = np.array([item["embedding"] for item in stored])
print(f"Loaded {len(stored)} stored embeddings")
print("ultima imagem:", stored[-1]["image"])
# ================== REFERENCE ==================
print("Generating reference embedding(s)...")

ref_embeddings = get_embeddings(REFERENCE_IMAGE)

if len(ref_embeddings) == 0:
    raise RuntimeError("No face detected in reference image")

print(f"Detected {len(ref_embeddings)} face(s) in reference")

# ================== SEARCH & SAVE ==================
output_root = Path(OUTPUT_DIR)
output_root.mkdir(exist_ok=True)

saved = set()

for ref_idx, ref_emb in enumerate(ref_embeddings):
    ref_dir = output_root / f"ref_face_{ref_idx}"
    ref_dir.mkdir(exist_ok=True)

    for i, item in enumerate(stored):
        sim = cosine_similarity(ref_emb, stored_embeddings[i])

        if sim >= SIMILARITY_THRESHOLD:
            src = Path(INPUT_IMAGES_DIR) / item["image"]

            if not src.exists():
                continue

            # evita copiar a mesma imagem várias vezes
            key = (ref_idx, src.name)
            if key in saved:
                continue

            dst = ref_dir / src.name
            shutil.copy2(src, dst)

            saved.add(key)

            print(
                f"[SAVED] ref_face {ref_idx} ← {src.name} "
                f"(sim={sim:.3f})"
            )

print(f"\nTotal images saved: {len(saved)}")
