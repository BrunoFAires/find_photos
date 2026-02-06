import os
import shutil
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

DIR_REFERENCE = "reference"
DIR_INPUT = "imagens"
DIR_OUTPUT = "matched"

THRESHOLD = 0.8

device = torch.device("cpu")

os.makedirs(DIR_OUTPUT, exist_ok=True)

mtcnn = MTCNN(image_size=160, margin=20, device=device)
facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)

    if face is None:
        return None

    face = face.unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = facenet(face)

    return embedding.cpu().numpy()[0]


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

reference_embeddings = []

for file in os.listdir(DIR_REFERENCE):
    path = os.path.join(DIR_REFERENCE, file)
    emb = get_embedding(path)

    if emb is not None:
        reference_embeddings.append(emb)

if not reference_embeddings:
    raise RuntimeError("No face detected in reference images")

reference_embedding = np.mean(reference_embeddings, axis=0)
print("Reference embedding created.")
input_files = os.listdir(DIR_INPUT)
input_files.sort()

for file in input_files:
    path = os.path.join(DIR_INPUT, file)

    try:
        emb = get_embedding(path)
        if emb is None:
            print(f"No face detected in {file}")
            continue

        similarity = cosine_similarity(reference_embedding, emb)
        print(f"{file} → similarity: {similarity:.3f}")

        if similarity >= THRESHOLD:
            shutil.copy(path, os.path.join(DIR_OUTPUT, file))
            print(f"✔ {file} matched")

    except Exception as e:
        print(f"Error processing {file}: {e}")

print("Done.")
