import json
import argparse
import requests
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_dir", default="reference")
    parser.add_argument("--embeddings", default="embeddings.json")
    parser.add_argument("--images_source", default="images_source.json")
    parser.add_argument("--output_dir", default="output/matched")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--type", choices=["original", "thumb"], default="original")
    parser.add_argument("--workers", type=int, default=8)

    return parser.parse_args()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_reference_embedding(ref_dir, device):
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    facenet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    embeddings = []

    for img_path in Path(ref_dir).iterdir():
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        img = Image.open(img_path).convert("RGB")
        face = mtcnn(img)

        if face is None:
            continue

        face = face.unsqueeze(0).to(device)

        with torch.no_grad():
            emb = facenet(face)

        embeddings.append(emb.cpu().numpy()[0])

    if not embeddings:
        raise RuntimeError("No face detected in reference images")

    return np.mean(embeddings, axis=0)


def find_matching_images(embeddings_file, reference_embedding, threshold):
    with open(embeddings_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    matched_images = set()

    for item in data:
        emb = np.array(item["embedding"])
        sim = cosine_similarity(reference_embedding, emb)

        if sim >= threshold:
            matched_images.add(item["image"])

    return sorted(matched_images)


def download_images(images_source, selected_names, output_dir, img_type, workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with open(images_source, "r", encoding="utf-8") as f:
        source = json.load(f)

    index = {item["filename"] + ".jpg": item for item in source}
    output_dir.mkdir(parents=True, exist_ok=True)

    def download(name):
        if name not in index:
            return f"[SKIP] {name} not found in images_source"

        item = index[name]

        if img_type not in item:
            return f"[SKIP] {name} has no '{img_type}'"

        out_path = output_dir / name
        if out_path.exists():
            return f"[SKIP] {name} exists"

        try:
            r = requests.get(item[img_type], timeout=30)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            return f"[OK] {name}"
        except Exception as e:
            return f"[ERROR] {name}: {e}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(download, n) for n in selected_names]
        for f in as_completed(futures):
            print(f.result())

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building reference embedding...")
    ref_emb = build_reference_embedding(args.reference_dir, device)

    print("Searching for matches...")
    matched = find_matching_images(
        args.embeddings,
        ref_emb,
        args.threshold
    )

    print(f"Matched images: {len(matched)}")

    if not matched:
        print("No matches found.")
        return

    print("Downloading matched images...")
    download_images(
        args.images_source,
        matched,
        Path(args.output_dir),
        args.type,
        args.workers
    )

    print("Done.")


if __name__ == "__main__":
    main()
