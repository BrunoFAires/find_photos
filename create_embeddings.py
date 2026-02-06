import json
import argparse
from pathlib import Path
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="imagens")
    parser.add_argument("--output", default="embeddings.json")
    parser.add_argument("--start_name", type=str, default=None)
    parser.add_argument("--max_faces", type=int, default=15)
    return parser.parse_args()

def load_image(path, max_size=1600):
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))

    return img

def get_embeddings(image_path, mtcnn, facenet, device, max_faces):
    img = load_image(image_path)
    faces = mtcnn(img)

    if faces is None:
        return []

    faces = faces[:max_faces].to(device)

    with torch.no_grad():
        emb = facenet(faces)

    return emb.cpu().numpy()

def list_images(input_dir):
    files = sorted(
        [p for p in input_dir.iterdir()
         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name
    )

    if not files:
        raise RuntimeError("No images found in input_dir")

    return files

def load_existing_results(output_path):
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mtcnn = MTCNN(
        image_size=160,
        margin=40,
        keep_all=True,
        device=device
    )

    facenet = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(device)

    input_dir = Path(args.input_dir)
    files = list_images(input_dir)

    start_idx = 0
    if args.start_name:
        names = [p.name for p in files]
        if args.start_name not in names:
            raise ValueError(f"start_name '{args.start_name}' not found")
        start_idx = names.index(args.start_name)

    print(f"Starting from index {start_idx}: {files[start_idx].name}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = load_existing_results(output_path)

    for idx, path in enumerate(files[start_idx:], start=start_idx):
        print(f"[{idx}] Processing {path.name}")

        embeddings = get_embeddings(
            path,
            mtcnn=mtcnn,
            facenet=facenet,
            device=device,
            max_faces=args.max_faces
        )

        for face_idx, emb in enumerate(embeddings):
            results.append({
                "image": path.name,
                "face_index": face_idx,
                "embedding": emb.tolist()
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f)

    print("Done.")


if __name__ == "__main__":
    main()
