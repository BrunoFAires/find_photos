import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="imagens")
    parser.add_argument("--output", default="embeddings.json")
    parser.add_argument("--start_name", type=str, default=None)
    parser.add_argument("--max_faces", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def load_image(path, max_size=1600):
    img = Image.open(path).convert("RGB")
    w, h = img.size

    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)))

    return img

def process_image(args):
    """
    Executado em PROCESSO separado
    """
    image_path, max_faces = args

    device = torch.device("cpu")

    mtcnn = MTCNN(
        image_size=160,
        margin=40,
        keep_all=True,
        device=device
    )

    facenet = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(device)

    try:
        img = load_image(image_path)
        faces = mtcnn(img)

        if faces is None:
            return []

        faces = faces[:max_faces]

        with torch.no_grad():
            emb = facenet(faces)

        results = []
        for idx, e in enumerate(emb):
            results.append({
                "image": image_path.name,
                "face_index": idx,
                "embedding": e.cpu().numpy().tolist()
            })

        return results

    except Exception as e:
        return [{
            "image": image_path.name,
            "error": str(e)
        }]


def list_images(input_dir):
    files = sorted(
        [p for p in input_dir.iterdir()
         if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda p: p.name
    )

    if not files:
        raise RuntimeError("No images found")

    return files


def load_existing_results(output_path):
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    files = list_images(input_dir)

    start_idx = 0
    if args.start_name:
        names = [p.name for p in files]
        if args.start_name not in names:
            raise ValueError(f"start_name '{args.start_name}' not found")
        start_idx = names.index(args.start_name)

    files = files[start_idx:]
    print(f"Starting from: {files[0].name}")
    print(f"Workers: {args.workers}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = load_existing_results(output_path)

    tasks = [(p, args.max_faces) for p in files]

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_image, t) for t in tasks]

        for future in as_completed(futures):
            res = future.result()
            if res:
                results.extend(res)

                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f)

    print("Done.")


if __name__ == "__main__":
    main()
