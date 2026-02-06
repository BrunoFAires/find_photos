import json
import argparse
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="images_source.json")
    parser.add_argument("--output_dir", default="imagens")
    parser.add_argument("--start_name", required=True)
    parser.add_argument("--type", choices=["original", "thumb"], default="original")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_images(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        images = json.load(f)

    return sorted(images, key=lambda x: x["filename"])


def slice_from_start_name(images, start_name):
    names = [img["filename"] for img in images]

    if start_name not in names:
        raise ValueError(f"start_name '{start_name}' not found")

    start_idx = names.index(start_name)
    sliced = images[start_idx:]

    print(f"Starting from: {sliced[0]['filename']}")
    return sliced


def ensure_output_dir(path):
    path.mkdir(exist_ok=True)


def download_image(item, out_dir, img_type):
    filename = item["filename"]

    if img_type not in item:
        return f"[SKIP] {filename} has no '{img_type}' field"

    out_path = out_dir / f"{filename}.jpg"
    if out_path.exists():
        return f"[SKIP] {filename}.jpg exists"

    try:
        r = requests.get(item[img_type], timeout=30)
        r.raise_for_status()

        with open(out_path, "wb") as f:
            f.write(r.content)

        return f"[OK] {filename}.jpg"

    except Exception as e:
        return f"[ERROR] {filename}: {e}"


def download_parallel(images, out_dir, img_type, workers):
    print(f"Workers: {workers}")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_image, item, out_dir, img_type)
            for item in images
        ]

        for future in as_completed(futures):
            print(future.result())


def main():
    args = parse_args()

    images = load_images(args.input)
    images = slice_from_start_name(images, args.start_name)

    out_dir = Path(args.output_dir)
    ensure_output_dir(out_dir)

    download_parallel(
        images=images,
        out_dir=out_dir,
        img_type=args.type,
        workers=args.workers
    )

    print(f"Done.\nTotal images processed: {len(images)}")


if __name__ == "__main__":
    main()
