import argparse
import cv2
import numpy as np
import pandas as pd
import tqdm
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.data.image import get_image as insightface_get_image
from pathlib import Path


def load_dataframes(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"Scanning data path: {data_path}")
    data_path_df = pd.DataFrame({
        "document_pages_dir": [p for p in data_path.rglob("*.images") if p.is_dir()],
    })
    print(f"Found {len(data_path_df)} document page directories")
    data_path_df['library'] = data_path_df['document_pages_dir'].apply(lambda p: p.parent.name)
    data_path_df['document'] = data_path_df['document_pages_dir'].apply(lambda p: p.stem)

    faces_records: list[dict] = []
    for p in data_path.rglob("*.people_gator.jsonl"):
        if not p.is_file():
            continue
        try:
            recs = pd.read_json(p, lines=True).to_dict(orient="records")
            faces_records.extend(recs)
        except Exception as e:
            print(f"Warning: failed to read {p}: {e}")

    data_faces_df = pd.DataFrame(faces_records)
    print(f"Loaded faces dataframe with {len(data_faces_df)} records")

    data_path_df['page'] = data_path_df['document_pages_dir'].apply(lambda p: [q.name for q in p.glob("*.jpg")])
    data_path_df = data_path_df.explode("page").reset_index(drop=True)
    data_faces_df = data_faces_df.merge(data_path_df, on=["library", "document", "page"], how="outer")
    print(f"After merge, faces dataframe has {len(data_faces_df)} rows")
    return data_path_df, data_faces_df


def init_app(model_name: str, providers: list[str]):
    print(f"Initializing insightface model {model_name} with providers={providers}")
    app = FaceAnalysis(name=model_name, providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    # sanity check loaded model
    image = insightface_get_image("t1")
    faces = app.get(image)
    print(f"Sanity check: model returned {len(faces)} faces for test image")

    recognition_models = []
    for m in app.models.values():
        if getattr(m, 'taskname', '') == 'recognition':
            recognition_models.append(m)
    print(f"Found {len(recognition_models)} recognition model(s)")
    return app, recognition_models


def process_and_save(recognition_models, data_faces_df: pd.DataFrame, data_path: Path, output_dir: Path, model_name: str):
    all_faces: dict[str, np.ndarray] = {}
    tqdm_bar = tqdm.tqdm(total=len(data_faces_df), desc="Processing pages")
    for (library, document, page), g in data_faces_df.groupby(["library", "document", "page"]):
        library, document, page = str(library), str(document), str(page)
        page_path = data_path / library / (document + ".images") / page
        img = cv2.imread(str(page_path))

        faces = {}
        for face_name, page_keypoints, confidence in g[['crop_name', 'page_keypoints', 'confidence']].values:
            if not face_name or not page_keypoints or not confidence:
                tqdm_bar.write(f"No faces and keypoints found for {library}/{document}/{page} with face name {face_name}. Skipping.")
                tqdm_bar.update(1)
                continue
            page_keypoints = np.array(page_keypoints, dtype=float)
            face = Face(bbox=None, kps=page_keypoints, det_score=confidence)
            faces[f"{library}/{document}.peoplegator_aligned_crops/{face_name}"] = face

        for recognition_model in recognition_models:
            for face_name, face in faces.items():
                try:
                    recognition_model.get(img, face)
                except Exception as e:
                    tqdm_bar.write(f"Error processing face {face_name} in {library}/{document}/{page}: {e}")
                    continue

        for face_name, face in faces.items():
            if face.embedding is None:
                tqdm_bar.write(f"No embedding found for face {face_name} in {library}/{document}/{page}. Skipping.")
                tqdm_bar.update(1)
                continue
            all_faces[face_name] = face.embedding
            tqdm_bar.set_postfix_str(f"Processed {face_name}")
            tqdm_bar.update(1)

    sorted_faces = sorted(all_faces.keys())
    embeddings = np.array([all_faces[face_name] for face_name in sorted_faces])

    (output_dir / model_name).mkdir(parents=True, exist_ok=True)
    print(f"Saving {len(sorted_faces)} embeddings to {output_dir / model_name}")
    with open(output_dir / model_name / "image_paths.txt", "w") as f:
        f.write("\n".join(sorted_faces))

    np.save(output_dir / model_name / "embeddings.npy", embeddings)


def main():
    parser = argparse.ArgumentParser(description='Embed faces using insightface on PeopleGator data structure.')
    parser.add_argument('-d', '--data-path', type=Path, default=Path("/mnt/backup/HistoricalNERProcessing/notebooks/people_gator__data"), help='Base data path containing .images and .people_gator.jsonl files')
    parser.add_argument('-o', '--output-dir', type=Path, default=Path(".embeddings"), help='Directory to save the embeddings and image paths')
    parser.add_argument('-m', '--model', type=str, default='buffalo_l', help='InsightFace model name')
    parser.add_argument('-p', '--providers', type=str, default='CUDAExecutionProvider,CPUExecutionProvider', help='Comma-separated providers for ONNX/onnxruntime')
    args = parser.parse_args()
    data_path = args.data_path
    output_dir = args.output_dir
    model_name = args.model
    providers = [p.strip() for p in args.providers.split(',') if p.strip()]
    _, data_faces_df = load_dataframes(data_path)
    _, recognition_models = init_app(model_name, providers)
    process_and_save(recognition_models, data_faces_df, data_path, output_dir, model_name)


if __name__ == "__main__":
    main()
