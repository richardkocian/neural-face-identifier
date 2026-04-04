from collections.abc import Callable

import numpy as np
import timm
import cv2
import pathlib
import torch
import tqdm

ARCFACE_KEYPOINTS = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
        [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)

def image_transform(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (image - 127.5) / 128.0
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    return image

@torch.inference_mode()
def embed_face_dir(model: Callable[..., torch.Tensor], transform: Callable[[np.ndarray], np.ndarray], face_dir: pathlib.Path, device: torch.device):
    embeddings: list[np.ndarray] = []
    image_paths: list[str] = []
    images = [p for ext in ['jpg', 'jpeg', 'png'] for p in tqdm.tqdm(face_dir.rglob(f'*.{ext}'), desc=f"Counting images for extension {ext}")]
    for img_path in tqdm.tqdm(images, desc="Embedding faces"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = transform(img)
        embedding = model(torch.from_numpy(img).unsqueeze(0).to(device)).squeeze().detach().cpu().numpy()
        embeddings.append(embedding)
        image_paths.append(str(img_path.relative_to(face_dir)))
    return np.array(embeddings), image_paths

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Embed faces using a pre-trained model.')
    parser.add_argument('-i', '--input', type=pathlib.Path,
                        required=True, help='Directory containing face images.')
    parser.add_argument('-o', '--output', type=pathlib.Path,
                        required=True, help='Directory to save the embeddings and image paths.')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to use for computation (default: cuda).')
    parser.add_argument('-m', '--model', type=str, default='hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3',
                        help='Pre-trained model to use (default: hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3).')
    args = parser.parse_args()
    print(f"Using device: {args.device}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = timm.create_model(args.model, pretrained=True)
    model.to(device)
    model.eval()
    embeddings, image_paths = embed_face_dir(model, image_transform, args.input, device)
    image_paths_contets = "\n".join(image_paths) + "\n"
    
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", embeddings)
    with open(args.output / "image_paths.txt", "w") as f:
        f.write(image_paths_contets)


if __name__ == "__main__":    
    main()
    