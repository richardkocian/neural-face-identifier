from peoplegator_namedfaces.clustering.baselines.src.third_party.swinface.inference import get_model, image_transform
from peoplegator_namedfaces.clustering.baselines.src.embed_faces import embed_face_dir
import pathlib
import torch
import numpy as np

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Embed faces using a pre-trained SwinFace model.')
    parser.add_argument('-i', '--input', type=pathlib.Path,
                        required=True, help='Directory containing face images.')
    parser.add_argument('-o', '--output', type=pathlib.Path,
                        required=True, help='Directory to save the embeddings and image paths.')
    parser.add_argument('-d', '--device', type=str, default='cuda', help='Device to use for computation (default: cuda).')
    parser.add_argument('-m', '--model-weight', type=pathlib.Path, required=True, help='Path to the SwinFace model weights.')
    args = parser.parse_args()
    print(f"Using device: {args.device}")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model_weight)
    model.to(device)
    model.eval()
    def _model_wrapper(img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            output = model(img_tensor)
            return output['Recognition']
    embeddings, image_paths = embed_face_dir(_model_wrapper, image_transform, args.input, device)
    image_paths_contets = "\n".join(image_paths) + "\n"
    
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", embeddings)
    with open(args.output / "image_paths.txt", "w") as f:
        f.write(image_paths_contets)

if __name__ == "__main__":
    main()