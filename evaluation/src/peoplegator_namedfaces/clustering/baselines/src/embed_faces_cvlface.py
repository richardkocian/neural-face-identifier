import cv2
from peoplegator_namedfaces.clustering.baselines.src.embed_faces import (embed_face_dir)
import pathlib
import torch
import numpy as np
import os
from huggingface_hub.file_download import hf_hub_download
from transformers import AutoModel
import sys
import shutil

def download(repo_id: str, path: str, HF_TOKEN: bool | str | None=None):
    os.makedirs(path, exist_ok=True)
    files_path = os.path.join(path, 'files.txt')
    if not os.path.exists(files_path):
        hf_hub_download(repo_id, 'files.txt', token=HF_TOKEN,
                        local_dir=path)
    with open(os.path.join(path, 'files.txt'), 'r') as f:
        files = f.read().split('\n')
    for file in [f for f in files if f] + ['config.json', 'wrapper.py', 'model.safetensors']:
        full_path = os.path.join(path, file)
        if not os.path.exists(full_path):
            hf_hub_download(repo_id, file, token=HF_TOKEN,
                            local_dir=path)


# helpfer function to download huggingface repo and use model
def load_model_from_local_path(path, HF_TOKEN: bool | str | None=None):
    cwd = os.getcwd()
    os.chdir(path)
    sys.path.insert(0, path)
    model = AutoModel.from_pretrained(
        path, trust_remote_code=True, token=HF_TOKEN)
    os.chdir(cwd)
    sys.path.pop(0)
    return model


# helpfer function to download huggingface repo and use model
def load_model_by_repo_id(repo_id, save_path, HF_TOKEN: bool | str | None=None, force_download=False):
    if force_download:
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
    download(repo_id, save_path, HF_TOKEN)
    return load_model_from_local_path(save_path, HF_TOKEN)


def image_transform(img: np.ndarray) -> np.ndarray:
    img = img[:,:,::-1]  # BGR to RGB
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img /= 255.0
    img -= 0.5
    img /= 0.5
    return img


def get_models(repo: str = "minchul/cvlface_adaface_vit_base_kprpe_webface4m", HF_TOKEN: bool | str | None = None):
    path = os.path.expanduser(f"~/.cvlface_cache/{repo}")
    repo_id = repo
    model = load_model_by_repo_id(repo_id, path, HF_TOKEN)
    if "_kprpe_" in repo:
        aligner = load_model_by_repo_id(
            'minchul/cvlface_DFA_mobilenet', path, HF_TOKEN)
        return model, aligner
    return model, None

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Embed faces using a pre-trained SwinFace model.')
    parser.add_argument('-i', '--input', type=pathlib.Path,
                        required=True, help='Directory containing face images.')
    parser.add_argument('-o', '--output', type=pathlib.Path,
                        required=True, help='Directory to save the embeddings and image paths.')
    parser.add_argument('-m', '--model-repo', type=str, required=True, help='Path to the SwinFace model weights.')
    args = parser.parse_args()
    TOKEN = "TODO-ADD-TOKEN" # TODO add token
    raise ValueError("Add token")
    model, aligner = get_models(args.model_repo, HF_TOKEN=TOKEN)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    if aligner is not None:
        aligner.to(device)
        aligner.eval()
    
    def _krpe_wrapper(img_tensor: torch.Tensor) -> torch.Tensor:
        assert aligner is not None, "Aligner model is required for KPRPE models"
        with torch.inference_mode():
            aligned_x, orig_ldmks, aligned_ldmks, score, thetas, bbox = aligner(img_tensor)
            output = model(aligned_x, orig_ldmks)
            return output
    
    def _model_wrapper(img_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            output = model(img_tensor)
            return output
        
    if aligner is not None:
        embeddings, image_paths = embed_face_dir(_krpe_wrapper, image_transform, args.input, device)
    else:
        embeddings, image_paths = embed_face_dir(_model_wrapper, image_transform, args.input, device)
    
    image_paths_contets = "\n".join(image_paths) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", embeddings)
    with open(args.output / "image_paths.txt", "w") as f:
        f.write(image_paths_contets)

if __name__ == "__main__":
    main()
