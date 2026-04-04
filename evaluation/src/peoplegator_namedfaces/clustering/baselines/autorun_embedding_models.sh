#!/bin/bash
models=(
    "hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3"
    "hf_hub:gaunernst/vit_tiny_patch8_112.cosface_ms1mv3"
    "hf_hub:gaunernst/vit_tiny_patch8_112.arcface_ms1mv3"
    "hf_hub:gaunernst/vit_tiny_patch8_112.adaface_ms1mv3"
    "hf_hub:gaunernst/convnext_nano.cosface_ms1mv3"
    "hf_hub:gaunernst/convnext_atto.cosface_ms1mv3"
)
# Run the embedding script for each model
## python src/embed_faces.py \
##     -i .data \
##     -o .embeddings/vit_small_patch8_gap_112.cosface_ms1mv3 \
##     -d cuda \
##     -m hf_hub:gaunernst/vit_small_patch8_gap_112.cosface_ms1mv3

for model in "${models[@]}"; do
    # extract part after ':' then take last segment after '/'
    model_name="${model#*:}"
    model_name="${model_name##*/}"
    uv run python src/embed_faces.py \
        -i .data \
        -o .embeddings/"$model_name" \
        -d cuda \
        -m "$model"
done

models=(
    "antelopev2"
    "buffalo_l"
    "buffalo_m"
    "buffalo_s"
    "buffalo_sc"
)
for model in "${models[@]}"; do
    # extract part after ':' then take last segment after '/'
    model_name="${model#*:}"
    model_name="${model_name##*/}"
    uv run python src/embed_faces_insightface.py \
        -o .embeddings/ \
        -m "$model"
done

models=(
    "minchul/cvlface_adaface_vit_base_kprpe_webface4m"
    "minchul/cvlface_adaface_vit_base_webface4m"
    "minchul/cvlface_adaface_vit_base_kprpe_webface12m"
    "minchul/cvlface_adaface_ir101_webface12m"
    "minchul/cvlface_adaface_ir101_ms1mv3"
    "minchul/cvlface_adaface_ir101_webface4m"
    "minchul/cvlface_adaface_ir101_ms1mv2"
    "minchul/cvlface_adaface_ir50_ms1mv2"
    "minchul/cvlface_adaface_ir50_casia"
    "minchul/cvlface_adaface_ir50_webface4m"
    "minchul/cvlface_adaface_ir18_webface4m"
    "minchul/cvlface_arcface_ir101_webface4m"
    "minchul/cvlface_adaface_ir18_casia"
    "minchul/cvlface_adaface_ir18_vgg2"
)

for model in "${models[@]}"; do
    # extract part after ':' then take last segment after '/'
    printf "Processing model: %s\n" "$model"
    model_name="${model#*:}"
    model_name="${model_name##*/}"
    uv run python src/embed_faces_cvlface.py \
        -i .data \
        -o ".embeddings/$model_name" \
        -m "$model"
done

model=swinface
printf "Processing model: %s\n" "$model"
uv run python src/embed_faces_swinface.py \
    -i .data \
    -o ".embeddings/$model" \
    -m ".weights/swinface.pt"
