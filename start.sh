#!/bin/bash
# Le symbole "&" lance la commande en arrière-plan

# Linear PCA
.venv/bin/python scripts/pca/pca_explorer.py \
    --model scripts/pca/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --sliders 5 \
    --range 3.0 &

# Latent explorer
.venv/bin/python scripts/visualization/latent_explorer.py models/NeuralNetwork_centered_tanh.bin data/mean_femur.obj &

# PCA explorer
.venv/bin/python scripts/visualization/latent_projection/project_training_femurs.py --model NeuralNetwork_centered_tanh.bin 
.venv/bin/python scripts/visualization/latent_pca_explorer_pca.py models/NeuralNetwork_centered_tanh.bin mean_femur.obj -n 3 --start-at-mean &

# Non linear PCA
.venv/bin/python scripts/pca/tangent_pca_explorer.py \
    --model scripts/pca/model/tangent_pca \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --components 7 \
    --sigma 3.0 &


.venv/bin/python scripts/visualization/comparison medical_femur_analysis.py data/training/L_Femur_11_DECIM.obj.FINAL.obj &


# "wait" empêche le script principal de se fermer tout de suite
wait