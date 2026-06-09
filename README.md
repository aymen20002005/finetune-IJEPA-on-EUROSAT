# I-JEPA sur EuroSAT: Linear Probing et kNN

Évaluation du modèle **I-JEPA** (Image-based Joint-Embedding Predictive Architecture) de Meta/Facebook sur **EuroSAT** avec deux modes uniquement: **linear probing** et **kNN**.

## 🎯 Description

Ce projet utilise le modèle **I-JEPA ViT-H/14** pré-entraîné disponible sur Hugging Face (`facebook/ijepa_vith14_1k`) sur le dataset:
- **EuroSAT** : imagerie satellite (10 classes)

## 🚀 Installation Rapide

```bash
# Cloner le repository
git clone <url>
cd finetune-IJEPA

# Installer les dépendances
pip install -r requirements.txt
```

## 📦 Dépendances Principales

- PyTorch >= 2.0.0
- transformers >= 4.30.0 (pour charger I-JEPA depuis Hugging Face)
- torchvision >= 0.15.0

## 🎓 Utilisation

### Exécution Simple

```bash
# EuroSAT (par défaut)
python train_ijepa.py --dataset EuroSAT
```

### Modes d'Entraînement

**Linear Probing** (seule la tête de classification est entraînée) :
```bash
python train_ijepa.py --dataset EuroSAT --mode linear_probe
```

**kNN** (aucun entraînement par gradient, classification sur features gelées) :
```bash
python train_ijepa.py --dataset EuroSAT --mode knn --knn-k 20
```

### Options Avancées

```bash
python train_ijepa.py \
  --dataset EuroSAT \
  --mode linear_probe \
  --epochs 20 \
  --batch-size 32 \
  --lr 1e-3
```

### Configuration Personnalisée

Modifiez `config.yaml` puis :
```bash
python train_ijepa.py --config config.yaml
```

## 📊 Structure du Projet

```
finetune-IJEPA/
├── src/
│   ├── pretrained_models.py  # Chargement I-JEPA depuis Hugging Face
│   ├── data_loader.py        # Dataset EuroSAT (CSV)
│   └── train.py              # Linear probing + kNN
├── train_ijepa.py            # Script principal
├── config.yaml               # Configuration par défaut
├── requirements.txt          # Dépendances
└── README.md                 # Ce fichier
```

## 🔍 Comment Ça Marche ?

1. **Chargement du Modèle** : I-JEPA ViT-H/14 est téléchargé depuis Hugging Face
2. **Ajout d'une Tête de Classification** : Une couche linéaire pour la prédiction de classes
3. **Linear probing** : Entraînement supervisé de la tête de classification (encodeur gelé)
4. **kNN** : Extraction de features CLS puis classification k-plus proches voisins
5. **Évaluation** : Accuracy sur le split test

## 📈 Monitoring

Utilisez TensorBoard pour visualiser l'entraînement :

```bash
tensorboard --logdir logs/
```

Métriques disponibles :
- Loss (train/test)
- Accuracy (train/test)
- Learning rate

## 💾 Checkpoints

Les modèles sont sauvegardés dans `checkpoints/` (mode `linear_probe`) :
- `best_model.pth` : Meilleur modèle (meilleure accuracy)
- `final_model.pth` : Modèle final
- `checkpoint_epoch_X.pth` : Checkpoints périodiques

## ⚙️ Configuration par Défaut

```yaml
dataset:
  name: 'EuroSAT'
  data_path: '/kaggle/input/datasets/apollo2506/eurosat-dataset/EuroSAT'
  batch_size: 64
  image_size: 224

model:
  pretrained_name: 'ijepa-vith14-1k'  # facebook/ijepa_vith14_1k

training:
  mode: 'linear_probe'
  knn_k: 20
  epochs: 20
  base_lr: 1e-3
  
finetuning:
  freeze_encoder: True
```

## 🎯 Performances Attendues

Le dataset EuroSAT est utilisé via les fichiers `train.csv` et `test.csv` présents dans le dossier `/kaggle/input/datasets/apollo2506/eurosat-dataset/EuroSAT`.

### EuroSAT
- **Linear probe** : accuracy dépend des hyperparamètres (epochs, LR, batch size)
- **kNN** : accuracy dépend principalement de `--knn-k` et de la qualité des features

*Note : Les performances dépendent des hyperparamètres*

## 🛠️ Troubleshooting

**CUDA Out of Memory** :
```bash
python train_ijepa.py --batch-size 32  # Réduire batch size
```

**Entraînement lent** :
- Vérifier que CUDA est disponible : `torch.cuda.is_available()`
- Augmenter `num_workers` dans `config.yaml`

**Modèle ne converge pas** :
- Essayer `--mode knn` pour un baseline sans entraînement
- Ajuster `--knn-k` ou réduire le learning rate en linear probing

## 📚 Références

- **I-JEPA Paper** : [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243)
- **I-JEPA Model** : [facebook/ijepa_vith14_1k](https://huggingface.co/facebook/ijepa_vith14_1k)
- **Vision Transformer** : [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)

## 📄 Licence

Voir le fichier `LICENSE`

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou un pull request.

---

**Note** : Ce projet utilise le modèle I-JEPA pré-entraîné par Meta AI. Les poids sont automatiquement téléchargés depuis Hugging Face lors de la première exécution.
