# I-JEPA Fine-tuning sur MNIST, CIFAR-10 et CIFAR-100

Fine-tuning du modèle **I-JEPA** (Image-based Joint-Embedding Predictive Architecture) de Meta/Facebook sur des datasets de classification d'images.

## 🎯 Description

Ce projet utilise le modèle **I-JEPA ViT-H/14** pré-entraîné disponible sur Hugging Face (`facebook/ijepa_vith14_1k`) pour faire du fine-tuning supervisé sur :
- **MNIST** : Chiffres manuscrits (10 classes)
- **CIFAR-10** : Images naturelles (10 classes)  
- **CIFAR-100** : Images naturelles (100 classes)

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

### Entraînement Simple

```bash
# MNIST (par défaut)
python train_ijepa.py --dataset MNIST

# CIFAR-10
python train_ijepa.py --dataset CIFAR10

# CIFAR-100
python train_ijepa.py --dataset CIFAR100
```

### Modes d'Entraînement

**Full Fine-tuning** (tout le modèle est entraîné) :
```bash
python train_ijepa.py --dataset MNIST --mode full
```

**Linear Probing** (seule la tête de classification est entraînée) :
```bash
python train_ijepa.py --dataset MNIST --mode linear
```

**Progressive Fine-tuning** (commence gelé, dégèle après 10 epochs) :
```bash
python train_ijepa.py --dataset CIFAR10 --mode progressive
```

### Options Avancées

```bash
python train_ijepa.py \
  --dataset CIFAR10 \
  --mode full \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4
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
│   ├── data_loader.py        # Datasets MNIST/CIFAR
│   └── train.py              # Trainer pour fine-tuning
├── train_ijepa.py            # Script principal
├── config.yaml               # Configuration par défaut
├── requirements.txt          # Dépendances
└── README.md                 # Ce fichier
```

## 🔍 Comment Ça Marche ?

1. **Chargement du Modèle** : I-JEPA ViT-H/14 est téléchargé depuis Hugging Face
2. **Ajout d'une Tête de Classification** : Une couche linéaire pour la prédiction de classes
3. **Fine-tuning** : Entraînement supervisé avec Cross-Entropy Loss
4. **Évaluation** : Accuracy sur test set

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

Les modèles sont sauvegardés dans `checkpoints/` :
- `best_model.pth` : Meilleur modèle (meilleure accuracy)
- `final_model.pth` : Modèle final
- `checkpoint_epoch_X.pth` : Checkpoints périodiques

## ⚙️ Configuration par Défaut

```yaml
dataset:
  name: 'MNIST'
  batch_size: 64
  image_size: 224

model:
  pretrained_name: 'ijepa-vith14-1k'  # facebook/ijepa_vith14_1k

training:
  epochs: 50
  base_lr: 2e-4
  
finetuning:
  freeze_encoder: False  # Full fine-tuning
```

## 🎯 Performances Attendues

Les datasets sont téléchargés automatiquement lors du premier lancement.

### MNIST (50 epochs)
- **Full fine-tuning** : ~99%+ accuracy
- **Linear probe** : ~98%+ accuracy

### CIFAR-10 (50-100 epochs)  
- **Full fine-tuning** : ~85-92% accuracy
- **Linear probe** : ~75-85% accuracy

### CIFAR-100 (100+ epochs)
- **Full fine-tuning** : ~70-80% accuracy
- **Linear probe** : ~55-65% accuracy

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
- Essayer le mode `--mode linear` d'abord
- Réduire le learning rate : `--lr 1e-4`

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
