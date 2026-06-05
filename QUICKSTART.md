# 🚀 Quick Start Guide - I-JEPA Fine-tuning

Guide de démarrage rapide pour fine-tuner I-JEPA sur vos datasets.

## ⚡ Installation (2 minutes)

```bash
# 1. Cloner le repository
git clone <url>
cd finetune-IJEPA

# 2. Installer les dépendances
pip install -r requirements.txt
```

## 🎯 Entraînement en Une Commande

### MNIST (Simple et Rapide)
```bash
python train_ijepa.py --dataset MNIST
```
- ✅ Dataset téléchargé automatiquement
- ✅ Modèle I-JEPA chargé depuis Hugging Face
- ✅ ~99% accuracy attendue

### CIFAR-10
```bash
python train_ijepa.py --dataset CIFAR10 --epochs 100
```
- ✅ 50,000 images d'entraînement
- ✅ ~85-92% accuracy attendue

### CIFAR-100
```bash
python train_ijepa.py --dataset CIFAR100 --epochs 150
```
- ✅ 100 classes
- ✅ ~70-80% accuracy attendue

## 🎓 Modes d'Entraînement

### 1. Full Fine-tuning (Recommandé)
Entraîne tout le modèle :
```bash
python train_ijepa.py --dataset MNIST --mode full
```

### 2. Linear Probing (Rapide)
Entraîne seulement la tête de classification :
```bash
python train_ijepa.py --dataset MNIST --mode linear
```
- ⚡ 3-5x plus rapide
- 💾 Moins de mémoire GPU requise
- 📊 ~2-5% de moins en accuracy

### 3. Progressive (Stratégie avancée)
Commence gelé, puis dégèle après 10 epochs :
```bash
python train_ijepa.py --dataset CIFAR10 --mode progressive
```

## ⚙️ Options Personnalisées

```bash
python train_ijepa.py \
  --dataset CIFAR10 \
  --mode full \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-4
```

Options disponibles :
- `--dataset` : MNIST, CIFAR10, CIFAR100
- `--mode` : full, linear, progressive
- `--epochs` : Nombre d'époques
- `--batch-size` : Taille du batch (défaut: 64)
- `--lr` : Learning rate (défaut: 2e-4)
- `--config` : Fichier de config personnalisé

## 📊 Monitoring en Temps Réel

Dans un autre terminal :
```bash
tensorboard --logdir logs/
```
Puis ouvrir http://localhost:6006

## 💾 Résultats

Les checkpoints sont sauvegardés dans :
- `checkpoints/ijepa_mnist/best_model.pth` (meilleur modèle)
- `checkpoints/ijepa_cifar10/best_model.pth`
- `checkpoints/ijepa_cifar100/best_model.pth`

## 🎯 Configuration Avancée

Modifiez `config.yaml` pour un contrôle total :

```yaml
dataset:
  name: 'MNIST'
  batch_size: 64

model:
  pretrained_name: 'ijepa-vith14-1k'

training:
  epochs: 50
  base_lr: 2e-4

finetuning:
  freeze_encoder: False
```

## 🔧 Troubleshooting

### Problème : Out of Memory
**Solution** :
```bash
python train_ijepa.py --batch-size 32  # ou 16
```

### Problème : Entraînement lent
**Solution** :
- Vérifier GPU : `nvidia-smi`
- Vérifier CUDA : `python -c "import torch; print(torch.cuda.is_available())"`
- Essayer mode linear : `--mode linear`

### Problème : Accuracy faible
**Solution** :
```bash
# 1. Essayer linear probe d'abord
python train_ijepa.py --dataset MNIST --mode linear --epochs 20

# 2. Puis full fine-tuning
python train_ijepa.py --dataset MNIST --mode full --epochs 50
```

## 📚 Prochaines Étapes

1. ✅ Lancer un premier entraînement sur MNIST
2. ✅ Visualiser avec TensorBoard
3. ✅ Tester différents modes (full/linear/progressive)
4. ✅ Essayer CIFAR-10 et CIFAR-100

## 🎓 Ressources

- [I-JEPA Paper](https://arxiv.org/abs/2301.08243)
- [Model Hugging Face](https://huggingface.co/facebook/ijepa_vith14_1k)
- [Documentation complète](README.md)

---

**Besoin d'aide ?** Consultez le [README.md](README.md) complet ou ouvrez une issue.
