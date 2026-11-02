# Reprendre YOLO et retrouver les sauvegardes

Le calcul et la sauvegarde s'exécutent directement sur la VM Colab `cv-yolo`.
Ils ne dépendent ni d'une réponse de l'agent, ni de son quota de tokens, ni d'un
terminal Windows restant ouvert. Le calcul lancé le 1 novembre 2026 poursuit
100 époques sur T4 avec le code r02 figé et le protocole VOC 2007 enregistré.

## Où regarder

Dans Google Drive :
`MyDrive/Projets-CV-MVA/YOLO/2026-11-01_iou_aug_seed123/`.

- `acceptance_02/` : copie initiale immuable des deux checkpoints à l'époque 2,
  source r02, archives de preuves, SHA256 et métadonnées.
- `autonomous_watch_state.json` : `active_slot` désigne le dernier dossier
  entièrement copié et vérifié. Ne choisissez pas un dossier d'après sa date.
- `checkpoint_slot_a/` et `checkpoint_slot_b/` : sauvegardes alternées. Le slot
  précédent reste utilisable pendant la copie suivante.
- `checkpoint_receipts/` : reçus historiques avec taille et SHA256 des fichiers.
- `BILAN.json` : état et métriques de la dernière sauvegarde vérifiée.
- `FINAL.json` : créé seulement après copie finale vérifiée. `backup_verified`
  confirme la sauvegarde ; `training_status` distingue réussite, interruption et
  disparition brutale du processus. `experiment_complete` indique si les 100
  époques sont terminées. L'évaluation finale sur test reste une étape séparée.

Chaque snapshot contient `last.pt`, `best.pt`, l'optimiseur SGD, le scaler AMP,
les états RNG Python/NumPy/Torch/CUDA/DataLoader, configuration, époque et historique
nécessaires à la reprise du planning de taux d'apprentissage, manifests de données,
environnement logiciel, journaux, source r02 et bilan. `snapshot.json` permet de
vérifier les empreintes. Si un arrêt survient entre deux écritures de checkpoints,
le bilan décrit précisément quel checkpoint complet a été retenu pour la reprise.

## Fonctionnement autonome

Un gardien distant relance uniquement le processus de sauvegarde s'il plante.
Il ne redémarre et n'arrête jamais l'entraînement. Le watcher contrôle l'identité
du vrai processus d'entraînement et détecte aussi une disparition sans mise à jour
de `run.json`. Il prend un snapshot de contrôle au démarrage, puis à chaque tranche
de dix époques et systématiquement à la fin ou à une interruption, même sans
nouvelle époque. Les copies Drive sont isolées dans un sous-processus avec timeout ;
les erreurs transitoires sont retentées tant que la VM existe. Un reçu final n'est
publié qu'après vérification SHA256 de tous les fichiers du snapshot.

**Limite physique :** si Colab détruit la VM, ou si l'accès Drive reste indisponible
jusqu'à sa destruction, aucun processus ne peut encore transférer ses fichiers.
Les sauvegardes déjà vérifiées sur Drive restent disponibles. Une interruption
au milieu d'une époque perd les calculs depuis le dernier checkpoint complet.
Un arrêt simultané du gardien et du watcher peut également interrompre la protection.

## Vérifier le calcul depuis PowerShell

```powershell
colab exec -s cv-yolo --file /mnt/c/Users/nassi/Desktop/Projets-CV-MVA/01-YOLOv1-object-detection/colab_artifacts/remote_status.py
```

Sur la VM, les journaux et reçus sont dans `/content/yolo_outputs/` :
`continuation_100.json`, `continuation_100.log`, `autonomous_watch_v2_launch.json`,
`autonomous_guard_state.json`, `autonomous_watch_state.json`, `autonomous_watch.log`.
Les PID courants sont dans ces reçus ; ne réutilisez pas un ancien PID sans vérifier
son identité. Les scripts du gardien et du watcher sont conservés sur Drive dans
`automation_v2/` avec leurs empreintes.

## Restaurer après remplacement de la VM

1. Créer une nouvelle session GPU, monter Drive, puis lire `FINAL.json` s'il existe
   et `autonomous_watch_state.json` pour choisir le slot vérifié. Si aucune copie
   périodique n'est disponible, utiliser `acceptance_02/`.
2. Vérifier les tailles et SHA256 contre `snapshot.json` (ou `verified.json` pour
   la copie initiale). Restaurer **ensemble** tous les fichiers du slot dans
   `/content/yolo_outputs/iou_aug_seed123/`.
3. Extraire `source_r02.tar.gz` du slot, ou `yolov1_source_r02.tar.gz` de la copie
   initiale, dans `/content/yolov1_project_r02/`. Installer les dépendances et
   relancer `download_voc.py --data-dir /content/yolo_data` : MD5 et SHA des archives
   sont vérifiés ; les manifests de train et validation doivent correspondre aux
   empreintes enregistrées dans les checkpoints.
4. Reprendre depuis le dernier checkpoint complet avec le code et la configuration
   identiques. Le programme refuse une divergence de source/configuration/manifests.

```sh
cd /content/yolov1_project_r02
python train.py --voc-root /content/yolo_data/VOCdevkit \
  --train-manifest /content/yolo_data/manifests/train.json \
  --val-manifest /content/yolo_data/manifests/val.json \
  --out /content/yolo_outputs/iou_aug_seed123 \
  --resume /content/yolo_outputs/iou_aug_seed123/last.pt
```

La reprise d'une nouvelle VM exige un nouveau reçu d'identité du processus et un
nouveau gardien : les anciens PID ne sont pas réutilisables. Conserver les dossiers
de sauvegarde précédents et écrire dans un nouveau dossier de reprise sur Drive.
Ne modifier ni le code r02 ni la configuration pendant la reprise d'un même run.
