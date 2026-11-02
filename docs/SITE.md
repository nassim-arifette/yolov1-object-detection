# Site autonome et GitHub Pages

Depuis ce dépôt :

```bash
python export_site_assets.py
python tools/check_site.py
python -m http.server 8000 --directory site
```

Ouvrir http://localhost:8000, ou directement `site/index.html`. Les démonstrations
IoU, NMS et collisions de grille n'utilisent ni modèle, ni API, ni connexion
réseau. Les documents et figures sont copiés dans `site/`. Les figures anciennes
sont identifiées comme telles ; aucun mAP du pipeline corrigé n'est inventé.

Le dépôt possède son propre `.github/workflows/pages.yml`. Pour le déployer,
choisir **Settings → Pages → Build and deployment → GitHub Actions** dans ce
dépôt, puis exécuter **Publish research website** ou pousser une modification de
`site/` sur `main`. Le workflow vérifie les liens et publie uniquement ce dossier.
L'URL est fournie par le déploiement réussi ; la préparation locale n'est pas une
publication. Aucun dépôt parent ou voisin n'est nécessaire.

[Documentation officielle GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
