# Améliorations pour latent_explorer.py

Basé sur l'analyse de l'espace latent effectuée dans `view_latent_3d.py` et `pca_latent_analysis.py`.

## Résumé de l'Analyse

Les projections latentes des 22 fémurs d'entraînement révèlent :

| Métrique | Valeur |
|----------|--------|
| **Score de planarité** | 95.15% |
| Variance PC1 (dans le plan) | 63.14% |
| Variance PC2 (dans le plan) | 32.01% |
| Variance PC3 (normale) | 4.85% |
| Vecteur normal au plan | `[0.8279, -0.2877, 0.4814]` |

**Conclusion** : Les données d'entraînement se distribuent sur un **plan 2D** dans l'espace latent 3D (les 3 premières dimensions). La variabilité réelle est capturée par seulement 2 directions principales.

---

## Améliorations Proposées

### 1. 🎚️ Mode "Exploration dans le Plan"

**Problème actuel** : Les 10 sliders explorent les axes z0-z9 indépendamment, mais ces axes ne correspondent pas aux directions de variabilité réelle des données.

**Amélioration** :
- Ajouter un **mode "Plan"** avec 2 sliders contrôlant le déplacement le long des axes principaux PC1 et PC2 du plan.
- Utiliser la matrice de rotation PCA pour transformer les coordonnées du plan vers l'espace latent complet.

```python
# Exemple de calcul
def get_plane_basis(latent_projections):
    """Retourne les 2 vecteurs de base du plan dans l'espace latent 10D."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca.fit(latent_projections)
    pc1 = pca.components_[0]  # Direction principale (63% variance)
    pc2 = pca.components_[1]  # Seconde direction (32% variance)
    return pc1, pc2

# Slider callback
def explore_in_plane(t1, t2, centroid, pc1, pc2):
    """Calcule le vecteur latent depuis les coordonnées du plan."""
    return centroid + t1 * pc1 + t2 * pc2
```

---

### 2. 📊 Affichage de la Position dans l'Espace Latent

**Amélioration** : Ajouter un indicateur visuel montrant où se situe la forme courante par rapport aux fémurs d'entraînement.

Options :
- Mini-plot 2D (PC1 vs PC2) avec les fémurs d'entraînement en points et la position courante en croix rouge
- Distance au plan comme indicateur de "réalisme" (proche = forme réaliste)
- Distance au centroïde comme indicateur d'"extrémité"

```python
# Calcul de la distance au plan
def distance_to_plane(latent_vector, centroid, normal):
    """Distance du point au plan de distribution des données."""
    return abs(np.dot(latent_vector - centroid, normal))
```

---

### 3. ⚠️ Avertissement de Sortie de Distribution

**Problème** : L'utilisateur peut explorer des zones où aucun fémur d'entraînement n'existe, générant des formes irréalistes.

**Amélioration** :
- Calculer la distance au plan et aux données d'entraînement
- Afficher un avertissement visuel (texte rouge) quand :
  - Distance au plan > seuil (ex: 2× la distance max des données)
  - Distance au centroïde > rayon de la distribution

```python
# Seuils basés sur l'analyse
MAX_DISTANCE_TO_PLANE = 0.20  # 0.1972 = max observé
WARNING_RADIUS = 2.0  # 2 écarts-types du centroïde

def check_distribution_bounds(latent_vector, centroid, normal, training_latents):
    dist_plane = distance_to_plane(latent_vector, centroid, normal)
    dist_centroid = np.linalg.norm(latent_vector[:3] - centroid)
    
    warnings = []
    if dist_plane > MAX_DISTANCE_TO_PLANE:
        warnings.append(f"⚠️ Hors plan (dist={dist_plane:.3f})")
    # etc.
    return warnings
```

---

### 4. 🔄 Bouton "Snap to Plane"

**Amélioration** : Ajouter un bouton pour projeter la position courante sur le plan de distribution.

```python
def snap_to_plane(latent_vector, centroid, normal):
    """Projette le vecteur latent sur le plan."""
    d = np.dot(latent_vector - centroid, normal)
    return latent_vector - d * normal
```

---

### 5. 📈 Chargement des Données d'Analyse au Démarrage

**Amélioration** : Charger automatiquement `latent_projections.npz` pour :
- Calculer le plan et ses caractéristiques
- Initialiser les bornes des sliders basées sur les données réelles
- Permettre la comparaison avec les fémurs d'entraînement

```python
def load_training_analysis():
    """Charge l'analyse PCA pré-calculée."""
    data_file = Path(__file__).parent / "latent_projection" / "latent_projections.npz"
    if data_file.exists():
        data = np.load(data_file, allow_pickle=True)
        return {
            'latents': data['latents'],
            'femur_names': data['femur_names'],
            'centroid': np.mean(data['latents'], axis=0),
            # PCA à calculer...
        }
    return None
```

---

### 6. 🎯 Plages des Sliders Basées sur les Données Réelles

**Problème actuel** : Plage fixe de ±5.0 autour de la baseline, arbitraire.

**Amélioration** : Calculer la plage réelle des données d'entraînement pour chaque dimension.

```python
def compute_slider_ranges(training_latents, margin=1.2):
    """Calcule les plages min/max pour chaque dimension."""
    ranges = []
    for i in range(training_latents.shape[1]):
        min_val = training_latents[:, i].min()
        max_val = training_latents[:, i].max()
        center = (min_val + max_val) / 2
        extent = (max_val - min_val) / 2 * margin
        ranges.append((center - extent, center + extent))
    return ranges
```

---

### 7. 🔀 Toggle entre Modes d'Exploration

**Amélioration** : Ajouter des boutons radio ou toggle pour choisir le mode :

1. **Mode "Raw z"** : 10 sliders pour z0-z9 (mode actuel)
2. **Mode "Plan 2D"** : 2 sliders pour PC1 et PC2 dans le plan
3. **Mode "Guided"** : Exploration contrainte à rester près du plan

---

### 8. 📍 Visualisation des Fémurs d'Entraînement Proches

**Amélioration** : Afficher les N fémurs d'entraînement les plus proches du point latent courant.

- Liste textuelle dans l'interface
- Optionnellement : afficher le mesh du fémur le plus proche en transparence comme référence

```python
def find_nearest_training_femurs(current_latent, training_latents, femur_names, n=3):
    """Trouve les n fémurs d'entraînement les plus proches."""
    distances = np.linalg.norm(training_latents - current_latent, axis=1)
    indices = np.argsort(distances)[:n]
    return [(femur_names[i], distances[i]) for i in indices]
```

---

### 9. 📐 Affichage des Statistiques en Temps Réel

**Amélioration** : Zone d'info dynamique affichant :

- Distance au centroïde
- Distance au plan
- Coordonnées dans le plan (PC1, PC2)
- Fémur le plus proche et sa distance
- Score de "réalisme" (inverse de la distance au plan)

---

### 10. 🔧 Refactoring Suggéré

**Structure de fichiers recommandée** :

```
visualization/
├── latent_explorer.py          # Interface principale (simplifiée)
├── latent_analysis.py          # Module d'analyse PCA/plan (nouveau)
├── latent_projection/
│   ├── latent_projections.npz  # Données pré-calculées
│   └── plane_params.npz        # Paramètres du plan (nouveau)
```

**Nouveau module `latent_analysis.py`** :

```python
class LatentSpaceAnalysis:
    """Analyse de l'espace latent basée sur les données d'entraînement."""
    
    def __init__(self, projections_path):
        self.load_projections(projections_path)
        self.compute_plane()
        self.compute_bounds()
    
    @property
    def centroid(self): ...
    
    @property  
    def plane_normal(self): ...
    
    @property
    def plane_basis(self): ...
    
    def project_to_plane(self, latent): ...
    
    def distance_to_plane(self, latent): ...
    
    def distance_to_distribution(self, latent): ...
    
    def get_slider_ranges(self): ...
    
    def find_nearest_femurs(self, latent, n=3): ...
```

---

## Priorité des Améliorations

| Priorité | Amélioration | Complexité | Impact |
|----------|-------------|------------|--------|
| 🔴 Haute | #5 Chargement des données d'analyse | Faible | Élevé |
| 🔴 Haute | #6 Plages sliders basées sur données | Faible | Élevé |
| 🟡 Moyenne | #1 Mode exploration dans le plan | Moyenne | Élevé |
| 🟡 Moyenne | #3 Avertissement hors distribution | Faible | Moyen |
| 🟢 Basse | #2 Affichage position dans l'espace | Moyenne | Moyen |
| 🟢 Basse | #4 Snap to plane | Faible | Faible |
| 🟢 Basse | #7 Toggle modes | Moyenne | Moyen |
| 🟢 Basse | #8 Visualisation fémurs proches | Élevée | Moyen |
| 🟢 Basse | #9 Stats temps réel | Faible | Faible |
| 🟢 Basse | #10 Refactoring | Moyenne | Long terme |

---

## Fichiers à Créer/Modifier

1. **Modifier** : `visualization/latent_explorer.py`
2. **Créer** : `visualization/latent_analysis.py` (module d'analyse)
3. **Créer** : `visualization/latent_projection/plane_params.npz` (paramètres du plan pré-calculés)

---

## Notes Techniques

- Le vecteur normal `[0.8279, -0.2877, 0.4814]` est calculé sur les 3 premières dimensions de l'espace latent après centrage et mise à l'échelle.
- Pour l'espace latent complet (10D), une PCA 10D donne les vraies directions principales.
- Le score de planarité de 95.15% suggère que les dimensions z3-z9 contiennent peu d'information discriminante pour les fémurs d'entraînement.
