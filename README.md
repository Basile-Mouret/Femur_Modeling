# Femur_Modeling
Research Project in Artificial Intelligence for the Applied Mathematics Master 1 at UGA


# Règles
1. Ordre: Déclarer -> Définir -> Raporter
2. Commit au propre
   1. FEAT: Ajout d'une fonctionalité, classes...
   2. FIX: Lorsqu'un bug est corrigé
3. Commenter pour doxygèn 
   1. Devant fonction est classes faire un espace de commentaire
   2. Utilisation de *@brief* pour ce que ça fait et *@param* pour décrire les paramètres
   ```cpp
   /* 
   * @brief Function saying Hello to a person of our choice
   *
   * @param name: str 
   */
   ```
4. Nomer les variables
   1. pas de tiret de 8 ( _ ) tout en minuscule avec des majuscule à chaque nouveau mots. *femurTriangle*
   2. Pour les attributs d'une classe il sont précédé de la lettre m_. *m_color*.
   3. Si constante tout est en majuscue et donc utilisation du tiret du 8 ( _ )

# TODO
1. Object vecteur et matrice avec les méthode apppropriés, posibilité de eigen pour la gestion de mémoire (pas le droit des méthodes numérique)
2. Fonction d'activation et de coût (peut prendre des array)
3. Produit vectoriel et matriciel optimisé
4. Classe réseau de neurones avec les méthodes nécessaire
5. Parser nos fichier, classe fémur 


# Compilation

**1**
If new src files add them in the *CMakeList.txt* files in the command *add_executable*

**2**
```bash
cd build

# If CmakeLists.txt modified
cmake ..
make

#else
make
```

**3**
```bash
../bin/main
```


