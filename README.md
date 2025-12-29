# Femur_Modeling
Research Project in Artificial Intelligence for the Applied Mathematics Master 1 at UGA


# Règles
1. Ordre: Déclarer -> Définir -> Raporter
2. Commit au propre
   1. FEAT: Ajout d'une fonctionalité, classes...
   2. FIX: Lorsqu'un bug est corrigé
3. Commenter pour doxygèn 
   1. Devant fonctions et classes laisser un espace pour les commentaires
   2. Utilisation de *@brief* pour ce que ça fait et *@param* pour décrire les paramètres
   ```cpp
   /**
   * @brief Function saying Hello to a person of our choice
   *
   * @param name: str 
   */
   ```
4. Nommer les variables
   1. pas de tiret de 8 ( _ ) tout en minuscule avec des majuscule à chaque nouveau mots. *femurTriangle*.
   2. Pour les attributs d'une classe ils sont précédés de la lettre m_. *m_color*.
   3. Si constante, ou macro tout est en majuscue et donc utilisation du tiret du 8 ( _ ). *EARTH_GRAVITY*.

# TODO
1. Object vecteur et matrice avec les méthode apppropriés, posibilité d'utiliser la librairie eigen pour la gestion de mémoire (pas le droit des méthodes numérique)
2. Fonction d'activation et de coût (peut prendre des array)
3. Produit vectoriel et matriciel optimisé
4. Classe réseau de neurones avec les méthodes nécessaire
5. Méthode d'entrainement pour le réseau de neurone (déscente de gradient)
6. Parser nos fichiers, classe fémur 

# Avancées
- Object matrice2D et Vecteur fait avec produit matriciel. Lisez le code pour comprendre comment les utiliser (fichier linalg.hpp et linalg.cpp)
- Neural Network Function
- Neural Network les test ont été fait avec l'IA. Voir si on corrige ça.  

# Compilation

**1**
If new src files add them in the *CMakeList.txt* files in the command *add_executable*

**2**
```bash
cd build

# If CmakeLists.txt modified
cmake ..
make "[Name executable]"

#else
make "[Name executable]"
```

**3**
```bash
../bin/"[Name executable]"
```


