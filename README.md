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
1. Produit vectoriel et matriciel optimisé
2. Docu Doxygen
3. Voir resultats RDN avec differentes fonction d'activation et d'erreur

# Ideas
- voir si on peut utiliser multithreading (tâches), au moins utiliser avx, voire au mieux écrire un noyau cuda
- enregistrement plus simple du NN (hdf5 ou format hugging face) 
- avoir un serveur pour entrainer le réseau
- le RDN sort la déformation par rapport au fémur moyen plutôt que les positions absolues des points du fémur (convergence et entrainement plus rapide, plus précis)

# Remarks
- In linalg.hpp, in class VEctor, nhadmaard return 0 if size don't match, and *this for overload+ et -
- Modifer valeur d'un élement d'un vecteur ou matrice avec setCoeff ou &() et modifier la reference
- a quoi sert le destructor du reseau de neurones si ya rien dedans ?

# Avancées
- Object matrice2D et Vecteur fait avec produit matriciel. Lisez le code pour comprendre comment les utiliser (fichier linalg.hpp et linalg.cpp)
- Neural Network Function
- Neural Network les test ont été fait avec l'IA. Voir si on corrige ça.
- Visu 3D du Fémur

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
Use the flag `-DCMAKE_BUILD_TYPE=Release` for better runtime

**3**
```bash
../bin/"[Name executable]"
```
