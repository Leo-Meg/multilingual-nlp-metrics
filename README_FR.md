# NLP multilingue, ce que mesure vraiment une perplexité

Modèles n-grammes et perplexité, écrits sans dépendance, pour répondre à une question restée sans réponse dans mon rendu de M2.

**Léo Mégret**, Master Linguistique Informatique, Université Paris Cité

> **État du dépôt, version 2.** Je mène ce travail par étapes, chacune dans son
> propre dossier. Je publie au fur et à mesure plutôt qu'une fois tout terminé.

---

## Pourquoi ce dépôt

Le TP qui ouvrait mon cours de NLP multilingue demandait d'entraîner un modèle de
langue par langue et de comparer les perplexités. La question centrale de
l'énoncé était de savoir si des perplexités se comparent entre langues.

J'avais répondu non, sans pouvoir le démontrer. Ce dépôt est la démonstration.

L'énoncé demandait 41 langues, 40 000 phrases d'entraînement chacune et
l'installation de KenLM. Je fais l'inverse, cinq langues, un corpus parallèle de
40 phrases écrit à la main, et tout dans le code. Le résultat tourne sur un
ordinateur portable et reste exécutable dans dix ans.

---

## Les versions publiées

| | Dossier | Contenu | Tests |
|---|---|---|---:|
| **1** | `1.multilingue_python_projet` | Modèles de langue et perplexité | 18 |
| **2** | `2.multilingue_python_projet` | BLEU, implémenter la métrique puis la casser | 16 |

Soit **34 tests** au total. Chaque dossier contient tout le contenu du
précédent, plus une étape.

---

## Lancer la dernière version

```bash
cd 2.multilingue_python_projet
python -m src.bleu
python -m tests.test_bleu
```

---

## Ce qui reste ouvert

Mes mesures portent sur des traductions déjà produites. La façon dont le texte
est fabriqué à partir des probabilités du modèle reste hors de mon champ.

Je soupçonne que ce choix pèse autant que le modèle lui-même, et je n'ai rien
pour le vérifier.

---

## Comment je travaille

Quatre règles que je me suis données en commençant, et que je compte tenir sur
tout le dépôt.

**Rien à télécharger.** Le corpus est écrit dans le code. Mes notebooks de master
commençaient tous par un `wget` vers un serveur universitaire ou un montage de
Google Drive. Deux ans plus tard, la moitié ne s'exécutent plus.

**Rien n'est affirmé sans mesure.** Chaque chiffre de ce fichier correspond à une
commande qu'on peut relancer.

**Les erreurs de mes rendus sont citées, pas effacées.** Quand un résultat que
j'avais rendu en cours était faux ou incomplet, je le dis et je donne le résultat
correct.

**Les résultats négatifs restent.** Quand une expérience montre l'inverse de ce
que j'attendais, j'écris ce que j'ai trouvé.

**Le code est commenté en français.**

---

---

*Version anglaise, [README.md](README.md).*
