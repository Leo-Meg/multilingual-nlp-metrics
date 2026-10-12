"""
Corpus parallèle multilingue embarqué, et outils de segmentation.

Pourquoi un corpus parallèle
-----------------------------
C'était l'enjeu méthodologique central du TP n°1 de NLP multilingue
(« One model to rule them all », Guillaume Wisniewski, M2). L'énoncé demandait
d'extraire, pour chacune des 41 langues de wiki40b, exactement 40 000 phrases
d'entraînement et 3 000 de test. La question 2.1.11 était : *pourquoi imposer
que tous les jeux aient la même taille ?*

Réponse : pour que **la langue soit la seule variable** de l'expérience. Si les
corpus diffèrent en taille ou en contenu, on mesure la différence de corpus et
non celle des langues.

Je vais plus loin ici : mon corpus est **parallèle** (les mêmes phrases traduites
dans chaque langue), ce qui neutralise aussi la variable « contenu ». C'est la
seule façon honnête de comparer une perplexité entre langues — et l'occasion de
rappeler que même comme ça, la comparaison reste discutable (voir
`ngrammes.py`).

Pourquoi j'embarque tout dans le code
--------------------------------------
Mes notebooks de M2 commençaient par `tfds.load("wiki40b/fr", data_dir="gs://…")`
et un montage de Google Drive. Deux ans plus tard, plus rien ne s'exécute. Ici,
tout est dans le fichier : le dépôt tournera encore dans dix ans.
"""

from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
# Le corpus parallèle
# --------------------------------------------------------------------------- #

# 40 phrases, traduites dans cinq langues de types morphologiques différents.
# J'ai choisi des phrases de registre courant, avec des structures variées
# (déclaratives, coordonnées, subordonnées, négations) pour que les modèles de
# langue aient de quoi apprendre autre chose que des collocations figées.
CORPUS_PARALLELE: dict[str, list[str]] = {
    "francais": [
        "le petit chat dort dans le jardin",
        "les étudiants travaillent sur un corpus annoté",
        "nous devons évaluer le système avant la fin du mois",
        "le modèle apprend une représentation des mots",
        "la technologie du langage ne traite pas toutes les langues également",
        "le professeur explique la structure de la phrase",
        "chaque mot reçoit une étiquette grammaticale",
        "les données sont réparties en trois ensembles distincts",
        "je pense que cette méthode donne de meilleurs résultats",
        "il fait beau ce matin dans le quartier",
        "elle a écrit un long article sur la sémantique",
        "nous avons lu tous les articles de ce numéro",
        "le chien court vers la maison quand il pleut",
        "les enfants jouent dans la cour de l'école",
        "cette question reste ouverte pour le moment",
        "le train arrive en gare avec un peu de retard",
        "la ville a construit une nouvelle bibliothèque",
        "personne ne sait comment ce système fonctionne vraiment",
        "les chercheurs publient leurs résultats chaque année",
        "il faut comparer les deux méthodes sur le même corpus",
        "le texte contient beaucoup de mots rares",
        "nous ne pouvons pas comparer ces deux ensembles directement",
        "la phrase est correcte mais elle reste ambiguë",
        "les langues du monde présentent une grande diversité",
        "ce livre explique très bien les concepts de base",
        "le système produit une réponse en quelques secondes",
        "les résultats confirment notre hypothèse de départ",
        "elle travaille sur ce projet depuis deux ans",
        "le programme calcule la probabilité de chaque séquence",
        "nous observons une nette amélioration des performances",
        "les mots rares posent toujours des difficultés",
        "il a expliqué le problème avec beaucoup de clarté",
        "la réunion commence à neuf heures du matin",
        "ce modèle nécessite énormément de données annotées",
        "les linguistes étudient la structure des langues naturelles",
        "le corpus contient mille phrases annotées manuellement",
        "nous devons choisir entre deux approches différentes",
        "cette langue possède une morphologie très riche",
        "les résultats varient beaucoup selon la langue considérée",
        "il est difficile de comparer des corpus de tailles différentes",
    ],
    "anglais": [
        "the small cat sleeps in the garden",
        "the students work on an annotated corpus",
        "we must evaluate the system before the end of the month",
        "the model learns a representation of the words",
        "language technology does not treat all languages equally",
        "the teacher explains the structure of the sentence",
        "each word receives a grammatical label",
        "the data is split into three distinct sets",
        "i think that this method gives better results",
        "the weather is nice this morning in the neighbourhood",
        "she wrote a long article about semantics",
        "we have read all the articles in this issue",
        "the dog runs towards the house when it rains",
        "the children play in the school yard",
        "this question remains open for the moment",
        "the train arrives at the station a little late",
        "the city has built a new library",
        "nobody knows how this system really works",
        "the researchers publish their results every year",
        "we must compare the two methods on the same corpus",
        "the text contains many rare words",
        "we cannot compare these two sets directly",
        "the sentence is correct but it remains ambiguous",
        "the languages of the world show a great diversity",
        "this book explains the basic concepts very well",
        "the system produces an answer in a few seconds",
        "the results confirm our initial hypothesis",
        "she has been working on this project for two years",
        "the program computes the probability of each sequence",
        "we observe a clear improvement in performance",
        "rare words always cause difficulties",
        "he explained the problem with great clarity",
        "the meeting starts at nine in the morning",
        "this model requires a huge amount of annotated data",
        "linguists study the structure of natural languages",
        "the corpus contains a thousand manually annotated sentences",
        "we must choose between two different approaches",
        "this language has a very rich morphology",
        "the results vary a lot depending on the language considered",
        "it is difficult to compare corpora of different sizes",
    ],
    "espagnol": [
        "el pequeño gato duerme en el jardín",
        "los estudiantes trabajan en un corpus anotado",
        "debemos evaluar el sistema antes del fin de mes",
        "el modelo aprende una representación de las palabras",
        "la tecnología del lenguaje no trata todas las lenguas igualmente",
        "el profesor explica la estructura de la frase",
        "cada palabra recibe una etiqueta gramatical",
        "los datos se reparten en tres conjuntos distintos",
        "pienso que este método da mejores resultados",
        "hace buen tiempo esta mañana en el barrio",
        "ella escribió un largo artículo sobre semántica",
        "hemos leído todos los artículos de este número",
        "el perro corre hacia la casa cuando llueve",
        "los niños juegan en el patio de la escuela",
        "esta cuestión sigue abierta por el momento",
        "el tren llega a la estación con un poco de retraso",
        "la ciudad ha construido una nueva biblioteca",
        "nadie sabe cómo funciona realmente este sistema",
        "los investigadores publican sus resultados cada año",
        "debemos comparar los dos métodos en el mismo corpus",
        "el texto contiene muchas palabras raras",
        "no podemos comparar estos dos conjuntos directamente",
        "la frase es correcta pero sigue siendo ambigua",
        "las lenguas del mundo presentan una gran diversidad",
        "este libro explica muy bien los conceptos básicos",
        "el sistema produce una respuesta en pocos segundos",
        "los resultados confirman nuestra hipótesis inicial",
        "ella trabaja en este proyecto desde hace dos años",
        "el programa calcula la probabilidad de cada secuencia",
        "observamos una clara mejora del rendimiento",
        "las palabras raras siempre causan dificultades",
        "él explicó el problema con mucha claridad",
        "la reunión empieza a las nueve de la mañana",
        "este modelo necesita muchísimos datos anotados",
        "los lingüistas estudian la estructura de las lenguas naturales",
        "el corpus contiene mil frases anotadas manualmente",
        "debemos elegir entre dos enfoques diferentes",
        "esta lengua posee una morfología muy rica",
        "los resultados varían mucho según la lengua considerada",
        "es difícil comparar corpus de tamaños diferentes",
    ],
    "allemand": [
        "die kleine katze schläft im garten",
        "die studenten arbeiten an einem annotierten korpus",
        "wir müssen das system vor dem monatsende evaluieren",
        "das modell lernt eine repräsentation der wörter",
        "die sprachtechnologie behandelt nicht alle sprachen gleich",
        "der lehrer erklärt die struktur des satzes",
        "jedes wort erhält eine grammatische bezeichnung",
        "die daten werden in drei verschiedene mengen aufgeteilt",
        "ich denke dass diese methode bessere ergebnisse liefert",
        "das wetter ist heute morgen schön im viertel",
        "sie hat einen langen artikel über semantik geschrieben",
        "wir haben alle artikel dieser ausgabe gelesen",
        "der hund läuft zum haus wenn es regnet",
        "die kinder spielen auf dem schulhof",
        "diese frage bleibt vorerst offen",
        "der zug kommt mit etwas verspätung im bahnhof an",
        "die stadt hat eine neue bibliothek gebaut",
        "niemand weiß wie dieses system wirklich funktioniert",
        "die forscher veröffentlichen ihre ergebnisse jedes jahr",
        "wir müssen die beiden methoden am selben korpus vergleichen",
        "der text enthält viele seltene wörter",
        "wir können diese beiden mengen nicht direkt vergleichen",
        "der satz ist korrekt aber er bleibt mehrdeutig",
        "die sprachen der welt zeigen eine große vielfalt",
        "dieses buch erklärt die grundbegriffe sehr gut",
        "das system liefert eine antwort in wenigen sekunden",
        "die ergebnisse bestätigen unsere anfangshypothese",
        "sie arbeitet seit zwei jahren an diesem projekt",
        "das programm berechnet die wahrscheinlichkeit jeder sequenz",
        "wir beobachten eine deutliche verbesserung der leistung",
        "seltene wörter verursachen immer schwierigkeiten",
        "er hat das problem mit großer klarheit erklärt",
        "die sitzung beginnt um neun uhr morgens",
        "dieses modell benötigt sehr viele annotierte daten",
        "linguisten untersuchen die struktur natürlicher sprachen",
        "das korpus enthält tausend manuell annotierte sätze",
        "wir müssen zwischen zwei verschiedenen ansätzen wählen",
        "diese sprache besitzt eine sehr reiche morphologie",
        "die ergebnisse variieren stark je nach betrachteter sprache",
        "es ist schwierig korpora unterschiedlicher größe zu vergleichen",
    ],
    "turc": [
        "küçük kedi bahçede uyuyor",
        "öğrenciler açıklamalı bir derlem üzerinde çalışıyorlar",
        "sistemi ay sonundan önce değerlendirmeliyiz",
        "model kelimelerin bir temsilini öğreniyor",
        "dil teknolojisi bütün dilleri eşit biçimde işlemiyor",
        "öğretmen cümlenin yapısını açıklıyor",
        "her kelime dilbilgisel bir etiket alıyor",
        "veriler üç ayrı kümeye bölünüyor",
        "bu yöntemin daha iyi sonuçlar verdiğini düşünüyorum",
        "bu sabah mahallede hava güzel",
        "anlambilim üzerine uzun bir makale yazdı",
        "bu sayıdaki bütün makaleleri okuduk",
        "yağmur yağdığında köpek eve doğru koşuyor",
        "çocuklar okul bahçesinde oynuyorlar",
        "bu soru şimdilik açık kalıyor",
        "tren istasyona biraz gecikmeyle geliyor",
        "şehir yeni bir kütüphane inşa etti",
        "bu sistemin gerçekten nasıl çalıştığını kimse bilmiyor",
        "araştırmacılar sonuçlarını her yıl yayımlıyorlar",
        "iki yöntemi aynı derlem üzerinde karşılaştırmalıyız",
        "metin çok sayıda nadir kelime içeriyor",
        "bu iki kümeyi doğrudan karşılaştıramayız",
        "cümle doğru ama belirsiz kalıyor",
        "dünyanın dilleri büyük bir çeşitlilik gösteriyor",
        "bu kitap temel kavramları çok iyi açıklıyor",
        "sistem birkaç saniyede bir yanıt üretiyor",
        "sonuçlar başlangıç hipotezimizi doğruluyor",
        "iki yıldır bu proje üzerinde çalışıyor",
        "program her dizinin olasılığını hesaplıyor",
        "performansta belirgin bir iyileşme gözlemliyoruz",
        "nadir kelimeler her zaman zorluk çıkarıyor",
        "sorunu büyük bir açıklıkla anlattı",
        "toplantı sabah dokuzda başlıyor",
        "bu model çok fazla açıklamalı veri gerektiriyor",
        "dilbilimciler doğal dillerin yapısını inceliyorlar",
        "derlem elle açıklanmış bin cümle içeriyor",
        "iki farklı yaklaşım arasında seçim yapmalıyız",
        "bu dil çok zengin bir morfolojiye sahip",
        "sonuçlar ele alınan dile göre çok değişiyor",
        "farklı boyutlardaki derlemleri karşılaştırmak zordur",
    ],
}

# Classification typologique (source : WALS, comme dans le TP de M2).
TYPES_MORPHOLOGIQUES = {
    "anglais": "flexionnelle (pauvre)",
    "espagnol": "flexionnelle",
    "francais": "flexionnelle",
    "allemand": "flexionnelle (+ composition)",
    "turc": "agglutinante",
}

DEBUT = "<d>"      # début de phrase
FIN = "<f>"        # fin de phrase
INCONNU = "<unk>"  # mot hors-vocabulaire


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #


def segmenter_mots(texte: str) -> list[str]:
    """Segmentation en mots, avec isolement de la ponctuation.

    En M2 nous utilisions `polyglot`, qui s'appuie sur l'algorithme Unicode Text
    Segmentation d'ICU (question 2.1.12 du TP : « quelle information polyglot
    utilise-t-il pour identifier les frontières ? »). Sa vraie force est d'être
    **sensible à la langue** : il applique des règles différentes selon l'écriture
    et la langue détectée.

    Ma version est délibérément simple et **identique pour toutes les langues**.
    C'est un choix : je veux que la seule différence entre mes mesures vienne
    des langues elles-mêmes, pas de segmenteurs différents. Cela a un coût — les
    clitiques du français (« l'article ») et les contractions de l'allemand
    (« im » = « in dem ») ne sont pas traités — et je le documente plutôt que de
    le masquer.
    """
    texte = re.sub(r"([.,!?;:()\"«»…])", r" \1 ", texte)
    return texte.lower().split()


def segmenter_caracteres(texte: str) -> list[str]:
    """Segmentation en caractères, utile pour les langues sans espaces.

    Je normalise en NFC pour que « é » composé (e + accent combinant) et « é »
    précomposé comptent comme un seul et même caractère. Sans cela, deux textes
    visuellement identiques donnent des vocabulaires différents — un piège
    classique dès qu'on travaille sur plusieurs écritures.
    """
    return list(unicodedata.normalize("NFC", texte.lower()))


def decouper_train_test(
    phrases: list[str], proportion_train: float = 0.75, graine: int = 0
) -> tuple[list[str], list[str]]:
    """Découpage reproductible, avec mélange déterministe.

    J'utilise un générateur explicite plutôt que `random` global : sinon, deux
    appels dans le même processus ne donnent pas le même découpage, et les
    résultats deviennent impossibles à reproduire.
    """
    import random

    rng = random.Random(graine)
    melangees = list(phrases)
    rng.shuffle(melangees)
    coupe = int(proportion_train * len(melangees))
    return melangees[:coupe], melangees[coupe:]


def type_token_ratio(phrases: list[str]) -> float:
    """TTR = formes distinctes / tokens.

    Question 1.1.1 du TP : indicateur de la **diversité lexicale**, et par
    ricochet de la complexité morphologique — une langue riche en flexions
    produit beaucoup de formes pour un même lemme.

    Sa limite (que l'énoncé demandait aussi de discuter) : le TTR **décroît
    mécaniquement** avec la longueur du texte. Il n'est comparable qu'à taille de
    corpus strictement égale, ce que mon corpus parallèle garantit.
    """
    tokens = [m for p in phrases for m in segmenter_mots(p)]
    return len(set(tokens)) / max(len(tokens), 1)


def statistiques(phrases: list[str]) -> dict[str, float]:
    """Profil descriptif d'un corpus."""
    tokens = [m for p in phrases for m in segmenter_mots(p)]
    caracteres = [c for p in phrases for c in p if not c.isspace()]
    return {
        "nb_phrases": float(len(phrases)),
        "nb_tokens": float(len(tokens)),
        "nb_types": float(len(set(tokens))),
        "ttr": len(set(tokens)) / max(len(tokens), 1),
        "tokens_par_phrase": len(tokens) / max(len(phrases), 1),
        "caracteres_par_token": len(caracteres) / max(len(tokens), 1),
        "hapax": float(sum(
            1 for t in set(tokens) if tokens.count(t) == 1
        )) / max(len(set(tokens)), 1),
    }


if __name__ == "__main__":
    print("=== Corpus parallèle multilingue ===\n")

    print(f"{len(CORPUS_PARALLELE)} langues × "
          f"{len(CORPUS_PARALLELE['francais'])} phrases, strictement parallèles.\n")

    entete = (f"  {'langue':>10} | {'type':>28} | {'tokens':>7} | {'types':>6} | "
              f"{'TTR':>6} | {'car/tok':>8} | {'hapax':>6}")
    print(entete)
    print("  " + "-" * (len(entete) - 2))

    for langue in sorted(CORPUS_PARALLELE):
        s = statistiques(CORPUS_PARALLELE[langue])
        print(f"  {langue:>10} | {TYPES_MORPHOLOGIQUES[langue]:>28} | "
              f"{s['nb_tokens']:>7.0f} | {s['nb_types']:>6.0f} | {s['ttr']:>6.3f} | "
              f"{s['caracteres_par_token']:>8.2f} | {s['hapax']:>6.3f}")

    print(
        "\nComment lire ce tableau\n"
        "-----------------------\n"
        "* Le corpus est PARALLÈLE : les cinq versions disent exactement la même\n"
        "  chose. Tout écart vient donc de la langue, pas du contenu. C'est la\n"
        "  seule façon honnête de comparer des langues, et c'est ce que le TP de\n"
        "  M2 imposait en exigeant des jeux de taille identique (question 2.1.11).\n"
        "\n"
        "* Le TTR et le nombre de caractères par token croissent avec la richesse\n"
        "  morphologique : l'anglais en bas, le turc agglutinant en haut. Le turc\n"
        "  dit la même chose en MOINS de tokens (les suffixes remplacent les mots\n"
        "  grammaticaux) mais avec des tokens PLUS LONGS et plus variés.\n"
        "\n"
        "* La proportion d'hapax (formes vues une seule fois) est le signal le\n"
        "  plus direct : plus elle est élevée, plus un modèle de langue aura de\n"
        "  mal à estimer des probabilités fiables. C'est exactement ce que je\n"
        "  mesure au module suivant."
    )
