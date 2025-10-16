
# Masterarbeit - TITEL

Dieses Repository enthält Code und Daten für die Masterarbeit "TITEL".

## Ordnerstruktur

- `blenderproc/` — BlenderProc-Skripte, Assets etc. für Generierung der synthetischen Trainingsdaten (siehe `blenderproc/README.md` für Details).
- `dataset/` — Datensatz für das Training und die Evaluation des Objektdetektors.
- `notebooks/` — Jupyter Notebooks für Einsatz von Domain Adaptation sowie Training und Evaluation des Objektdetektors. Sind für Kaggle intendiert.
- `misc/` — Hilfsskripte zur Ausführung des Signifikanztests sowie dem Sampling der Testdaten aus HANDAL und METU-ALET.

## Voraussetzungen

1. Domain Adaptation, Training und Evaluation des Objektdetektors: Jupyter Notebook/Lab oder über Cloud-Plattformen, wie etwa Kaggle/Colab
    - Die Notebooks sind auch unter [Master: Object Detection](https://www.kaggle.com/code/diminini/master-object-detection) und [Master: Domain Adaptation](https://www.kaggle.com/code/diminini/master-domain-adaptation) auf Kaggle zu finden und können dort direkt ausgeführt werden.
2. Generierung der synthetischen Trainingsdaten: BlenderProc (siehe `blenderproc/README.md`)

