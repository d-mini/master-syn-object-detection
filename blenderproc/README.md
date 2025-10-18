
Generierung synthetischer Daten mit BlenderProc
=========================================================

Dieser Unterordner enthält BlenderProc-Skripte zur Erzeugung synthetischer Trainingsdaten für die Experimente der Masterarbeit.

Ordner-Struktur
------------------------------------
- `assets/` — Für die Generierung relevante Assets.
    - `assets/objects/` — 3D-Modelle (Zielobjekte und Distraktoren).
    - `assets/materials/` — Materialdateien.
    - `assets/hdr/` — HDR-Bilder, die für die Beleuchtung verwendet werden.
    - `assets/scenes/` — Blender-Dateien mit komplexen 3D-Szenen.
- `scripts/` — Skripte zur Generierung synthetischer Daten für die einzelnen Experimente (Baseline, DR-*).

Voraussetzungen
-------------
1. BlenderProc (siehe https://github.com/DLR-RM/BlenderProc).

Beispiele
----------------------
Erzeuge 100 Bilder für das Baseline-Experiment (Szenario 1):

```bash
blenderproc run scripts/baseline.py --scenario 1 --inst-per-class 100
```

Erzeuge Bilder für DR-Dis0 (Szenario 2):
```bash
blenderproc run scripts/dr_dis.py --scenario 2 --type-level 0
```

Erzeuge Bilder für DR-Hin1 (Szenario 1, Baustellen-Szene):
```bash
blenderproc run scripts/dr_hin1.py --scenario 1 --scene construction
```

Erzeuge Bilder für DR-Mix (Szenario 1, Stadt-Szene):
```bash
blenderproc run scripts/dr_mix.py --scenario 1 --scene city
```

Die generierten Bilder werden im Verzeichnis `output/` gespeichert.

`--scenario` ist für alle Skripte erforderlich. `--inst-per-class` kann für alle Skripte gesetzt werden (Default: 1).

Die Datei `scripts/config.yaml` definiert gemeinsame Einstellungen wie Kamerauflösung und die (maximale) Anzahl an Samples, die pro Pixel gerendert werden.

Sonstige Skripte
-----------
- `scripts/helpers.py` — Hilfsfunktionen, die von mehreren Skripten verwendet werden.
- `scripts/misc/frontal_shot.py` — erzeugt frontale Bilder der Zielobjekte vor grünem Hintergrund. Wurde für Abbildungen in der Masterarbeit verwendet.
- `scripts/misc/converter.py` — konvertiert COCO-Annotationen in YOLO-Labels; die Labels werden in einem Unterordner im gleichen Verzeichnis wie die Annotationen gespeichert.