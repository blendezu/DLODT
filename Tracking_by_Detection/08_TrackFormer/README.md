Generation 1 – Klassisch (Geschwindigkeit)

SORT (2016): Kalman-Filter für Bewegungsvorhersage + IoU-basierte Zuordnung via Hungarian Algorithm. ~260 FPS, kaum Re-ID.
IoU Tracker (2017): Noch simpler – rein auf Box-Überlappung. Extrem schnell, versagt bei Okklusion.
Centroid Tracker: Ordnet Tracks über euklidische Distanz der Mittelpunkte zu.

Generation 2 – Appearance / Re-ID

DeepSORT (2017): Ergänzt SORT um ein CNN-Embedding für visuelle Ähnlichkeit, toleriert Okklusion deutlich besser.
StrongSORT (2022): DeepSORT mit EMA-Embeddings (glattere Feature-Updates), OSNet Re-ID, AFLink-Nachbearbeitung.
BoT-SORT (2022): Fügt Global Motion Compensation (GMC) hinzu – kompensiert Kamerabewegung vor dem Matching.

Generation 3 – Hochperformant

ByteTrack (2022): Kernidee – auch low-confidence Detektionen werden für Assoziierung genutzt, kein hartes Konfidenz-Cutoff. SOTA auf MOTChallenge.
OC-SORT (2022): Observation-Centric, korrigiert Kalman-Drift bei Okklusion durch Re-Aktivierung aus echten Beobachtungen.
MotionTrack (2023): Optimiert für schnelle, nicht-lineare Bewegungen.

Generation 4 – Transformer / End-to-End

TrackFormer (2021): DETR-Basis, Detektion und Tracking in einem einzigen Netz.
MOTR (2021): Track Queries propagieren über Frames – kein separates Re-ID nötig.
MeMOT (2022): Langzeit-Memory-Mechanismus für wiederkehrende Objekte.

Generation 5 – Spezialisiert / Hybrid

SparseTrack (2023): Pseudo-Depth-Karte für bessere Okklusions-Behandlung.
Deep OC-SORT: OC-SORT mit Appearance-Kosten, hybride Assoziierung.
Hybrid SORT (2023): Kombiniert IoU + Pose-Keypoints + Appearance-Features.