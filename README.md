# Abnormal Action Detection in Videos

In this project we plan to apply results concerning online action detection in videos to abnormal event spotting in real-life datasets.
Therefore we will combine the method porposed by Waqas Sultani et al.(2018) with the online setting and compare online versus offline settings results.


## Real-world Anomaly Detection in Surveillance Videos : 

In this section we reproduce the results of Sultani et al.(2018). In this part we work in offline settings, and we are only interested in temporal detection.

## Online Action Detection :
In this part we are interset in abnormal action detection in videos, but in this section we are using online settings. We use the same C3D descriptors as in the previous, and we feed them to a RNN, in order to perform temporal abnormal action detection in videos.

## Application (`aad/`)

Application prête à l’emploi : API HTTP, interface web, et ligne de commande.

### Installation

```bash
pip install -r requirements.txt
```

Pour les descripteurs **C3D** (TensorFlow), utilisez un interpréteur **Python 3.10–3.12** puis :

```bash
pip install -r requirements-c3d.txt
```

Sans TensorFlow, l’outil utilise un **backend léger** basé sur le mouvement (OpenCV). Variable optionnelle : `AAD_BACKEND=lite` pour forcer ce mode.

### Lancer le serveur

```bash
uvicorn aad.server:app --reload --host 0.0.0.0 --port 8000
```

Ouvrez `http://127.0.0.1:8000` pour téléverser une vidéo et afficher la chronologie des scores.

### Ligne de commande

```bash
python -m aad chemin/vers/video.mp4 -o resultats.json
```
