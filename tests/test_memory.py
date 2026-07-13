"""Mémoire : lecture tolérante, écriture atomique, accès concurrents."""
import json
import os
import threading


def test_mem_absente_retourne_vide(app_mod):
    assert app_mod._mem() == {}


def test_mem_corrompue_retourne_vide(app_mod):
    with open(app_mod.MEM_FILE, "w") as f:
        f.write("{pas du json")
    assert app_mod._mem() == {}


def test_roundtrip(app_mod):
    app_mod._mem_save({"clé": "valeur"})
    assert app_mod._mem() == {"clé": "valeur"}


def test_ecriture_atomique_ne_laisse_pas_de_fichier_temporaire(app_mod):
    app_mod._mem_save({"a": "1"})
    d = os.path.dirname(app_mod.MEM_FILE)
    leftovers = [f for f in os.listdir(d) if f.startswith(".ruche_mem_")]
    assert leftovers == []


def test_ecritures_concurrentes_ne_perdent_rien(app_mod):
    """50 threads écrivent chacun leur clé : avec le verrou + os.replace,
    aucune écriture ne doit être perdue ni le fichier corrompu."""
    def write(i):
        app_mod._mem_set(f"k{i}", f"v{i}")

    threads = [threading.Thread(target=write, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with open(app_mod.MEM_FILE) as f:
        data = json.load(f)  # le fichier reste du JSON valide
    assert data == {f"k{i}": f"v{i}" for i in range(50)}
