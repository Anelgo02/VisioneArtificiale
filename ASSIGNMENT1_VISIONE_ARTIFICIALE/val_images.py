import os
import cv2
import numpy as np
from scipy.optimize import minimize
from registrator import funzione_obiettivo, applica_trasformazione

def salva_risultati(img_ref, img_mov, tx, ty, rot, nome_cartella, nome_base):
    # Applichiamo la trasformazione finale
    img_aligned = applica_trasformazione(img_mov, tx, ty, rot)
    # Calcoliamo la differenza (absdiff)
    img_diff = cv2.absdiff(img_ref, img_aligned)
    
    os.makedirs(nome_cartella, exist_ok=True)
    cv2.imwrite(f"{nome_cartella}/{nome_base}_aligned.png", img_aligned)
    cv2.imwrite(f"{nome_cartella}/{nome_base}_diff.png", img_diff)

def main():
    # 1. IMPOSTAZIONI (Le migliori trovate)
    BINS = 64
    METODO = 'Powell'
    
    # 2. GESTIONE CARTELLE
    val_dir = "DATASET/val" 
    output_dir = "RISULTATI_VAL"

    coppie = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])

    print(f"Avvio elaborazione test set con {METODO} e {BINS} bins...")

    for cartella in coppie:
        path = os.path.join(val_dir, cartella)
        files = os.listdir(path)
        
        f_r = [f for f in files if "R" in f][0]
        f_t = [f for f in files if "T" in f][0]
        
        img_ref = cv2.imread(os.path.join(path, f_r), 0)
        img_mov = cv2.imread(os.path.join(path, f_t), 0)

        # Pre-processing (Il tuo segreto del successo)
        img_ref_b = cv2.GaussianBlur(img_ref, (7, 7), 0)
        img_mov_b = cv2.GaussianBlur(img_mov, (7, 7), 0)

        # Ottimizzazione
        res = minimize(funzione_obiettivo, [0.0, 0.0, 0.0], 
                       args=(img_ref_b, img_mov_b, BINS), method=METODO)
        
        tx_e, ty_e, deg_e = res.x
        
        # Salva le immagini per lo ZIP finale
        salva_risultati(img_ref, img_mov, tx_e, ty_e, deg_e, output_dir, cartella)
        print(f"Coppia {cartella} completata.")

if __name__ == "__main__":
    main()