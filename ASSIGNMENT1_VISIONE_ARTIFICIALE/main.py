import argparse
import cv2
import numpy as np
from scipy.optimize import minimize
from registrator import applica_trasformazione, funzione_obiettivo

def main():
    # --- GESTIONE ARGOMENTI ---
    parser = argparse.ArgumentParser(description="Registrazione Immagini tramite Mutua Informazione")
    parser.add_argument("--ref", type=str, required=True, help="Percorso immagine Reference (Fissa)")
    parser.add_argument("--mov", type=str, required=True, help="Percorso immagine Moving (Da allineare)")
    args = parser.parse_args()

    # --- CARICAMENTO ---
    img_ref = cv2.imread(args.ref, 0)
    img_mov = cv2.imread(args.mov, 0)

    if img_ref is None or img_mov is None:
        print("Errore: Impossibile trovare o leggere le immagini.")
        return

    print("Analisi in corso... (Configurazione: Powell, 64 bins, Blur 7x7)")

    # --- PRE-PROCESSING ---
    # Usiamo il Blur 7x7 
    img_ref_b = cv2.GaussianBlur(img_ref, (7, 7), 0)
    img_mov_b = cv2.GaussianBlur(img_mov, (7, 7), 0)

    # --- OTTIMIZZAZIONE ---
    # Punto di partenza [0, 0, 0] (Tx, Ty, Theta)
    res = minimize(funzione_obiettivo, [0.0, 0.0, 0.0], 
                   args=(img_ref_b, img_mov_b, 64), method='Powell')

    tx, ty, theta = res.x

    # --- GENERAZIONE RISULTATI ---
    # Applichiamo i parametri trovati all'immagine originale (senza blur)
    img_aligned = applica_trasformazione(img_mov, tx, ty, theta)
    # Calcoliamo la mappa di differenza
    img_diff = cv2.absdiff(img_ref, img_aligned)

    # --- OUTPUT ---
    print(f"\nRegistrazione completata:")
    print(f" > Traslazione X: {tx:.2f} px")
    print(f" > Traslazione Y: {ty:.2f} px")
    print(f" > Rotazione:     {theta:.2f} gradi")

    # Salvataggio
    cv2.imwrite("risultato_allineato.png", img_aligned)
    cv2.imwrite("mappa_differenza.png", img_diff)
    print("\nRisultati salvati in 'risultato_allineato.png' e 'mappa_differenza.png'")

    # Visualizzazione
    cv2.imshow("Reference", img_ref)
    cv2.imshow("Allineata", img_aligned)
    cv2.imshow("Differenza", img_diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()