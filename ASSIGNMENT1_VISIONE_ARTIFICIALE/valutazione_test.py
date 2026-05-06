import os
import numpy as np
import cv2
from scipy.optimize import minimize
import registrator
from registrator import funzione_obiettivo, applica_trasformazione

def leggi_ground_truth(percorso_csv, nome_base_file, dataset_type='test'):
    """
    Legge i dati reali (Ground Truth) dal file CSV filtrando per il Test Set.
    """
    data = np.genfromtxt(percorso_csv, delimiter=';', dtype=None, encoding='utf-8', names=True)
    riga = data[(data['Filename'] == nome_base_file) & (data['Dataset'] == dataset_type)]
    if riga.size > 0:
        return float(riga['Tx'][0]), float(riga['Ty'][0]), float(riga['AngleRad'][0])
    return None, None, None

def esegui_test_finale():
    # --- CONFIGURAZIONE OTTIMALE (Risultante dalla Validazione) ---
    BINS = 64
    METODO = 'Powell'
    
    base_dir = "DATASET"
    test_dir = os.path.join(base_dir, "test")
    csv_path = os.path.join(base_dir, "GT.csv")
    output_dir = "RISULTATI_TEST"
    
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    err_t, err_a = [], []
    cartelle = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])

    print(f"--- ANALISI FINALE SU TEST SET ({METODO}, {BINS} bins) ---")
    print(f"Metrica Errore: MAE (media componenti x, y)\n")

    for cartella in cartelle:
        path = os.path.join(test_dir, cartella)
        files = [f for f in os.listdir(path) if not f.startswith('.')]
        
        try:
            f_r = [f for f in files if f.split('.')[0].endswith('R')][0]
            f_t = [f for f in files if f.split('.')[0].endswith('T')][0]
        except: continue

        nome_base = f_r.rsplit('_', 1)[0]
        img_ref = cv2.imread(os.path.join(path, f_r), 0)
        img_mov = cv2.imread(os.path.join(path, f_t), 0)

        # --- PRE-PROCESSING ---
        img_ref_p = cv2.GaussianBlur(img_ref, (7, 7), 0)
        img_mov_p = cv2.GaussianBlur(img_mov, (7, 7), 0)

        # --- OTTIMIZZAZIONE ---
        res = minimize(funzione_obiettivo, [0.0, 0.0, 0.0], 
                       args=(img_ref_p, img_mov_p, BINS), method=METODO)
        
        tx_e, ty_e, deg_e = res.x
        rad_e = np.deg2rad(deg_e)

        # --- CALCOLO ERRORE (MAE) ---
        tx_gt, ty_gt, rad_gt = leggi_ground_truth(csv_path, nome_base, dataset_type='test')
        
        if tx_gt is not None:
            # Calcolo MAE per singola coppia: (|dX| + |dY|) / 2
            err_x = abs(tx_e - tx_gt)
            err_y = abs(ty_e - ty_gt)
            mae_coppia = (err_x + err_y) / 2.0
            
            error_rad = abs(rad_e - rad_gt)
            
            err_t.append(mae_coppia)
            err_a.append(error_rad)
            print(f" > {cartella}: Errore Medio Traslazione {mae_coppia:.2f}px")

        # --- GENERAZIONE OUTPUT PER CONSEGNA ---
        # Applichiamo la trasformazione trovata all'immagine originale (senza blur)
        img_aligned = applica_trasformazione(img_mov, tx_e, ty_e, deg_e)
        cv2.imwrite(os.path.join(output_dir, f"{cartella}_aligned.png"), img_aligned)
        
        # Immagine differenza per valutare visivamente l'allineamento
        diff = cv2.absdiff(img_ref, img_aligned)
        cv2.imwrite(os.path.join(output_dir, f"{cartella}_diff.png"), diff)

    # --- RISULTATI FINALI ---
    print("\n" + "="*50)
    print(f"MAE FINALE TEST SET (Traslazione): {np.mean(err_t):.3f} px")
    print(f"MAE FINALE TEST SET (Rotazione):   {np.mean(err_a):.5f} rad")
    print("="*50)
    print(f"File salvati in: {output_dir}")

if __name__ == "__main__":
    esegui_test_finale()