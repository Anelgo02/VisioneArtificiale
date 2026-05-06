import os
import numpy as np
import cv2
from scipy.optimize import minimize
import registrator  # Importiamo il modulo per gestire la lista mi_history
from registrator import funzione_obiettivo

def leggi_ground_truth(percorso_csv, nome_base_file, dataset_type='val'):
    """
    Legge i dati reali (Ground Truth) dal file CSV.
    """
    data = np.genfromtxt(percorso_csv, delimiter=';', dtype=None, encoding='utf-8', names=True)
    
    # Filtriamo per Filename e Dataset (val o test)
    riga = data[(data['Filename'] == nome_base_file) & (data['Dataset'] == dataset_type)]
    
    if riga.size > 0:
        return float(riga['Tx'][0]), float(riga['Ty'][0]), float(riga['AngleRad'][0])
    return None, None, None

def esegui_validazione():
    # 1. DEFINIZIONE PERCORSI
    base_dir = "DATASET"
    val_dir = os.path.join(base_dir, "val")
    csv_path = os.path.join(base_dir, "GT.csv")
    
    if not os.path.exists("dati_grafici"):
        os.makedirs("dati_grafici")
    
    # 2. PARAMETRI DI TEST
    lista_bins = [64, 128, 256] 
    lista_metodi = ['Powell', 'Nelder-Mead'] 
    
    risultati_finali = {}
    cartelle_coppie = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])

    for metodo in lista_metodi:
        for bins in lista_bins:
            nome_test = f"{metodo}_{bins}bins"
            print(f"\n--- Analisi Pipeline: {nome_test} (Pre-processing: Blur 7x7) ---")
            err_t, err_a = [], []

            for cartella in cartelle_coppie:
                path = os.path.join(val_dir, cartella)
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
                
                tx_gt, ty_gt, rad_gt = leggi_ground_truth(csv_path, nome_base, dataset_type='val')
                if tx_gt is None: continue

                registrator.mi_history = []

                # --- OTTIMIZZAZIONE ---
                res = minimize(funzione_obiettivo, [0.0, 0.0, 0.0], 
                               args=(img_ref_p, img_mov_p, bins), method=metodo)
                
                tx_e, ty_e, deg_e = res.x
                rad_e = np.deg2rad(deg_e)

                # --- CALCOLO ERRORI  ---
                # Calcoliamo l'errore assoluto per ogni componente
                err_x = abs(tx_e - tx_gt)
                err_y = abs(ty_e - ty_gt)
                 
                # MAE
                errore_traslazione_coppia = (err_x + err_y) / 2.0
                
                error_rad = abs(rad_e - rad_gt)
                
                err_t.append(errore_traslazione_coppia)
                err_a.append(error_rad)

                # Salvataggio dati per grafici MI slide (solo per Powell 64 bins, le coppie c1 e c6)
                if bins == 64 and metodo == 'Powell' and cartella in ['c1', 'c6']:
                    np.savetxt(f"dati_grafici/mi_trend_{cartella}.txt", registrator.mi_history)

                print(f"  > {cartella}: MAE_xy={errore_traslazione_coppia:6.2f}px | Rot={error_rad:8.5f}rad")

            # --- STATISTICHE FINALI ---
            mae_t = np.mean(err_t)
            mae_a = np.mean(err_a)
            risultati_finali[nome_test] = (mae_t, mae_a)
            print(f"FINE {nome_test} -> MAE Finale Traslazione: {mae_t:.3f} px")

    print("\n" + "="*65)
    print(f"{'CONFIGURAZIONE PIPELINE':<30} | {'MAE Trans':<12} | {'MAE Rad':<10}")
    print("-" * 65)
    for k, v in risultati_finali.items():
        print(f"{k:<30} | {v[0]:<12.3f} | {v[1]:<10.5f}")
    print("="*65)

if __name__ == "__main__":
    esegui_validazione()