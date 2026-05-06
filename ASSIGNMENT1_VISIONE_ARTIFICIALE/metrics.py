import numpy as np

def calcola_entropia(distribuzione_prob):
    p = distribuzione_prob[distribuzione_prob > 0]
    entropia = -np.sum(p * np.log2(p))
    return entropia

def calcola_mutua_informazione(img_ref, img_mov, bins=256):
    
    # --- 1. MASCHERA (Per ignorare i bordi neri 0) ---
    # Consideriamo solo i pixel che sono > 0 in ENTRAMBE le immagini
    mask = (img_ref > 0) & (img_mov > 0)
    
    ref_validi = img_ref[mask]
    mov_validi = img_mov[mask]
    
    # Se la sovrapposizione è nulla o troppo piccola, MI è 0
    if len(ref_validi) < 10: 
        return 0.0

    # --- 2. ISTOGRAMMA CONGIUNTO ---
    # Usiamo ref_validi e mov_validi, non le immagini intere
    istogramma_2d, _, _ = np.histogram2d(
        ref_validi, 
        mov_validi, 
        bins=bins, 
        range=[[0, 256], [0, 256]]
    )

    # --- 3. PROBABILITÀ E ENTROPIE ---
    prob_congiunta = istogramma_2d / np.sum(istogramma_2d) 
    
    prob_ref = np.sum(prob_congiunta, axis=1)
    prob_mov = np.sum(prob_congiunta, axis=0)

    h_ref = calcola_entropia(prob_ref)
    h_mov = calcola_entropia(prob_mov)
    h_congiunta = calcola_entropia(prob_congiunta)

    # --- 4. FORMULA FINALE ---
    mi = h_ref + h_mov - h_congiunta
    return mi