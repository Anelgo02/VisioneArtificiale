import cv2
import numpy as np
#importiamo il nostro modulo matematico
import metrics 

def applica_trasformazione(img_mov,tx,ty,theta):
    """
    Applica una roto-traslazione all'immagine moving rispetto al suo centro.
    Utilizza l'interpolazione bilineare come richiesto dalle specifiche.
    
    Parametri:
    img_mov (numpy.ndarray): L'immagine da spostare/ruotare.
    tx (float): Traslazione lungo l'asse X (colonne).
    ty (float): Traslazione lungo l'asse Y (righe).
    theta (float): Angolo di rotazione in gradi.
    
    Ritorna:
    numpy.ndarray: La nuova immagine trasformata.
    """

    # 1. Creo un clone digitale per garantire che l'immagine originale
    # non venga sporcata in memoria
    img_sicura = img_mov.copy()

    # Calcolo dimensioni e centro 
    # h altezza (numero di righe), w larghezza(numero di colonne)
    h, w = img_sicura.shape[:2]
    # centro in base al quale fare le rotazioni,
    # in prima istanza avevo settato il centro come il centro dell'immagine ma
    # impostandolo a (0,0) l'errore diminuisce 
    centro = (0,0) 

    # 3. Creazione della matrice Roto-traslazione (2x3)
    M = cv2.getRotationMatrix2D(centro,theta,1.0)

    # Aggiungo le componenti di traslazione alla terza colonna della matrice
    M[0,2] += tx
    M[1,2] += ty

    # 4. Applicazione del Warping
    # cv2.INTER_LINEAR forza l'interpolazione bilineare
    img_warped = cv2.warpAffine(img_sicura, M, (w,h), flags=cv2.INTER_LINEAR)

    return img_warped

# Lista globale per salvare la storia della MI durante un'ottimizzazione
mi_history = []

def funzione_obiettivo(parametri, img_ref,img_mov,bins):
    """
    Funzione di "costo" che verrà passata all'ottimizzatore (es. Powell).
    L'ottimizzatore proverà a minimizzare il valore restituito da questa funzione.
    
    Parametri:
    parametri (list/tuple): Contiene i 3 valori provati dall'ottimizzatore [tx, ty, theta].
    img_ref (numpy.ndarray): L'immagine di riferimento fissa.
    img_mov (numpy.ndarray): L'immagine da allineare.
    bins (int): Il numero di bin per l'istogramma (es. 64, 128, 256).
    
    Ritorna:
    float: Il valore NEGATIVO della Mutua Informazione.
    """

    # Spacchettiamo l'array dei parametri generato da SciPy
    tx, ty, theta = parametri

    # 1. Spostiamo temporaneamente l'immagine con i parametri attuali
    img_trasformata = applica_trasformazione(img_mov, tx, ty, theta)

    # 2. Calcoliamo quanto è "buono" questo spostamento
    mi = metrics.calcola_mutua_informazione(img_ref, img_trasformata, bins)
    
    # 3. IL TRUCCO DELL'OTTIMIZZATORE
    # SciPy cerca il MINIMO assoluto. Noi cerchiamo la Mutua Informazione MASSIMA.
    # Restituendo "-mi", inganniamo la libreria facendole cercare il nostro massimo.

    # Salviamo il valore positivo della MI per il grafico
    mi_history.append(mi)
    
    return -mi