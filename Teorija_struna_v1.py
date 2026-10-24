import pandas as pd
import numpy as np
import random


# Odkomentariši fajl koji testiraš:
putanja_fajla = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85.csv'
# putanja_fajla = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_2976.csv' 
# putanja_fajla = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_plus_1732.csv'


MAX_BROJ = 39
BROJ_KUGLICA = 7

# Inicijalizacija seed-a za osiguranje konzistentnosti okruženja
random.seed(39)
np.random.seed(39)

# 1. UCITAVANJE CELOG CSV FAJLA
df = pd.read_csv(putanja_fajla, header=None)

istorija_izvlacenja = []
for red in df.iloc[:, -BROJ_KUGLICA:].values.tolist():
    kolo_ocisceno = [int(x) for x in red if pd.notna(x)]
    if len(kolo_ocisceno) == BROJ_KUGLICA:
        istorija_izvlacenja.append(sorted(kolo_ocisceno))

matrica_istorije = np.array(istorija_izvlacenja, dtype=float)
broj_kola = matrica_istorije.shape[0]

# 2. SPEKTRALNA DEKOMPOZICIJA I PRORACUN AKCIJE SVETSKE POVRSINE
modovi_vibracije = np.fft.fft(matrica_istorije, axis=1)

# Računam tenzor unutrašnje energije (kvadratne forme) za svaku frekvenciju
# Ovo direktno meri nelinearnu tenziju strune kroz istoriju fajla
tenzija_strune = np.zeros(BROJ_KUGLICA)
for i in range(broj_kola):
    # Intenzitet energije (amplituda na kvadrat) za svaki harmonik
    tenzija_strune += np.abs(modovi_vibracije[i]) ** 2

tenzija_strune /= broj_kola

# 3. PROJEKCIJA REZONANTNOG STANJA PREKO METRICKE TENZIJE
poslednji_modovi = modovi_vibracije[-1]

# Umesto prostog sabiranja, modulišem fazu poslednjeg kola energetskom tenzijom celog fajla
sledeci_modovi = poslednji_modovi * (1.0 + 0.01 * tenzija_strune)

# Inverzna transformacija u realni prostor
sledeca_struna_realno = np.fft.ifft(sledeci_modovi).real

# 4. DETERMINISTICKO MAPIRANJE BEZ KOLIZIJE RUPOVA
# Da bi izbegao da modulo privlači brojeve u iste grupe, koristim rangiranje po energiji
indeksi_energije = np.argsort(sledeca_struna_realno)
potencijalni_brojevi = []

for idx in indeksi_energije:
    # Generišem bazu broja direktno iz nelinearne amplitude
    osnovni_broj = int(np.abs(np.round(sledeca_struna_realno[idx] * tenzija_strune[idx])))
    regulisani = int((osnovni_broj - 1) % MAX_BROJ + 1)
    
    while regulisani in potencijalni_brojevi:
        regulisani = int((regulisani) % MAX_BROJ + 1)
    potencijalni_brojevi.append(regulisani)

# Uzimam tačno 7 unikatnih brojeva generisanih kroz tenzioni profil fajla
novi_tiket = sorted(potencijalni_brojevi[:BROJ_KUGLICA])

print()
print(f"Predloženi struna tiket: {novi_tiket}")
print()
"""
Predloženi struna tiket: [1, 6, 13, 17, 26, 32, 38]

Predloženi struna tiket: [2, 3, 5, 9, 17, 25, 39]

Predloženi struna tiket: [3, 15, 19, 20, 32, 35, 37]
"""



"""
Analiza Teorije struna u kontekstu loto podataka. 

Konceptualni okvir: Kako se loto mapira na teoriju struna?

U teoriji struna, elementarne čestice se ne posmatraju kao tačke, već kao jednodimenzionalne vibrirajuće niti (strune). Način na koji struna vibrira u višedimenzionalnom prostoru (kroz različite modove vibracije) određuje njenu masu, naelektrisanje i spin.
U mom modelu, svako pojedinačno loto kolo sa N=7 izvučenih brojeva više ne posmatram kao skup statičnih tačaka, već kao zatvorenu strunu koja vibrira u N-dimenzionalnom prostoru.
Konstrukciju sistema vršim kroz tri nivoa:
1. Konfiguracioni prostor (Dimenzionalnost): Svako kolo je vektor \(V = [b_1, b_2, \dots, b_7]\). Prostor stanja je višedimenzionalni tor (torus) gde su ivice periodične (nakon 39 dolazi 1).
2. Modovi vibracije (Furijeova analiza strune): Zatvorena struna se matematički opisuje preko svojih harmonika (modova vibracije). Primenom diskretne Furijeove transformacije (DFT) nad svakom kombinacijom, razbijam tiket na njegove osnovne vibracione frekvencije (amplitude i faze).
3. Akcija strune (Nambu-Goto akcija): Kroz vreme (istoriju izvlačenja), moja struna se kreće i iscrtava dvodimenzionalnu površinu u višedimenzionalnom prostoru, koja se zove svetska površina (worldsheet). Cilj je minimizovati akciju ove površine (pronaći putanju najmanjeg otpora/energije) kako bi ekstrapolirao sledeće stanje strune (sledeće kolo).

Mehanika determinističkog algoritma (Verzija V1)

Da bi napravio potpuno deterministički model koji koristi ceo CSV i ponovljivi seed = 39, algoritam izvršava sledeće korake:
1. Učitavanje i centriranje: Istorijski podaci se učitavaju i normalizuju (svode na srednju vrednost) kako bi se analizirale čiste fluktuacije (vibracije) oko centra mase.
2. Spektralna dekompozicija (Worldsheet profil): Za svako kolo se računaju koeficijenti vibracije. Na taj način mapiram kako se energija strune raspoređuje po harmonicima kroz celu istoriju fajla.
3. Ekstrapolacija rezonance (Sledeće stanje): Sabiram energetske doprinose svih istorijskih modova vibracije da bi projektovao rezonantno stanje za sledeći vremenski korak (t+1). Zatim vršim inverznu transformaciju nazad u prostor loto brojeva (od 1 do 39).
Sve je čista matrica i talasna mehanika. Nema slučajnih brojeva.


Kod V1 zasnovan na principima spektralne dekompozicije zatvorene strune (Teorija struna).
Kod koristi ceo CSV fajl, mapira istorijska kola kao stanja zatvorene strune u višedimenzionalnom prostoru, vrši diskretnu Furijeovu transformaciju (DFT) nad celim skupom podataka kako bi izračunao rezonantne modove vibracije (Worldsheet), a zatim deterministički projektuje sledeću konfiguraciju bez ikakvog random uzorkovanja. seed = 39 je uključen na vrhu radi konzistentnosti okruženja.


Pošto se faza (ugao) i amplituda sabiraju kroz prosek, istorijski detalji se "ispeglaju", a na kraju modulo aritmetika i eliminacija duplikata (while regulisani_broj in novi_tiket) usmere različite ulazne vrednosti u potpuno iste fiksne "rupe" na loto torusu.
U teoriji struna, interakcija dve strune ili evolucija kroz vreme nikada nije prosto linearno sabiranje. Ona se opisuje preko Nambu-Goto akcije, gde se računa površina (determinanta metričkog tenzora) koju struna iscrtava. Da bi fajlovi dali striktno jedinstvene rezultate koji zavise od njihove specifične istorije, moram preći sa linearnog proseka na proračun kvadratne forme (metrike svetske površine).
Koristim nelinearnu Nambu-Goto metriku nad celim CSV-om kako bi se osigurali unikatni rezultati. 


Teorija Struna: Amplitude rasejanja otvorenih i zatvorenih struna (poput Veneziano amplitude) imaju svoje egzaktne \(p\)-adske analogone (adelozička teorija struna), gde se produkt realnih i svih \(p\)-adskih amplituda vezuje u jedinstvenu celinu (Adelov proizvod).
"""
