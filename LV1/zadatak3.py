'''
Napišite program koji od korisnika zahtijeva unos brojeva u beskonacnoj petlji 
sve dok korisnik ne upiše „Done“ (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
(npr. slovo umjesto brojke) na nacin da program zanemari taj unos i ispiše odgovarajucu poruku.
'''

brojevi = []

while True:
    x = input("Uneste neki broj za nastavak ili Done za kraj: ")
    if x == 'Done' : 
        break
    try:
        broj = float(x)
        brojevi.append(broj)
    except ValueError:
        print("Niste upisali broj.")

ukupnoBrojeva = len(brojevi)
midVrijednost = sum(brojevi) / len(brojevi)
minVrijednost = min(brojevi)
maxVrijednost = max(brojevi)


brojevi.sort()

print(f"koliko je brojeva korisnik unio: {ukupnoBrojeva}")
print(f"srednja vrijednost: {midVrijednost}")
print(f"minimalna vrijednost: {minVrijednost}")
print(f"maksimalna vrijednost: {maxVrijednost}")
print(f"{brojevi}")