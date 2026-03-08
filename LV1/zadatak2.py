'''
Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
nekakvu ocjenu i nalazi se izmedu 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju 
sljedecih uvjeta: 
>= 0.9 A
>= 0.8 B
>= 0.7 C
>= 0.6 D
< 0.6 F
Ako korisnik nije utipkao broj, ispišite na ekran poruku o grešci (koristite try i except naredbe).
Takoder, ako je broj izvan intervala [0.0 i 1.0] potrebno je ispisati odgovarajucu poruku.
'''

try:

    x = float(input("Unesite neki brojm koji je izmedu 0.0 i 1.0: "))
    
    if x < 0.0 or x > 1.0:
        print("Broj nije u ispravnom intervalu.")
    elif x >= 0.9:
        print("A")
    elif x >= 0.8:
        print("B")
    elif x >= 0.7:
        print("C")
    elif x >= 0.6:
        print("D")
    else:
        print("F")

except ValueError:
    print("Niste upisali broj.")
