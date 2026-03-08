'''
Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je placen 
po radnom satu. Koristite ugradenu Python metodu  input(). Nakon toga izracunajte koliko 
je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na nacin da ukupni iznos 
izracunavate u zasebnoj funkciji naziva  total_euro.
'''

radniSati = int(input("Unesi radne sate: "))
euroPoSatu = float(input("Unesi satnicu: "))

def total_euro() :
    return radniSati*euroPoSatu

zarada = total_euro()
print(f"Ukupno si zaradila {zarada} eura.")