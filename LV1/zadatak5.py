'''
Napišite Python skriptu koja ce ucitati tekstualnu datoteku naziva SMSSpamCollection.txt
[1]. Ova datoteka sadrži 5574 SMS poruka pri cemu su neke oznacene kao spam, a neke kao ham.
a) Izracunajte koliki je prosjecan broj rijeci u SMS porukama koje su tipa ham, a koliko je 
prosjecan broj rijeci u porukama koje su tipa spam. 
b) Koliko SMS poruka koje su tipa spam završava usklicnikom ?
'''

counterHam=0
counterHamWord=0
counterSpam=0
counterSpamWord=0
counterSpam2 = 0
fhand=open("SMSSpamCollection.txt")

for line in fhand:
    line = line.strip()
    if line.startswith("ham"):
        counterHam += 1
        counterHamWord += len(line.split()) - 1 
    elif line.startswith("spam"):
        counterSpam += 1
        counterSpamWord += len(line.split()) - 1  
        if line.endswith("!"):
            counterSpam2 += 1

fhand.close()

print("Prosjek rijeci u ham porukama:", counterHamWord / counterHam)
print("Prosjecan broj rijeci u spam porukama:", counterSpamWord / counterSpam)
print("Broj spam poruka sa usklicnikom:", counterSpam2)