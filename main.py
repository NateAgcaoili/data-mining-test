import bayes
import ID3

classifications = { "1": "ID3", "2": "Bayes" }

print("\n---<<Welcome to final Data Mining project by Anthony Agcaoili>>--")
print("Please enter either [1] for ID3 or [2] for Bayes classification showcase.")
classification = input("Classification input: ")

try:
    print("\n" + classifications[classification], "has been selected.")
except KeyError:
    print("\nInvalid classification provided. Defaulting to [1] (ID3).")
    classification = '1'

if classification == '1':
    try:
        t1 = float(input("[FLOAT] Enter mixture ratio threshold: "))
    except ValueError:
        print("Invalid threshold provided. Defaulting to 10.")
        t1 = 10
    try:
        g = float(input("[FLOAT][0 < g < 0.015] Enter g value: "))
    except ValueError:
        print("Invalid g value provided. Defaulting to 0.007")
        g = 0.007
    if (g < 0) or (g >=0.015):
        print("g must be between 0 < g < 0.015. Defaulting to 0.007")
        g = 0.007
    ID3.main(t1, g)
else:
    bayes.main()
