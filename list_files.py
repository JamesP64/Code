import os

base = r"C:\Users\james\Documents\.UNI\Year_3\Individual Project\Code\data"
print("\nFILES IN DATA FOLDER:\n")
for f in os.listdir(base):
    print(repr(f))
