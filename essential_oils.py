import os, sys
import pandas as pd
import pubchempy as pcp
import re
import time
from pathlib import Path

root_path = Path(__file__).resolve().parent.as_posix()
sys.path.append(root_path)
print(f"Root path: {root_path}")

data_path = os.path.join(root_path, "data")
print(f"Data path: {data_path}")

sys.path.append(root_path)
sys.path.append(data_path)

class Smiles:
    def __init__(self, smiles_file = "smiles_names.csv", delay=0.25):
        self.delay = delay
        self.smiles_file = smiles_file
        if not os.path.exists(self.smiles_file):
            print(f"File {self.smiles_file} does not exist. Creating a new file.")
            # Create an empty csv with header smiles;name
            with open(self.smiles_file, "w") as f:
                f.write("smiles;name\n")

        smiles_df = pd.read_csv(self.smiles_file, sep=";")
        # Convert to dictionary, keyed by the smiles
        self.smiles_dict = smiles_df.set_index("smiles").T.to_dict("list")
        self.names_dict = {}
        for smiles, names in self.smiles_dict.items():
            for name in names:
                if name not in self.names_dict:
                    self.names_dict[name] = []
                self.names_dict[name].append(smiles)
        print(f"Loaded {len(self.smiles_dict)} smiles from {self.smiles_file}")

    def get_name(self, smiles):
        """Given a smiles string, returns the name of the compound."""
        if smiles in self.smiles_dict:
            return self.smiles_dict[smiles][0]
        else:
            # Look it up in pubchem
            print(f"Looking up {smiles} in pubchem... ", end="")
            # Wait for a bit to avoid hitting the API too hard
            time.sleep(self.delay)
            compounds = pcp.get_compounds(smiles, 'smiles')
            if compounds:
                name = compounds[0].iupac_name
                self.smiles_dict[smiles] = [name]
                if name not in self.names_dict:
                    self.names_dict[name] = [smiles]
                else:
                    self.names_dict[name].append(smiles)
                # Append to the file
                with open(self.smiles_file, "a") as f:
                    f.write(f"{smiles};{name}\n")
                print(f" -> {name}")
                return name
            else:
                print(f"Could not find {smiles} in pubchem")
                return None

    def get_smiles(self, name):
        if name in self.names_dict:
            return self.names_dict[name]        
        else:
            for sm, nm in self.smiles_dict.items():
                if name in nm:
                    return sm
                
            print(f"Looking up {name} in pubchem... ", end="")
            # Wait for a bit to avoid hitting the API too hard
            time.sleep(self.delay)
            compounds = pcp.get_compounds(name, 'name')
            if compounds:
                sm  = compounds[0].canonical_smiles
                print(f" -> {sm}")
                self.names_dict[name] = [sm]
                if sm not in self.smiles_dict:
                    self.smiles_dict[sm] = [name]
                    # Append to the file
                    with open(self.smiles_file, "a") as f:
                        f.write(f"{sm};{name}\n")
                return sm
            else:
                print(f"Could not find {name} in pubchem")
                return None
            

            

class EssentialOils:
    def __init__(self, essential_oils_file = "essential_oil_composition.csv", smiles_file = "smiles_names.csv"):
        self.essential_oils_file = essential_oils_file
        assert os.path.exists(self.essential_oils_file), f"Essential oils file {self.essential_oils_file} does not exist."

        self.smiles_db = Smiles(smiles_file)
        
        # Read the lines from the essential oils file
        with open(self.essential_oils_file, "r") as f:
            lines = f.readlines()

        # We want to split the lines on commas into name,smile
        # However some names are longer and themselves contain commas.
        # These are enclosed in double quotes.
        # We'll do this by finding any lines that start with a quote
        # finding the closing quote, and replacing ', ' between them with "_"
                          
        lines = [line.strip().replace('"', '').replace(", ","_") for line in lines[1:] if line.strip()]
        #
        print(f"Found {len(lines)} essential oils in the file.")
        oils, smiles = zip(*[l.split(",") for l in lines])        
        self.oils = list(set(oils))
        self.smiles = list(set(smiles))

        print(f"Found {len(self.oils)} unique essential oils.")
        print(f"Found {len(self.smiles)} unique SMILES strings.")

        names = {}
        missing_names = []
        for sm in self.smiles:
            names[sm] = self.smiles_db.get_name(sm)
            if names[sm] is None:
                missing_names.append(sm)

        self.names = names
        self.missing_names = missing_names
        print(f"Found {len(self.missing_names)} missing names.")

        self.components = {}
        for oi, sm in zip(oils, smiles):
            if oi not in self.components:
                self.components[oi] = []
            self.components[oi].append(sm)

    def has_all_components(self,smiles_list):
        """Given a list of smiles, returns the essential oils that have all the components in the list."""
        oils = []
        for oi, sm in self.components.items():
            if all(s in sm for s in smiles_list):
                oils.append(oi)
        return oils

                 
