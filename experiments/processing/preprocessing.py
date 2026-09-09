import os
import json
import argparse

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from oversampling import random_oversampling, smote_tomek_oversampling

import joblib

def process_csv(df):
    # Replace empty or whitespace-only strings with NaN
    df = df.replace(r"^\s*$", np.nan, regex=True)
    
    # 1. Remove rows where every single column is empty
    df = df.dropna(how="all")
    
    # 2. Identify non-metadata columns (e.g. antibiotic tests)
    metadata_cols = {"species", "code", "laboratory_species"}
    test_cols = [c for c in df.columns if c.lower() not in metadata_cols]
    
    # 3. Remove rows where all antibiotic test columns are empty (no AST data)
    if test_cols:
        df = df.dropna(subset=test_cols, how="all")
    
    # 4. Remove columns that have no data at all (100% NaN)
    df = df.dropna(axis=1, how="all")
    
    return df.reset_index(drop=True)


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--DRIAMS_dir", default=r"C:\Nguyen Tri\Paper\vi_khuan_khang_thuoc\DRIAMS\DRIAMS-A")
parser.add_argument("-f", "--Folder", help="What folder to get the data from", default="binned")
parser.add_argument("-n", "--Norm", help="Data normalization method. Supports \"none\",\"min-max\" and \"standard\"", default="standard", choices=["none", "min-max", "standard"])
parser.add_argument("-o", "--Oversampling", help="Method used to oversample data. Supports \"none\", \"random\" and \"smotetomek\"", default="none", choices=["none", "random", "smotetomek"])
parser.add_argument("-s", "--Split", help="How to do the split. Supports 80-20, 5x2, 100", default="80-20", choices=["80-20", "5x2", "100"])
parser.add_argument("-c", "--Filter", help="Use only files with specific string in their name")
args = parser.parse_args()


unprocessed_folder = args.DRIAMS_dir
processed_folder = "data/processed/"+args.Folder+"/"+args.Oversampling+"/"+args.Norm+"/"


if not os.path.exists(processed_folder):
    os.makedirs(processed_folder)

print("Saving at",  processed_folder)

# Determine antibiotic mapping
antibiotic_dict_paths = ["data/antibiotics.json", "experiments/data/antibiotics.json"]
antibiotic_dictionary = {}
for p in antibiotic_dict_paths:
    if os.path.exists(p):
        with open(p, "r") as f:
            antibiotic_dictionary = json.load(f)
        break


def load_driams_datasets(driams_dir, antibiotic_dict, filter_str=None):
    """
    Loads DRIAMS dataset from directory structure with 'id' and 'binned_6000' subfolders.
    Returns a list of (dataset_name, bacteria_df).
    """
    id_base = os.path.join(driams_dir, "id")
    binned_base = os.path.join(driams_dir, "binned_6000")

    if not (os.path.exists(id_base) and os.path.exists(binned_base)):
        return []

    years = sorted([
        y for y in os.listdir(id_base)
        if os.path.isdir(os.path.join(id_base, y)) and os.path.isdir(os.path.join(binned_base, y))
    ])

    loaded_datasets = []

    for year in years:
        if filter_str is not None and filter_str not in year:
            continue

        id_year_dir = os.path.join(id_base, year)
        csv_candidates = [f for f in os.listdir(id_year_dir) if f.endswith(".csv") and "clean" in f]
        if not csv_candidates:
            csv_candidates = [f for f in os.listdir(id_year_dir) if f.endswith(".csv")]
        if not csv_candidates:
            continue

        labels_file = os.path.join(id_year_dir, csv_candidates[0])
        print(f"Loading metadata for year {year}: {labels_file}")
        labels_df = pd.read_csv(labels_file)
        labels_df = process_csv(labels_df)

        if "code" not in labels_df.columns or "species" not in labels_df.columns:
            continue

        year_binned_dir = os.path.join(binned_base, year)

        # Target species: from antibiotic_dictionary if available, or species with >= 30 isolates
        target_species = [sp for sp in antibiotic_dict.keys() if sp in labels_df["species"].values]
        if not target_species:
            vc = labels_df["species"].value_counts()
            target_species = vc[vc >= 30].index.tolist()

        for species in target_species:
            sp_df = labels_df[labels_df["species"] == species].copy()

            if species in antibiotic_dict:
                antibiotics = [ab for ab in antibiotic_dict[species] if ab in sp_df.columns]
            else:
                metadata_cols = {"species", "code", "laboratory_species"}
                antibiotics = [c for c in sp_df.columns if c.lower() not in metadata_cols and sp_df[c].notna().sum() > 0]

            if not antibiotics:
                continue

            # Convert categorical resistance labels 'S' -> 0, 'R' -> 1
            sp_df[antibiotics] = sp_df[antibiotics].replace({"S": 0, "R": 1})
            for ab in antibiotics:
                sp_df[ab] = pd.to_numeric(sp_df[ab], errors="coerce")
            sp_df = sp_df.dropna(subset=antibiotics)

            if len(sp_df) < 10:
                continue

            # Load matching binned spectra
            spectra_list = []
            valid_rows = []
            for _, row in sp_df.iterrows():
                code = str(row["code"]).strip()
                spec_file = os.path.join(year_binned_dir, f"{code}.txt")
                if os.path.exists(spec_file):
                    try:
                        spec = np.loadtxt(spec_file, skiprows=1, usecols=1)
                        if len(spec) > 0:
                            spectra_list.append(spec)
                            valid_rows.append(row)
                    except Exception:
                        continue

            if len(spectra_list) < 10:
                continue

            spectra_arr = np.vstack(spectra_list)
            spec_cols = [str(i) for i in range(spectra_arr.shape[1])]
            df_spectra = pd.DataFrame(spectra_arr, columns=spec_cols)
            df_meta = pd.DataFrame(valid_rows).reset_index(drop=True)

            combined_df = pd.concat([df_spectra, df_meta[antibiotics]], axis=1)
            combined_df["species"] = species

            safe_name = f"{species.replace(' ', '_').replace('.', '')}_{year}"
            print(f"  -> Built dataset: {safe_name} ({combined_df.shape[0]} samples, {len(antibiotics)} antibiotics)")
            loaded_datasets.append((safe_name, combined_df))

    return loaded_datasets


# Prepare datasets: either from DRIAMS directory structure or individual CSV files
datasets_to_process = load_driams_datasets(args.DRIAMS_dir, antibiotic_dictionary, args.Filter)

# Fallback to standard CSV file listing if DRIAMS structure wasn't present
if not datasets_to_process:
    unprocessed_folder = args.DRIAMS_dir if os.path.isdir(args.DRIAMS_dir) else "data/unprocessed/" + args.Folder + "/"
    if os.path.exists(unprocessed_folder):
        unprocessed_files = [f for f in os.listdir(unprocessed_folder) if f.endswith(".csv")]
        if args.Filter is not None:
            unprocessed_files = [f for f in unprocessed_files if args.Filter in f]
        for f in unprocessed_files:
            file_path = os.path.join(unprocessed_folder, f)
            name = os.path.splitext(f)[0]
            df = pd.read_csv(file_path)
            datasets_to_process.append((name, df))

if not datasets_to_process:
    print("No datasets found to process. Please check --DRIAMS_dir or data path.")
    exit(0)

if args.Split == "5x2":
    seeds = [0, 1, 2, 3, 4]
    min_instances = 2
elif args.Split == "100":
    seeds = [0]
    min_instances = 2
else:
    seeds = [0]
    min_instances = 10

for seed in seeds:
    for file_name, bacteria in datasets_to_process:
        print(f"Processing: {file_name} (Seed: {seed})")
        ext = ".csv"

        malditof = bacteria[bacteria.columns.drop(list(bacteria.filter(regex='[^0-9]')))]

        if "species" in bacteria.columns:
            species = bacteria["species"].iloc[0]
            if species in antibiotic_dictionary:
                antibiotics = [ab for ab in antibiotic_dictionary[species] if ab in bacteria.columns]
            else:
                antibiotics = list(bacteria.columns.difference(malditof.columns))
                if "species" in antibiotics:
                    antibiotics.remove("species")
        else:
            antibiotics = list(bacteria.columns.difference(malditof.columns))

        trimmed_bac = pd.DataFrame(np.column_stack([malditof, bacteria[antibiotics]]))
        trimmed_bac.columns = list(malditof.columns) + antibiotics

        scrubbed_bac = trimmed_bac.dropna()

        value_counts = scrubbed_bac[antibiotics].value_counts()
        df_value_counts = value_counts.rename("Count").to_frame().reset_index()
        to_drop = df_value_counts.loc[df_value_counts["Count"] < min_instances].drop("Count", axis=1)
        reduced_aux_bac = pd.merge(scrubbed_bac, to_drop, indicator=True, how='outer', on=antibiotics)
        reduced_bac = reduced_aux_bac.loc[reduced_aux_bac["_merge"] == "left_only"].drop("_merge", axis=1)

        x = reduced_bac[reduced_bac.columns.drop(list(reduced_bac.filter(regex='[^0-9]')))]
        y = reduced_bac[antibiotics]

        if args.Split == "5x2":
            test_size = 0.5
        elif args.Split == "80-20":
            test_size = 0.2

        if args.Split != "100":
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y[antibiotics])
        else:
            x_train = x.copy(deep=True)
            x_test = x.copy(deep=True)
            y_train = y.copy(deep=True)
            y_test = y.copy(deep=True)

        if args.Norm != "none":
            os.makedirs(processed_folder + "scaler/", exist_ok=True)
            if args.Norm == "min-max":
                scaler = MinMaxScaler()
            elif args.Norm == "standard":
                scaler = StandardScaler()

            scaler_name =  processed_folder+"scaler/"+file_name+".save"
            scaler_name_base = scaler_name.replace("driamsB", "driams").replace("driamsC", "driams").replace("driamsD", "driams")
            if os.path.exists(scaler_name_base):
                scaler = joblib.load(scaler_name_base)
            else:
                if args.Split == "5x2":
                    scaler_name = processed_folder+"scaler/"+str(seed)+"_"+file_name+".save"

                if args.Norm == "min-max":
                    scaler = MinMaxScaler()
                elif args.Norm == "standard":
                    scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                joblib.dump(scaler, scaler_name)

            x_test = scaler.transform(x_test)        

        lc_name =  processed_folder+"encoder/"+file_name+"_encoder.save"
        lc_name_base = lc_name.replace("driamsB", "driams").replace("driamsC", "driams").replace("driamsD", "driams")

        agg_columns = pd.DataFrame()
        agg_columns["Class"] = y_train.astype(int).astype(str).agg(''.join, axis=1)
        agg_columns["Class"] = agg_columns["Class"].astype(str)

        if os.path.exists(lc_name_base) and "driams_" not in file_name: 
            lc = joblib.load(lc_name_base)
        else:
            lc = LabelEncoder()
            lc.fit(agg_columns.values.ravel())

            if not os.path.exists(processed_folder+"encoder/"):
                os.makedirs(processed_folder+"encoder/")

            if args.Split == "5x2":
                joblib.dump(lc, processed_folder+"encoder/"+str(seed)+"_"+file_name+"_encoder.save")
            else:
                joblib.dump(lc, lc_name)

        train = pd.DataFrame(np.column_stack([x_train, y_train]))
        train.columns = list(malditof.columns) + antibiotics
        train["Class"] = train[antibiotics].astype(int).astype(str).agg(''.join, axis=1)
        train["Class"] = train["Class"].astype(str)
        train = train[train["Class"].isin(lc.classes_)]
        train = train.drop(columns=["Class"])

        test = pd.DataFrame(np.column_stack([x_test, y_test]))
        test.columns = list(malditof.columns) + antibiotics
        test["Class"] = test[antibiotics].astype(int).astype(str).agg(''.join, axis=1)
        test["Class"] = test["Class"].astype(str)
        test = test[test["Class"].isin(lc.classes_)]
        test = test.drop(columns=["Class"])


        if args.Oversampling != "none":
            x_train = train[train.columns.drop(list(train.filter(regex='[^0-9]')))]
            y_train = train[antibiotics]

            agg_columns = pd.DataFrame()
            agg_columns["Class"] = y_train.astype(int).astype(str).agg(''.join, axis=1)
            agg_columns["Class"] = agg_columns["Class"].astype(str)

            y_train_lps = lc.transform(agg_columns.values.ravel())
            
            if args.Oversampling == "random":
                x_train, y_train_lps = random_oversampling(x_train, y_train_lps)
            elif args.Oversampling == "smotetomek":
                x_train, y_train_lps = smote_tomek_oversampling(x_train, y_train_lps)

            y_train_agg = lc.inverse_transform(y_train_lps)
            
            y_train = []
            for instance in y_train_agg:
                y_train.append(list(instance))
            y_train = pd.DataFrame(y_train)

            train = pd.DataFrame(np.column_stack([x_train, y_train]))
            train.columns = list(malditof.columns) + antibiotics
        

        if args.Split == "5x2":
            train_name_a = processed_folder+"train_A__"+str(seed)+"_"+file_name+ext
            test_name_a = processed_folder+"test_A__"+str(seed)+"_"+file_name+ext

            train.to_csv(train_name_a, index=False, mode='w')
            test.to_csv(test_name_a, index=False, mode='w')

            train_name_b = processed_folder+"train_B__"+str(seed)+"_"+file_name+ext
            test_name_b = processed_folder+"test_B__"+str(seed)+"_"+file_name+ext

            test.to_csv(train_name_b, index=False, mode='w')
            train.to_csv(test_name_b, index=False, mode='w')
        else:
            test_name = processed_folder+"test_"+file_name+ext
            test.to_csv(test_name, index=False, mode='w')

            if not args.Split == "100":
                train_name = processed_folder+"train_"+file_name+ext
                train.to_csv(train_name, index=False, mode='w')

        print("Done.\n")