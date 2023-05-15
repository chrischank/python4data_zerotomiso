####################################################
#Demonstration of XGBoost to predict Merc TCFD data#
#Maintainer: Christopher Chan                      #
#Version: 0.1.1                                    #
#Date: 2023-05-15                                  #
####################################################

import os, sys, re
import pathlib
import argparse

import numpy as np
import pyarrow as pa
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description = "Input ESData in parquet for TCFD subset and pivot")
parser.add_argument("--ESData", type=pathlib.Path, required=True,
                    help="Please specify path to typesetted ESData parquet file")
parser.add_argument("--AgentID", type=int, required=False, nargs="+",
                    help="Please provide an optional list of AgentID to be subset")
parser.print_help()
args = parser.parse_args()

for arg in vars(args):
    print("\t", args, getattr(args, arg))

# Setting paths
data_raw = pathlib.Path("../../data/raw")
data_interim = pathlib.Path("../../data/interim")
data_processed = pathlib.Path("../../data/processed")
data_external = pathlib.Path("../../data/external")

ESData = pd.read_parquet(args.ESData, engine = "pyarrow", dtype_backend = "pyarrow")


def scope_extract(df: pd.DataFrame, AgentID: list[int]|None=None) -> pd.DataFrame:
    # Let's make a list of relevant factors:
    Clim_factors = [4980, 4981, 4982, 4983, 4984, 4985, 4986, 4987, 4988, 4989,
                    4990, 4991, 4992, 4993, 4994, 4995, 4996, 4997, 4998, 4999,
                    5000, 5001, 5002, 5077, 5078, 5079, 5080, 5081, 5082, 5003,
                    5055, 5056, 5057, 5058, 5059, 5083, 4963]
    
    Clim = df[df["FactorId"].isin(Clim_factors)]
    
    if AgentID is not None:
        subset_Clim = Clim[Clim["AgentId"].isin(AgentID)] 

        return(subset_Clim)
    else:
        return(Clim)


def pivot_table(unpivot_df: pd.DataFrame) -> pd.DataFrame:
    pivot_df = pd.pivot_table(unpivot_df, values = "Answer", columns = "Question", index = "Date", 
                              aggfunc = lambda x: " ".join(x))

    return(pivot_df)

def main():
    Clim_extract = scope_extract(ESData, args.AgentID)

    # Check if TCFD_pivot in data_interim exist
    if os.path.exists(f"{data_interim}/TCFD_pivot"):
        print("TCFD_pivot directory exist.")
    else:
        os.mkdir(f"{data_interim}/TCFD_pivot")

    
    if args.AgentID is not None:
        AgentName = Clim_extract["AgentName"].values[0]
        AgentName = AgentName.replace("/", "_").replace(" ", "_")
        TCFDPivot = pivot_table(Clim_extract)
        
        with open(f"{data_interim}/TCFD_pivot/{AgentName}_TCFDPivot.csv", "wb") as pivot:
            TCFDPivot.to_csv(pivot, sep = ";")
    else:
        # Create a loop to output TCFD pivot table for each AgentId
        agent_list = Clim_extract["AgentId"].unique()
        print(agent_list)

        for i in agent_list:
            AID_Clim = Clim_extract[Clim_extract["AgentId"] == i]
            AgentName = AID_Clim["AgentName"].values[0]
            AgentName = AgentName.replace("/", "_").replace(" ", "_")
            TCFDPivot = pivot_table(AID_Clim)

            with open(f"{data_interim}/TCFD_pivot/{AgentName}_TCFDPivot.csv", "wb") as pivot:
                TCFDPivot.to_csv(pivot, sep = ";")


if __name__ == "__main__":
    main()
