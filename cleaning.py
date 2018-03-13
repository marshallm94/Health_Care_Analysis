import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xlrd
import re

def clean_excel(filepath):
    df = pd.read_excel("/Users/marsh/galvanize/dsi/projects/health_capstone/data/medicare_spending_by_county/pa_reimb_county_2003.xls")

pattern = "^Unnamed"
for column in test.columns:
    if "Unnamed" in column:
        print(column)
    else:
        continue




    return df

if __name__ == "__main__":
    # import csv's
    sahie = pd.read_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/data/health_insurance/SAHIE_31JAN17_13_18_47_11.csv")
    medicare = pd.read_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/data/medicare_county_level/cleaned_medicare_county_all.csv")

    xls = pd.ExcelFile("/Users/marsh/galvanize/dsi/projects/health_capstone/data/medicare_spending_by_county/pa_reimb_county_2003.xls")

    test = clean_excel
