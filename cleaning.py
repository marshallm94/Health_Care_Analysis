import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def format_excel(filepath):
    df = pd.read_excel(filepath)

    for column in df.columns:
        if "Unnamed" in column:
            loc = df.columns.get_loc(column)
            header = df.columns[loc - 1].replace(" ", "_").lower()
            new_name = header + "_price_age_sex_race_adj"
            values = df[column].values
            df.insert(loc, new_name, values)
            df.drop(column, axis=1, inplace=True)

    for x, column in enumerate(df.columns):
        if df[column][0] == "Age, sex & race-adjusted":
            loc = df.columns.get_loc(column)
            header = column.replace(" ", "_").lower()
            new_name = header + "_age_sex_race_adj"
            values = df[column].values
            df.insert(loc, new_name, values)
            df.drop(column, axis=1, inplace=True)
        elif x in [0,1,2]:
            new_name = column.lower().replace(" ", "_")
            values = df[column].values
            loc = df.columns.get_loc(column)
            df.insert(loc, new_name, values)
            df.drop(column, axis=1, inplace=True)

    return df.iloc[1:,:].reset_index(drop=True)


def change_type(df):
    for column in df.columns:
        if "_adj" in column:
            df[column] = df[column].astype(float)
    return df


def import_dfs(years):
    df = pd.DataFrame()
    for year in years:
        path = "/Users/marsh/galvanize/dsi/projects/health_capstone/data/medicare_spending_by_county/pa_reimb_county_{}.xls".format(str(year))
        subdf = format_excel(path)
        subdf = change_col_names(year, subdf)
        df = pd.concat([df, subdf])
    return df


def change_col_names(year, df):
    new_cols = []
    df['year'] = str(year)

    for column in df.columns:
        if str(year) in column:
            replacement = "_(" + str(year) + ")"
            new_cols.append(column.replace(replacement, ""))
        else:
            new_cols.append(column)

    df.columns = new_cols
    return df


def separate_states(df):
    abbr_dict = {
        'AK': 'Alaska',
        'AL': 'Alabama',
        'AR': 'Arkansas',
        'AS': 'American Samoa',
        'AZ': 'Arizona',
        'CA': 'California',
        'CO': 'Colorado',
        'CT': 'Connecticut',
        'DC': 'District of Columbia',
        'DE': 'Delaware',
        'FL': 'Florida',
        'GA': 'Georgia',
        'GU': 'Guam',
        'HI': 'Hawaii',
        'IA': 'Iowa',
        'ID': 'Idaho',
        'IL': 'Illinois',
        'IN': 'Indiana',
        'KS': 'Kansas',
        'KY': 'Kentucky',
        'LA': 'Louisiana',
        'MA': 'Massachusetts',
        'MD': 'Maryland',
        'ME': 'Maine',
        'MI': 'Michigan',
        'MN': 'Minnesota',
        'MO': 'Missouri',
        'MP': 'Northern Mariana Islands',
        'MS': 'Mississippi',
        'MT': 'Montana',
        'NA': 'National',
        'NC': 'North Carolina',
        'ND': 'North Dakota',
        'NE': 'Nebraska',
        'NH': 'New Hampshire',
        'NJ': 'New Jersey',
        'NM': 'New Mexico',
        'NV': 'Nevada',
        'NY': 'New York',
        'OH': 'Ohio',
        'OK': 'Oklahoma',
        'OR': 'Oregon',
        'PA': 'Pennsylvania',
        'PR': 'Puerto Rico',
        'RI': 'Rhode Island',
        'SC': 'South Carolina',
        'SD': 'South Dakota',
        'TN': 'Tennessee',
        'TX': 'Texas',
        'UT': 'Utah',
        'VA': 'Virginia',
        'VI': 'Virgin Islands',
        'VT': 'Vermont',
        'WA': 'Washington',
        'WI': 'Wisconsin',
        'WV': 'West Virginia',
        'WY': 'Wyoming'}
    state_dict = {v: k for k, v in abbr_dict.items()}
    df['state'] = df["Name"].map(state_dict)
    df['state'] = df['state'].astype(str)
    df['state'] = np.where(df['state'] == "nan", df.Name.str[-2:], df['state'])
    return df


if __name__ == "__main__":
    # import data
    sahie = pd.read_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/data/health_insurance/SAHIE_31JAN17_13_18_47_11.csv")
    medicare = pd.read_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/data/medicare_county_level/cleaned_medicare_county_all.csv")

    years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]

    med_spending = import_dfs(years)
    med_spending = change_type(med_spending)

    stratified = med_spending.groupby('year').mean()[['medicare_enrollees','medicare_enrollees_(20%_sample)']]

    sahie = separate_states(sahie)
