import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

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


def to_object(df, columns):
    for column in columns:
        df[column] = df[column].astype(object)


def to_float(df, columns):
    for column_name in columns:
        remove_commas(df, column_name)
        df[column_name] = df[column_name].astype(float)


def remove_commas(df, column_name):
    df[column_name] = df[column_name].astype(str)
    df[column_name] = df[column_name].str.replace(",", "")
    df[column_name] = df[column_name].str.replace("<", "")


def count_nans(data):
    total = []
    for column in data.select_dtypes(exclude=['object']).columns:
        count = np.sum(np.isnan(data[column].values))
        total.append((column, count))
    return total


def replace_nans(data):
    for column in data.select_dtypes(exclude=['object']).columns:
        items = data[column].dropna(axis=0, inplace = False)
        data[column].fillna(random.choice(list(items)), inplace = True)


def distribution_plot(df, column_name, target_column, xlab, ylab, title, plot_type="box", order=False):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    if plot_type == "box":
        ax = sns.boxplot(df[column_name], df[target_column])
    elif plot_type == "violin":
        ax = sns.violinplot(df[column_name], df[target_column])
    elif plot_type == "bar":
        ax = sns.barplot(df[column_name], df[target_column], palette="Greens_d", order=order)
    ax.set_xlabel(xlab, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylab, fontweight="bold", fontsize=14)
    plt.xticks(rotation=75)
    plt.suptitle(title, fontweight="bold", fontsize=16)
    plt.show()


if __name__ == "__main__":
    #===========================================================================
    #=========================== DATA CLEANING =================================
    #===========================================================================

    # sahie data set
    sahie = pd.read_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/data/health_insurance/SAHIE_31JAN17_13_18_47_11.csv")

    sahie = separate_states(sahie)
    sahie.drop(['Age Category','Income Category','Race Category','Sex Category'], axis=1, inplace=True)
    to_object(sahie, ["Year",'ID'])
    to_float(sahie, ['Uninsured: Number',"Uninsured: MOE",'Insured: Number','Insured: MOE'])
    sahie_nans = count_nans(sahie)
    sahie = sahie.dropna(axis=0)
    to_object

    # mediare data set
    medicare = pd.read_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/data/medicare_county_level/cleaned_medicare_county_all.csv")
    medicare.drop('unnamed:_0', axis=1, inplace=True)
    should_be_objects = list(medicare.select_dtypes(include=['int64']).columns)
    to_object(medicare, should_be_objects)
    medicare_nans = count_nans(medicare)
    # dropping all nans: While this does cut my data down to roughly a third of what it was, my reasoning is that since the purpose of this data set will be prediction, I would rather have the precision of my model decline due to the lack of data than have it ARTIFICIALLY increase due to reducing the noise in each column by imputing the mean.
    medicare = medicare.dropna(axis=0)

    # medicare spending by year data set
    years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    med_spending = import_dfs(years)
    med_spending = change_type(med_spending)
    to_object(med_spending, ['county_id'])
    med_spending_nans = count_nans(med_spending)

    stratified = med_spending.groupby('year').count()[['medicare_enrollees','medicare_enrollees_(20%_sample)']]

    states = list(sahie.groupby('state').count().index)

    #===========================================================================
    #=============== VISUALIZATION, EDA & HYPOTHESIS TESTING ===================
    #===========================================================================

    distribution_plot(sahie, "state", "Uninsured: %", "State", "Percentage (%) Un-Insured", "Percentage Un-Insured Across States")

    descending_uninssured = list(sahie.groupby("state").mean().sort_values('Uninsured: %', ascending=False).index)

    distribution_plot(sahie, 'state', 'Uninsured: %', 'State', "Percentage (%) Un-Insured", "Ordered Percentage Un-Insured", plot_type="bar", order=descending_uninssured)

    distribution_plot(sahie, "Year", "Uninsured: %", "Year", "Percentage (%) Un-Insured", "Percentage Un-Insured Across Years")
