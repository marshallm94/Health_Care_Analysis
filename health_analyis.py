import math as m
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from basis_expansions.basis_expansions import NaturalCubicSpline
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os,sys
import pymc3 as pm


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


def distribution_plot(df, column_name, target_column, xlab, ylab, title, filename, plot_type="box", order=None):
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
    if plot_type == "box":
        ax = sns.boxplot(df[column_name], df[target_column], order=order)
    elif plot_type == "violin":
        ax = sns.violinplot(df[column_name], df[target_column])
    elif plot_type == "bar":
        ax = sns.barplot(df[column_name], df[target_column], palette="Greens_d", order=order)
    ax.set_xlabel(xlab, fontweight="bold", fontsize=13)
    ax.set_ylabel(ylab, fontweight="bold", fontsize=13)
    plt.xticks(rotation=75)
    plt.suptitle(title, fontweight="bold", fontsize=16)
    plt.savefig(filename)
    plt.show()


def heatmap(df, filename):
    corr = df.corr()
    ylabels = ["{} = {}".format(col, x + 1) for x, col in enumerate(list(corr.columns))]
    xlabels = [str(x + 1) for x in range(len(ylabels))]
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(9, 5))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, xticklabels=xlabels, yticklabels=ylabels, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.5})
    plt.yticks(rotation=0)
    plt.suptitle("Correlation Between Attributes", fontweight="bold", fontsize=16)
    plt.savefig(filename)


def ANOVA(df, group_column, target_column):
    masks = []
    for val in list(df.groupby(group_column).count().index):
        mask = df[group_column] == val
        masks.append(mask)

    arrays = []
    for mask in masks:
        arr = df[mask][target_column]
        arrays.append(arr)
    return stats.f_oneway(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5], arrays[6], arrays[7], arrays[8])


def residual_plot(ax, x, y, y_hat, n_bins=50):
    residuals = y - y_hat
    ax.axhline(0, color="red", linestyle="--")
    ax.scatter(x, residuals, color="grey", alpha=0.5)
    ax.set_ylabel("Residuals ($y - \hat y$)")


if __name__ == "__main__":

    #===========================================================================
    #=========================== DATA CLEANING =================================
    #===========================================================================

    # sahie data set
    sahie = pd.read_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/data/health_insurance/SAHIE_31JAN17_13_18_47_11.csv")

    sahie = separate_states(sahie)
    sahie.drop(['Age Category','Income Category','Race Category','Sex Category','Demographic Group: MOE'], axis=1, inplace=True)
    to_object(sahie, ["Year",'ID'])
    to_float(sahie, ['Uninsured: Number',"Uninsured: MOE",'Insured: Number','Insured: MOE'])
    sahie_nans = count_nans(sahie)
    sahie = sahie.dropna(axis=0)
    to_object

    # mediare data set
    medicare = pd.read_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/data/medicare_county_level/cleaned_medicare_county_all.csv")
    medicare.drop('unnamed:_0', axis=1, inplace=True)
    should_be_objects = list(medicare.select_dtypes(include=['int64']).columns)
    should_be_objects.remove('year')
    to_object(medicare, should_be_objects)
    medicare_nans = count_nans(medicare)
    # dropping all nans: While this does cut my data down to roughly a third of what it was, my reasoning is that since the purpose of this data set will be prediction, I would rather have the precision of my model decline due to the lack of data than have it ARTIFICIALLY increase due to reducing the noise in each column by imputing the mean.
    medicare = medicare.dropna(axis=0)
    medicare['cost_per_beneficiary'] = medicare['total_actual_costs'] / medicare["beneficiaries_with_part_a_and_part_b"]

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

    heatmap(sahie, "figures/heatmap")

    descending_uninssured = list(sahie.groupby("state").mean().sort_values('Uninsured: %', ascending=False).index)

    distribution_plot(sahie, "state", "Uninsured: %", "State", "Percentage (%) Uninsured", "Percentage Uninsured Across States", filename="figures/state_vs_uninsured", order=descending_uninssured)

    distribution_plot(sahie, 'state', 'Uninsured: %', 'State', "Percentage (%) Uninsured", "Ordered Percentage Uninsured Across States", plot_type="bar", filename="figures/state_vs_uninsured_bar", order=descending_uninssured)

    distribution_plot(sahie, "Year", "Uninsured: %", "Year", "Percentage (%) Uninsured", "Percentage Uninsured Across Years", plot_type="violin", filename="figures/year_vs_uninsured")

    # create csv for anova in R
    df = sahie[['state','Year','Uninsured: %']]
    df.to_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/anova.csv")
    f, p = ANOVA(df, "Year", "Uninsured: %")

    #===========================================================================
    #=============================== MODELING ==================================
    #===========================================================================
    numerics = medicare[medicare.select_dtypes(exclude=['object']).columns]
    X = numerics.drop('cost_per_beneficiary', axis=1)
    y = pd.DataFrame(numerics['cost_per_beneficiary'])
    x_train, x_test, y_train, y_test = train_test_split(X, y)

    lasso_pipeline = Pipeline([
        ("standardize", StandardScaler()),
        ("lasso", Lasso(2.0))
    ])

    linear_pipeline = Pipeline([
        ("standardize", StandardScaler()),
        ("regression", LinearRegression())
    ])

    ridge_pipeline = Pipeline([
        ("standardize", StandardScaler()),
        ("ridge", Ridge(2.0))
    ])

    alphas = [100, 10, 5]
    alphas_2 = [4., 3., 2., 1.]

    for alpha in alphas_2:
        k = KFold(10)
        ridge_rsme_lis = []
        lasso_rsme_lis = []
        regularized_dict = {}
        for train_index, test_index in k.split(x_train):

            x_cv, x_cv_test = x_train.iloc[train_index,:], x_train.iloc[test_index,:]
            y_cv, y_cv_test = y_train.iloc[train_index,:], y_train.iloc[test_index,:]

            ridge_pipe = Pipeline([
                ("standardize", StandardScaler()),
                ("ridge", Ridge(alpha=alpha))
            ])
            lasso_pipe = Pipeline([
                ("standardize", StandardScaler()),
                ("lasso", Lasso(alpha=alpha))
            ])

            ridge_pipe.fit(x_cv, y_cv)
            lasso_pipe.fit(x_cv, y_cv)

            ridge_predictions = ridge_pipe.predict(x_cv_test)
            lasso_predictions = lasso_pipe.predict(x_cv_test)

            ridge_rsme = m.sqrt(mean_squared_error(y_cv_test, ridge_predictions))
            lasso_rsme = m.sqrt(mean_squared_error(y_cv_test, lasso_predictions))

            ridge_rsme_lis.append(ridge_rsme)
            lasso_rsme_lis.append(lasso_rsme)

        ridge_avg_cv_rsme = sum(ridge_rsme_lis)/len(ridge_rsme_lis)
        lasso_avg_cv_rsme = sum(lasso_rsme_lis)/len(lasso_rsme_lis)

        regularized_dict["Ridge"] = [alpha, ridge_avg_cv_rsme]
        regularized_dict["Lasso"] = [alpha, lasso_avg_cv_rsme]
        print(regularized_dict)

    models = [lasso_pipeline, linear_pipeline, ridge_pipeline]
    model_evaluation_dict = {}
    for model in models:
        k = KFold(10)
        rsme_lis = []
        for train_index, test_index in k.split(x_train):

            x_cv, x_cv_test = x_train.iloc[train_index,:], x_train.iloc[test_index,:]
            y_cv, y_cv_test = y_train.iloc[train_index,:], y_train.iloc[test_index,:]

            model.fit(x_cv, y_cv)
            predictions = model.predict(x_cv_test)
            rsme = m.sqrt(mean_squared_error(y_cv_test, predictions))
            model_name = list(model.named_steps.keys())[1]
            print(f"Model - {model_name} | RSME - {rsme}")
            rsme_lis.append(rsme)

        avg_cv_rsme = sum(rsme_lis)/len(rsme_lis)

        model_evaluation_dict[model_name] = avg_cv_rsme

    preds = ridge_pipeline.predict(x_test)
    test_rsme = m.sqrt(mean_squared_error(y_test, preds))
    print(test_rsme)

    #===========================================================================
    #========================= MODEL EVALUATION ================================
    #===========================================================================

    fig, ax = plt.subplots(figsize=(12, 3))
    residual_plot(ax, preds, y_test, preds)
    ax.set_title("Residuals by Predicted Values")
    ax.set_xlabel("$\hat y$")
    plt.show()
