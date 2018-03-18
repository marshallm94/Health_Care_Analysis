import math as m
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import scipy.stats as stats
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import os,sys


def format_excel(filepath):
    """
    Used in conjunction with other methods to import .xlsx files in ./data/medicare_spending_by_county
    """
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
    """
    Used in conjunction with other methods to import .xlsx files in ./data/medicare_spending_by_county

    Change columns with "_adj" in column name to be floats
    """
    for column in df.columns:
        if "_adj" in column:
            df[column] = df[column].astype(float)
    return df


def import_dfs(years, wd):
    """
    Used in conjunction with other methods to import .xlsx files in ./data/medicare_spending_by_county

    Iterates through each file in the above directory and combines all files into one Pandas DataFrame.
    """
    df = pd.DataFrame()
    for year in years:
        path = wd + "medicare_spending_by_county/pa_reimb_county_{}.xls".format(str(year))
        subdf = format_excel(path)
        subdf = change_col_names(year, subdf)
        df = pd.concat([df, subdf])
    return df


def change_col_names(year, df):
    """
    Used in conjunction with other methods to import .xlsx files in ./data/medicare_spending_by_county

    Removes redundant columns from data frame.
    """
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
    """
    Creates a new state column where each entry is the proper abbreviation of the appropriate state.
    """
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
    """
    Counts the NaN values in each column in a Pandas Data Frame, exluding columns of type object.
    """
    total = []
    for column in data.select_dtypes(exclude=['object']).columns:
        count = np.sum(np.isnan(data[column].values))
        total.append((column, count))
    return total


def replace_nans(data):
    """
    Replace NA's in a column by imputing a random choice of a non-NA value in that same column
    """
    for column in data.select_dtypes(exclude=['object']).columns:
        items = data[column].dropna(axis=0, inplace = False)
        data[column].fillna(random.choice(list(items)), inplace = True)


def distribution_plot(df, column_name, target_column, xlab, ylab, title, filename, plot_type="box", order=None):
    """
    Create various plot types leverage matplotlib.

    Inputs:
        df: (Pandas DataFrame)
        column_name: (str) - A column in df that you want to have on the x-axis
        target_column: (str) - A column in df that you want to have on the y_axis
        xlab, ylab, title: (all str) - Strings for the x label, y label and title of the plot, respectively.
        filename: (str) - the relative path to which you would like to save the image
        plot_type: (str) - "box", "violin" or "bar"
        order: (None (default) or list) - the ordering of the variable on the x-axis

    Output:
        None (displays figure and saves image)
    """
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111)
    if plot_type == "box":
        ax = sns.boxplot(df[column_name], df[target_column], order=order)
    elif plot_type == "violin":
        ax = sns.violinplot(df[column_name], df[target_column])
    elif plot_type == "bar":
        ax = sns.barplot(df[column_name], df[target_column], palette="Greens_d", order=order)
    ax.set_xlabel(xlab, fontweight="bold", fontsize=14)
    ax.set_ylabel(ylab, fontweight="bold", fontsize=14)
    plt.xticks(rotation=75)
    plt.suptitle(title, fontweight="bold", fontsize=16)
    plt.savefig(filename)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def heatmap(df, filename):
    """
    Creaates a heatmap of the correlation matrix of df (Pandas DataFrame).

    Inputs:
        df: (Pandas DataFrame)
        filename: (str) - the path to which you would like to save the image.

    Output:
        None (displays figure and saves image)
    """
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
    """
    Performs Analysis of Variance leveraging scipy.stats

    Inputs:
        group_column: (str) - The categorical column in df you would like to perform the test on.
        target_column: (str) - The continuous column in df to calculate the means of each group

    Output:
        scipy.stats.F-statistic object
    """
    arrays = []
    for val in list(df.groupby(group_column).count().index):
        mask = df[group_column] == val
        arr = df[mask][target_column]
        arrays.append(arr)

    return stats.f_oneway(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5], arrays[6], arrays[7], arrays[8])


def import_medicare_spending_data():
    """
    Used in conjunction with other methods to import .xlsx files in ./data/medicare_spending_by_county
    """

    # Other data sets proved more useful for analysis; abstracting med_soending data code away in case needed for further use
    years = [2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
    med_spending = import_dfs(years, data_wd)
    med_spending = change_type(med_spending)
    to_object(med_spending, ['county_id'])
    med_spending_nans = count_nans(med_spending)

    stratified = med_spending.groupby('year').count()[['medicare_enrollees','medicare_enrollees_(20%_sample)']]

    states = list(sahie.groupby('state').count().index)

    return med_spending


def tune_regularization_parameter(alphas, folds=10):
    """
    Iterates through alphas, cross validating both Lasso and Ridge Regression parameterized by that alpha, using the RMSE loss function.

    Inputs:
        alphas: (list) - List of alphas
        folds: (int) - number of folds for k-fold cross validation

    Output:
        Pandas DataFrame - Contains Alpha, Lasso_RMSE and Ridge_RMSE columns
    """
    regularized_dict = {"Ridge_RMSE": [], "Lasso_RMSE": [], "Alpha": []}
    for alpha in alphas:
        k = KFold(folds)

        ridge_rsme_lis = []
        lasso_rsme_lis = []

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

            ridge_rmse = m.sqrt(mean_squared_error(y_cv_test, ridge_predictions))
            lasso_rmse = m.sqrt(mean_squared_error(y_cv_test, lasso_predictions))

            ridge_rsme_lis.append(ridge_rmse)
            lasso_rsme_lis.append(lasso_rmse)

        avg_ridge = np.mean(ridge_rsme_lis)
        avg_lasso = np.mean(lasso_rsme_lis)

        regularized_dict["Alpha"].append(alpha)
        regularized_dict["Ridge_RMSE"].append(avg_ridge)
        regularized_dict["Lasso_RMSE"].append(avg_lasso)

    return pd.DataFrame(regularized_dict)


def cv_multiple_model_pipelines(models, folds=10):
    """
    Iterates through models, fitting and evaluating the RMSE of each model pipeline utilizing KFold cross validation with K = Folds

    Inputs:
        models: (list) - A list of model Pipelines
        folds: (int) - number of folds for k-fold cross validation

    Output:
        Dictionary - Keys = Names of models, Values = RMSE over KFold cross validation with K = folds
    """
    model_evaluation_dict = {}

    for model in models:

        k = KFold(folds)
        rmse_lis = []

        for train_index, test_index in k.split(x_train):

            x_cv, x_cv_test = x_train.iloc[train_index,:], x_train.iloc[test_index,:]
            y_cv, y_cv_test = y_train.iloc[train_index,:], y_train.iloc[test_index,:]

            model.fit(x_cv, y_cv)
            predictions = model.predict(x_cv_test)
            rmse = m.sqrt(mean_squared_error(y_cv_test, predictions))

            rmse_lis.append(rmse)

        avg_cv_rsme = np.mean(rmse_lis)
        model_name = list(model.named_steps.keys())[1]
        model_evaluation_dict[model_name] = avg_cv_rsme

    return model_evaluation_dict


def prediction_actual_df(predictions, true_values):
    """
    Creates a two column DataFrame with the predicted values and actual value on the same row.
    """
    pred_real_dict = {"Predicted": [], "Actual": []}
    for x, i in enumerate(predictions):
        predicted = round(predictions[x], 2)
        actual = round(true_values.iloc[x, 0], 2)
        pred_real_dict["Predicted"].append(predicted)
        pred_real_dict["Actual"].append(actual)

    return pd.DataFrame(pred_real_dict)


if __name__ == "__main__":

    #===========================================================================
    #=========================== DATA CLEANING =================================
    #===========================================================================

    # sahie data set
    data_wd = "/Users/marsh/galvanize/dsi/projects/health_capstone/data/"

    sahie = pd.read_csv(data_wd + "health_insurance/SAHIE_31JAN17_13_18_47_11.csv")

    sahie = separate_states(sahie)

    useless_columns = ['Age Category','Income Category','Race Category','Sex Category','Demographic Group: MOE']
    sahie.drop(useless_columns, axis=1, inplace=True)

    to_object(sahie, ["Year",'ID'])
    to_float(sahie, ['Uninsured: Number',"Uninsured: MOE",'Insured: Number','Insured: MOE'])

    sahie_nans = count_nans(sahie)
    sahie = sahie.dropna(axis=0)


    # medicare data set
    medicare = pd.read_csv(data_wd + "/medicare_county_level/cleaned_medicare_county_all.csv")

    medicare.drop('unnamed:_0', axis=1, inplace=True)

    to_object(medicare, ['Unnamed: 0','state_and_county_fips_code'])

    medicare_nans = count_nans(medicare)

    medicare = medicare.dropna(axis=0)
    medicare['cost_per_beneficiary'] = medicare['total_actual_costs'] / medicare["beneficiaries_with_part_a_and_part_b"]

    #===========================================================================
    #=============== VISUALIZATION, EDA & HYPOTHESIS TESTING ===================
    #===========================================================================

    heatmap(sahie, "figures/heatmap")

    state_uninssured_descending = list(sahie.groupby("state").mean().sort_values('Uninsured: %', ascending=False).index)

    distribution_plot(sahie, "state", "Uninsured: %", "State", "Percentage (%) Uninsured", "Percentage Uninsured Across States", filename="figures/state_vs_uninsured", order=state_uninssured_descending)

    distribution_plot(sahie, 'state', 'Uninsured: %', 'State', "Percentage (%) Uninsured", "Ordered Percentage Uninsured Across States", plot_type="bar", filename="figures/state_vs_uninsured_bar", order=state_uninssured_descending)

    distribution_plot(sahie, "Year", "Uninsured: %", "Year", "Percentage (%) Uninsured", "Percentage Uninsured Across Years", plot_type="violin", filename="figures/year_vs_uninsured")

    # create csv for anova in R
    df = sahie[['state','Year','Uninsured: %']]
    df.to_csv("/Users/marsh/galvanize/dsi/projects/health_capstone/anova.csv")
    f, p = ANOVA(df, "Year", "Uninsured: %")

    #===========================================================================
    #=============================== MODELING ==================================
    #===========================================================================

    # select continuous attributes for modeling
    numerics = medicare[medicare.select_dtypes(exclude=['object']).columns]

    X = numerics.drop('cost_per_beneficiary', axis=1)
    y = pd.DataFrame(numerics['cost_per_beneficiary'])

    x_train, x_test, y_train, y_test = train_test_split(X, y)

    alphas = [100., 10., 5., 4., 3., 2.]

    cross_validated_alphas = tune_regularization_parameter(alphas)

    # Model Pipelines to test
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

    models = [lasso_pipeline, linear_pipeline, ridge_pipeline]

    cross_validation_results = cv_multiple_model_pipelines(models)

    #===========================================================================
    #================== MODEL EVALUATION: TEST SET =============================
    #===========================================================================

    preds = lasso_pipeline.predict(x_test)
    test_rsme = m.sqrt(mean_squared_error(y_test, preds))
    print(f"\nLasso Regression testing RMSE | {test_rsme}")

    predicted_actual_df = prediction_actual_df(preds, y_test)

    coef_dict = {"Attribute": [], "Coefficient": []}
    for x, i in enumerate(x_test.columns):
        coef_dict["Attribute"].append(x_test.columns[x])
        coef_dict["Coefficient"].append(lasso_pipeline.named_steps['lasso'].coef_[x])

    coef_df = pd.DataFrame(coef_dict)
    non_zero = coef_df['Coefficient'] != 0
    coef_df = coef_df[non_zero]
    coef_df = coef_df.sort_values('Coefficient', ascending=False)
