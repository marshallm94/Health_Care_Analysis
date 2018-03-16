# Health Data

## Guiding Questions

1. Is there a statistically significant difference in the percentage of uninsured individuals across states?
    * regex on "Name" of SAHIE_31JAN17_13_18_47_11.csv file
    * ANOVA

2. Has there been a statistically significant change in the percentage of uninsured from year to year?
    * ANOVA

3. Can I predict total spending on medical expenses (aggregated across states)?
    * Many predictors across a few data sets ---> LASSO or Ridge Regression

## Workflow Timeline

### Tuesday, March 13, 2018
#### Data Cleaning

* Decide which data sets will best answer my questions (.xls files)
    * **complete**
        * *SAHIE_31JAN17_13_18_47_11.csv*
        * *medicare_spending_by_county/ -all*
        * *cleaned_medicare_county_all.csv*

* Put all desired data sets into .csv format and bring into Python
    * **complete**
        * *medicare_spending_by_county/ -all*

* On each data set, if not done already, create *State* column using regex on existing columns
    * **complete**

* Deal with NULLs and NaNs (impute or drop)
    * **complete**

### Wednesday, March 14, 2018
#### Visualization, EDA & Hypothesis Testing

* Create box plots of Percentage uninsured across States and Years.
    * Seaborn
    * **complete**

* Complete both Hypothesis Test's in Guiding Questions
    * ANOVA (Possible opportunity for writing to an R file)
    * **complete**

* Have a simple regression model complete by end of day, to be refined tomorrow
    * **complete**

### Thursday, March 15, 2018
#### Modeling

* Create modeling pipelines
    * **complete**
    * Ridge, Lasso and Linear Regression

* Finalize Presentation.md



Thanks to:
http://code.activestate.com/recipes/577305-python-dictionary-of-us-states-and-territories/ for the state-abbreviation dictionary.
