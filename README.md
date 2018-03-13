# Health Data

1. Is there a statistically significant difference in the percentage of uninsured individuals across states?
    * regex on "Name" of SAHIE_31JAN17_13_18_47_11.csv file
    * ANOVA

2. Has there been a statistically significant change in the percentage of uninsured from year to year?
    * ANOVA

3. Can I predict total spending on medical expenses (aggregated across states)?
    * Many predictors across a few data sets ---> LASSO or Ridge Regression

## Timeline

### Tuesday, March 13, 2018
#### Data Cleaning

* Decide which data sets will best answer my questions (.xls files)
    * cleaned_medicare_county_all.csv
    * SAHIE_31JAN17_13_18_47_11.csv
    * medicare_spending_by_county/*
    * /Users/marsh/galvanize/dsi/projects/health_capstone/data/medicare_county_level

* Put all desired data sets into .csv format and bring into Python
    * * medicare_spending_by_county/*
* On each data set, if not done already, create *State* column using regex on existing columns
* Deal with NULLs and NaNs (impute or drop)
