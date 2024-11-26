import pandas as pd
import numpy as np
import wrds

import capu as k
print(dir(k))

db = wrds.Connection(wrds_username='lisadekievit')
print("Connection set up")

df = db.raw_sql("""select datadate, fyear, conm, isin, curcd, act, ap, aqc, at, caps, ceq, ch, che, ibc, cogs, dd1, dlc, dltt, dpact, dv, dvt, ebit, ebitda, emp, epsincon, epsexnc, ninc, gdwl, sppch, invt, lct, lt, nicon, oancf, ppent, pstk, re, rect, revt, sale, txt, wcap, xint, xopr, xsga, naicsh, cshoi, cshpria, au, auop
                   from comp.g_funda
                   where datadate >= '01/01/2016'
                    AND datadate < '01/01/2024'
                    AND auop IS NOT NULL
                    AND auop != '0'
                    AND auop != '2'
                    AND auop != '3'
                    AND auop != '5'
                    AND curcd = 'EUR'
                   """, 
                   date_cols=['datadate'])
print("Data has been loaded in")
"""
df.shape
df.info()
df.head()

df['auop'].value_counts()
df['auop'].notna().sum()

df.to_csv('csv/df_eu.csv')"""


df = pd.read_csv('csv/df_eu.csv', index_col="Unnamed: 0")

df = df[~(df['auop'].isna())]


df['Country'] = df['isin'].str[:2]

df['X1'] = df["nicon"]/ df['at'] 
df['X2'] = df["nicon"]/ df['ceq']
df['X3'] = df["ebit"]/ df['sale']

df['X5'] = df['nicon']/ df['sale']
df['X6'] = df['re']/ df['at']
df['X7'] = df['ch']/ df['dlc']
df['X8'] = df['che']/ df['dlc']
df['X9'] = df['act']/ df['dlc']
df['X10'] = df['wcap']/ df['at']
df['X11'] = df['ebitda']
df['X12'] = df['ebitda']/ df['sale']
df['X13'] = df['sale']/ df['che']
df['X14'] = df['lt']/ df['at']
df['X15'] = df['cogs']/ df['invt']
df['X16'] = df['rect']/ df['at']
df['X17'] = df['rect']/ df['sale']
df['X20'] = df['cshpria']/ df['epsexnc']

# big 4 X23 BINARY and X140 
big_4_codes = ["4", "5", "6", "7"]
df['X23'] = df['au'].apply(lambda x: 1 if x in big_4_codes else 0)

df['X140'] = df['naicsh'].fillna('00').astype(str).str[:2]

# X31, X32
df['X31'] = np.where(df['at'] > 0.1, np.log(df['at']), np.nan)
df['X32'] = np.where(df['sale'] > 0.1, np.log(df['sale']), np.nan)

# X43 tm X49, X51 tm X57
df['X43'] = df['nicon'] / df['revt']
df['X44'] = df['oancf'] / df['at']
df['X45'] = df['xsga']
df['X46'] = df['revt'] / df['ap']
df['X47'] = df['revt'] / df['lt']
df['X48'] = df['oancf'] / df['ppent']
df['X49'] = df['oancf'] / df['xint']
df['X51'] = df['lct'] / df['ppent']
df['X52'] = df['xint'] / df['lct']
df['X53'] = df['che'] / df['ppent'] 

df['X55'] = df['che'] / df['revt']
df['X56'] = df['cogs'] / df['rect']
df['X57'] = df['ap'] / df['act']

df['X61'] = (df['revt'] - df['cogs']) / df['revt']
df['X62'] = df['act'] / df['lct']
df['X63'] = (df['act'] - df['invt']) / df['lct']
df['X64'] = (df['dlc'] + df['dltt']) / df['at']
df['X67'] = df['cshpria'] / df['ceq']
df['X69'] = df['oancf'] / df['revt']
df['X70'] = df['oancf'] / (df['dlc'] + df['dltt'])
df['X72'] = df['che'] / df['lt']
df['X73'] = (df['cshpria'] * df['cshoi']) + df['dlc'] + df['dltt'] - df['che']
df['X74'] = df['X73'] / df['ebitda']

# X81, X86 tm X88
df['X81'] = df['dd1']
df['X86'] = df['dvt']
df['X87'] = df['gdwl']

# X90 - 
df['X90'] = df['aqc']
df['X91'] = df['caps']
df['X92'] = df['ibc']
df['X93'] = df['dpact']
df['X94'] = df['dltt']
df['X95'] = df['dv']
df['X96'] = df['emp']
df['X97'] = df['epsincon']
df['X98'] = df['nicon']
df['X99'] = df['pstk']
df['X100'] = df['txt']
df['X101'] = df['xopr']
df['X102'] = df['cshoi']

# 28.10
df['X106'] = df['fyear'] - 2010
df['X107'] = df['act']
df['X108'] = df['ap']
df['X109'] = df['at']
df['X110'] = df['ceq']
df['X111'] = df['ch']
df['X112'] = df['che']
df['X113'] = df['cogs']
df['X114'] = df['cshoi']
df['X115'] = df['dlc']
df['X116'] = df['dltt']
df['X117'] = df['ebit']

df['X119'] = df['epsexnc']
df['X120'] = df['invt']
df['X121'] = df['lct']
df['X122'] = df['lt']
df['X123'] = df['nicon'] 
df['X124'] = df['oancf']
df['X125'] = df['ppent']
df['X126'] = df['re']
df['X127'] = df['rect']
df['X128'] = df['revt']
df['X129'] = df['sale']
df['X130'] = df['wcap']
df['X131'] = df['xint']
df['X132'] = df['xsga']
df['X133'] = df['cshpria']

# delete columns
df = df.drop(["nicon", 'isin', 'ceq', 'at', 'act', 'ebit', 'ebitda', 'sale',
              're', 'ch', 'che', 'dlc', 'wcap', 'lt', 'cogs', 'invt', 
              'rect', "epsexnc", 'cshpria', 'au', "ap", 'lct', 'oancf', 'ppent', 
              'revt', 'xint', 'xsga', 'dltt','cshoi','dd1',
              'aqc','caps','ibc','dpact','dv','emp','epsincon','nicon','pstk','txt','xopr','cshoi',
              'dvt','gdwl', 
              'ninc', 'sppch', 'naicsh', 'curcd', 'fyear', 'datadate'], axis= 1)

df.info()





# EDA > https://www.kaggle.com/code/shaunalexander/tps-eda-tab-net-w-optuna
print("The dataframe has {} rows and {} columns".format(df.shape[0],df.shape[1]))

# Calculate the correlation matrix for numeric columns
nm_cols = df.select_dtypes(include='float64').columns
correlation_matrix = df[nm_cols].corr()
print("Correlation Matrix:")
print(correlation_matrix)

# Set the threshold for "high" correlation
threshold = 0.7
high_corr = correlation_matrix[(correlation_matrix.abs() > threshold) & (correlation_matrix != 1)]
high_corr = high_corr.dropna(how='all').dropna(axis=1, how='all')
# Convert the correlation matrix to a tidy DataFrame of variable pairs and their correlation
high_corr_pairs = high_corr.stack().reset_index()
high_corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']
high_corr_pairs = high_corr_pairs[high_corr_pairs['Variable 1'] < high_corr_pairs['Variable 2']]
print("Unique High Correlation Pairs:")
#high_corr_pairs.to_excel('results/241106 EDA/cor.xlsx')

# Calculate the correlation between numeric columns and 'auop'
nm_cols = df.select_dtypes(include='float64').columns
correlation_with_auop = df[nm_cols].corrwith(df['auop']).abs()
#correlation_with_auop.to_clipboard()
correlation_with_auop.sort_values()

# Filter for high correlations with the target
threshold = 0.7
high_corr_with_auop = correlation_with_auop[correlation_with_auop > threshold]
# Convert to a DataFrame for readability
high_corr_pairs_with_auop = high_corr_with_auop.reset_index()
high_corr_pairs_with_auop.columns = ['Variable', 'Correlation with auop']
print("High Correlation with Target Variable 'auop':")
print(high_corr_pairs_with_auop)

from scipy.stats import chi2_contingency

# Function to calculate Cramér's V
def cramers_v(x, y):
    contingency_table = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

# Calculate Cramér's V for X23 and auop
cramers_v_x23_auop = cramers_v(df['X23'], df['auop'])
print(f"Cramér's V between X23 and auop: {cramers_v_x23_auop}")

# Calculate Cramér's V for X140 and auop
cramers_v_x140_auop = cramers_v(df['X140'], df['auop'])
print(f"Cramér's V between X140 and auop: {cramers_v_x140_auop}")

# Calculate Cramér's V for X140 and auop
cramers_v_x140_auop = cramers_v(df['Country'], df['auop'])
print(f"Cramér's V between X140 and auop: {cramers_v_x140_auop}")

print(df['auop'].value_counts())


# Create a cross-tabulation table for X140 and auop
distribution_table = pd.crosstab(df['X140'], df['auop'], normalize='index') * 100
distribution_table.columns = ['auop=4 (%)', 'auop=1 (%)']
# Add the counts of each category in X140 for reference
distribution_table['Count'] = df['X140'].value_counts()
print("Distribution of 'auop' per category of 'X140':")
print(distribution_table)

# Create a cross-tabulation table for 'Country' and auop
distribution_table = pd.crosstab(df['Country'], df['auop'], normalize='index') * 100
distribution_table.columns = ['auop=4 (%)', 'auop=1 (%)']
# Add the counts of each category in X140 for reference
distribution_table['Count'] = df['Country'].value_counts()
print("Distribution of 'auop' per category of 'Country':")
print(distribution_table)

import pandas as pd
import matplotlib.pyplot as plt

# Get the 10 largest countries by count
top_countries = df['Country'].value_counts().head(5).index

# Create a new column to group smaller countries under "Other"
df['Country_Grouped'] = df['Country'].where(df['Country'].isin(top_countries), 'Other')

# Group by 'Country_Grouped' and calculate the distribution of 'X114'
distribution = df.groupby('Country_Grouped')['X114'].value_counts(normalize=True).unstack()

# Plotting the distribution and saving it directly to a file
plt.figure(figsize=(12, 6))
distribution.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Distribution of X114 for Top 5 Countries and Others')
plt.xlabel('Country')
plt.ylabel('Proportion')
plt.legend(title='X114', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save the plot
plt.savefig('results/241120/Plots/241122/distribution_X114_Top5_Countries.png', dpi=300, bbox_inches='tight')
plt.close()



# Group by 'Country' and calculate statistics for 'X114'
statistics = (
    df.groupby('Country')['X114']
    .agg(
        Mean='mean',
        Standard_Deviation='std',
        Count='size',
        Mode=lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
)

# Reset the index for a cleaner display
statistics.reset_index(inplace=True)
# Save the table to a CSV file for reference
statistics.to_csv('results/241120/Plots/241122/X114_statistics_per_country.csv', index=False)
# Display the table
print(statistics)

# Group by 'Country' and calculate statistics for 'X114'
statistics = (
    df.groupby('Country')['X10']
    .agg(
        Mean='mean',
        Standard_Deviation='std',
        Count='size',
        Mode=lambda x: x.mode().iloc[0] if not x.mode().empty else None
    )
)

# Reset the index for a cleaner display
statistics.reset_index(inplace=True)
# Save the table to a CSV file for reference
#statistics.to_csv('results/241120/Plots/241122/X114_statistics_per_country.csv', index=False)
# Display the table
print(statistics)

# Sort statistics by the 'Count' column in ascending order
sorted_statistics = statistics.sort_values(by='Count', ascending=False)
# Display the sorted table
print(sorted_statistics)