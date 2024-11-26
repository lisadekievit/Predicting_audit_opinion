import pandas as pd
import numpy as np
import wrds
from sklearn.model_selection import GroupShuffleSplit

db = wrds.Connection(wrds_username='lisadekievit')

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

df.shape
df.info()
df.head()

df['auop'].value_counts()
df['auop'].notna().sum()

df.to_csv('csv/df_eu.csv')
df = pd.read_csv('csv/df_eu.csv', index_col="Unnamed: 0")

df = df[~(df['auop'].isna())]

df['X140'].value_counts()

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

# Splitting the data 
groups = df['conm']
X = df.drop(columns=['auop'])
y = df['auop'] 
# Split into training and remaining (validation + test)
gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
train_idx, temp_idx = next(gss.split(X, y, groups=groups))
X_train, X_temp = X.iloc[train_idx], X.iloc[temp_idx]
y_train, y_temp = y.iloc[train_idx], y.iloc[temp_idx]
# Shuffle the training set while preserving the correspondence between X and y
X_train = X_train.sample(frac=1, random_state=42).reset_index(drop=True)
y_train = y_train.sample(frac=1, random_state=42).reset_index(drop=True)
# Further split the remaining data into validation and test sets
gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss.split(X_temp, y_temp, groups=X_temp['conm']))
X_val, X_test = X_temp.iloc[val_idx], X_temp.iloc[test_idx]
y_val, y_test = y_temp.iloc[val_idx], y_temp.iloc[test_idx]
# Shuffle the validation set
X_val = X_val.sample(frac=1, random_state=42).reset_index(drop=True)
y_val = y_val.sample(frac=1, random_state=42).reset_index(drop=True)


# Drop colums that er not usefull
to_be_dropped = ["conm"]
X_train_1 = X_train.drop(to_be_dropped, axis= 1)
X_val_1 = X_val.drop(to_be_dropped, axis= 1)
X_test_1 = X_test.drop(to_be_dropped, axis= 1)

X_train_1.to_csv('csv/CX_train_eu.csv')
X_val_1.to_csv('csv/CX_val_eu.csv')
X_test_1.to_csv('csv/CX_test_eu.csv')
y_train.to_csv('csv/Cy_train_eu.csv')
y_val.to_csv('csv/Cy_val_eu.csv')
y_test.to_csv('csv/Cy_test_eu.csv')
