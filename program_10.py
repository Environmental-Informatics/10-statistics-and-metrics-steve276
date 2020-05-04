#!/bin/env python
# Created on March 25, 2020
#  by Keith Cherkauer
#
# This script serves as the solution set for assignment-10 on descriptive
# statistics and environmental informatics.  See the assignment documention 
# and repository at:
# https://github.com/Environmental-Informatics/assignment-10.git for more
# details about the assignment.
#
#Modified by Miriam Stevens
#May 04, 2020
#Modified from the template to calcualte monthly and annual stream flow metrics
#Calculates monthly and annual averages
#Saves annual and monthly metrics CSV files
#Saves annual and monthly averages to txt files
#
import pandas as pd
import scipy.stats as stats
import numpy as np

def ReadData( fileName ):
    """This function takes a filename as input, and returns a dataframe with
    raw data read from that file in a Pandas DataFrame.  The DataFrame index
    should be the year, month and day of the observation.  DataFrame headers
    should be "agency_cd", "site_no", "Date", "Discharge", "Quality". The 
    "Date" column should be used as the DataFrame index. The pandas read_csv
    function will automatically replace missing values with np.NaN, but needs
    help identifying other flags used by the USGS to indicate no data is 
    availabiel.  Function returns the completed DataFrame, and a dictionary 
    designed to contain all missing value counts that is initialized with
    days missing between the first and last date of the file."""
    
    # define column names
    colNames = ['agency_cd', 'site_no', 'Date', 'Discharge', 'Quality']

    # open and read the file
    DataDF = pd.read_csv(fileName, header=1, names=colNames,  
                         delimiter=r"\s+",parse_dates=[2], comment='#',
                         na_values=['Eqp'])
    
    DataDF = DataDF.set_index('Date')
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isna().sum()
    
    MissingValues_dict = {'Missing Values': MissingValues}
    
    # replace negative values with nan
    DataDF.loc[DataDF['Discharge'] < 0, 'Discharge'] = np.NaN
    
    return( DataDF, MissingValues )

def ClipData( DataDF, startDate, endDate ):
    """This function clips the given time series dataframe to a given range 
    of dates. Function returns the clipped dataframe and and the number of 
    missing values."""
    
    #clip data 
    DataDF = DataDF.loc[startDate:endDate]  
    
    # quantify the number of missing values
    MissingValues = DataDF["Discharge"].isnull().sum()
    
    return( DataDF, MissingValues )

def CalcTqmean(Qvalues):
    """This function computes the Tqmean of a series of data, typically
       a 1 year time series of streamflow, after filtering out NoData
       values.  Tqmean is the fraction of time that daily streamflow
       exceeds mean streamflow for each year. Tqmean is based on the
       duration rather than the volume of streamflow. The routine returns
       the Tqmean value for the given data array."""
    
    # count number of values greater than mean
    vals_gt_mean = (Qvalues > Qvalues.mean()).sum()

    Tqmean = vals_gt_mean / len(Qvalues)
    
    return ( Tqmean )

def CalcRBindex(Qvalues):
    """This function computes the Richards-Baker Flashiness Index
       (R-B Index) of an array of values, typically a 1 year time
       series of streamflow, after filtering out the NoData values.
       The index is calculated by dividing the sum of the absolute
       values of day-to-day changes in daily discharge volumes
       (pathlength) by total discharge volumes for each year. The
       routine returns the RBindex value for the given data array."""

    # absolute difference between values
    Dtd_abs_diff = Qvalues.diff().abs() 
    
    # sum of differences
    pathlength = Dtd_abs_diff.sum()
    
    # total annual discharge
    discharge_tot = Qvalues.sum()
    
    RBindex = pathlength/discharge_tot
    
    return ( RBindex )

def Calc7Q(Qvalues):
    """This function computes the seven day low flow of an array of 
       values, typically a 1 year time series of streamflow, after 
       filtering out the NoData values. The index is calculated by 
       computing a 7-day moving average for the annual dataset, and 
       picking the lowest average flow in any 7-day period during
       that year.  The routine returns the 7Q (7-day low flow) value
       for the given data array."""
     
    val7Q = Qvalues.rolling(7).mean().min()      
    
    return ( val7Q )

def CalcExceed3TimesMedian(Qvalues):
    """This function computes the number of days with flows greater 
       than 3 times the annual median flow. The index is calculated by 
       computing the median flow from the given dataset (or using the value
       provided) and then counting the number of days with flow greater than 
       3 times that value.   The routine returns the count of events greater 
       than 3 times the median annual flow value for the given data array."""
       
    median_flow = Qvalues.median()
    
    median3x = (Qvalues > 3*median_flow).sum()
    
    return ( median3x )

def GetAnnualStatistics(DataDF):
    """This function calculates annual descriptive statistcs and metrics for 
    the given streamflow time series.  Values are retuned as a dataframe of
    annual values for each water year.  Water year, as defined by the USGS,
    starts on October 1."""
    
    colNames = ['site_no','Mean Flow', 'Peak Flow', 'Median Flow', 'Coeff Var',\
                    'Skew', 'Tqmean', 'R-B Index', '7Q', '3xMedian']
    
    # define index as dates of ressmpled data. 
    # data is resampled annually with end of year in September
    annualIndex = DataDF.resample('A-SEP').mean().index
    
    # create empty dataframe
    WYDataDF = pd.DataFrame(data=0, index=annualIndex, columns=colNames)
    
    # resample data
    WYData = DataDF.resample('A-SEP')
    
    #add metrics to dataframe
    WYDataDF['site_no'] = WYData['site_no'].min()
    WYDataDF['Mean Flow'] = WYData['Discharge'].mean()
    WYDataDF['Peak Flow'] = WYData['Discharge'].max()
    WYDataDF['Median Flow'] = WYData['Discharge'].median()
    WYDataDF['Coeff Var'] = WYData['Discharge'].std() / WYData['Discharge'].mean()*100
    WYDataDF['Skew'] = WYData['Discharge'].skew()
    WYDataDF['Tqmean'] = WYData['Discharge'].apply(lambda x: CalcTqmean(x))
    WYDataDF['R-B Index'] = WYData['Discharge'].apply(lambda x: CalcRBindex(x))
    WYDataDF['7Q'] = WYData['Discharge'].apply(lambda x: Calc7Q(x))
    WYDataDF['3xMedian'] = WYData['Discharge'].apply(lambda x: CalcExceed3TimesMedian(x))
    
    return ( WYDataDF )

def GetMonthlyStatistics(DataDF):
    """This function calculates monthly descriptive statistics and metrics 
    for the given streamflow time series.  Values are returned as a dataframe
    of monthly values for each year."""
    
    colNames = ['site_no','Mean Flow', 'Coeff Var', 'Tqmean', 'R-B Index']
    
    # define index as dates of ressmpled data. 
    # data is resampled monthly
    monthlyIndex = DataDF.resample('MS').mean().index
    
    # create empty dataframe
    MoDataDF = pd.DataFrame(data=0, index=monthlyIndex, columns=colNames)
    
    # resample data
    MoData = DataDF.resample('MS')
    
    #add metrics to dataframe
    MoDataDF['site_no'] = MoData['site_no'].min()
    MoDataDF['Mean Flow'] = MoData['Discharge'].mean()
    MoDataDF['Coeff Var'] = MoData['Discharge'].std() / MoData['Discharge'].mean()*100
    MoDataDF['Tqmean'] = MoData['Discharge'].apply(lambda x: CalcTqmean(x))
    MoDataDF['R-B Index'] = MoData.apply({'Discharge': lambda x: CalcRBindex(x)})

    return ( MoDataDF )

def GetAnnualAverages(WYDataDF):
    """This function calculates annual average values for all statistics and
    metrics.  The routine returns an array of mean values for each metric
    in the original dataframe."""
    
    AnnualAverages = WYDataDF.mean()
    
    return( AnnualAverages )

def GetMonthlyAverages(MoDataDF):
    """This function calculates annual average monthly values for all 
    statistics and metrics.  The routine returns an array of mean values 
    for each metric in the original dataframe."""
    
    MonthlyAverages = MoDataDF.groupby(MoDataDF.index.month).mean()
    
    return( MonthlyAverages )

# the following condition checks whether we are running as a script, in which 
# case run the test code, otherwise functions are being imported so do not.
# put the main routines from your code after this conditional check.

if __name__ == '__main__':

    # define filenames as a dictionary
    # NOTE - you could include more than jsut the filename in a dictionary, 
    #  such as full name of the river or gaging site, units, etc. that would
    #  be used later in the program, like when plotting the data.
    fileName = { "Wildcat": "WildcatCreek_Discharge_03335000_19540601-20200315.txt",
                 "Tippe": "TippecanoeRiver_Discharge_03331500_19431001-20200315.txt" }
    
    # define blank dictionaries (these will use the same keys as fileName)
    DataDF = {}
    MissingValues = {}
    WYDataDF = {}
    MoDataDF = {}
    AnnualAverages = {}
    MonthlyAverages = {}
    
    # process input datasets
    for file in fileName.keys():
        
        print( "\n", "="*50, "\n  Working on {} \n".format(file), "="*50, "\n" )
        
        DataDF[file], MissingValues[file] = ReadData(fileName[file])
        print( "-"*50, "\n\nRaw data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # clip to consistent period
        DataDF[file], MissingValues[file] = ClipData( DataDF[file], '1969-10-01', '2019-09-30' )
        print( "-"*50, "\n\nSelected period data for {}...\n\n".format(file), DataDF[file].describe(), "\n\nMissing values: {}\n\n".format(MissingValues[file]))
        
        # calculate descriptive statistics for each water year
        WYDataDF[file] = GetAnnualStatistics(DataDF[file])
        
        # calcualte the annual average for each stistic or metric
        AnnualAverages[file] = GetAnnualAverages(WYDataDF[file])
        
        print("-"*50, "\n\nSummary of water year metrics...\n\n", WYDataDF[file].describe(), "\n\nAnnual water year averages...\n\n", AnnualAverages[file])

        # calculate descriptive statistics for each month
        MoDataDF[file] = GetMonthlyStatistics(DataDF[file])

        # calculate the annual averages for each statistics on a monthly basis
        MonthlyAverages[file] = GetMonthlyAverages(MoDataDF[file])
        
        print("-"*50, "\n\nSummary of monthly metrics...\n\n", MoDataDF[file].describe(), "\n\nAnnual Monthly Averages...\n\n", MonthlyAverages[file])
        
        
# save outputs
# annual metrics to csv

a_Wildcat = WYDataDF['Wildcat'] 
a_Wildcat['Station'] = 'Wildcat'

a_Tippe = WYDataDF['Tippe'] 
a_Tippe['Station'] = 'Tippe'

annualTable = a_Wildcat.append(a_Tippe)
annualTable.to_csv('Annual_Metrics.csv', sep = ',', index=True)
 
    

# monthly metrics to csv

m_Wildcat = MoDataDF['Wildcat'] 
m_Wildcat['Station'] = 'Wildcat'

m_Tippe = MoDataDF['Tippe'] 
m_Tippe['Station'] = 'Tippe'

m_Wildcat = m_Wildcat.append(m_Tippe)
m_Wildcat.to_csv('Monthly_Metrics.csv', sep = ',', index=True)



# annual averages to txt

aav_Wildcat = AnnualAverages['Wildcat'] 
aav_Wildcat['Station'] = 'Wildcat'

aav_Tippe = AnnualAverages['Tippe'] 
aav_Tippe['Station'] = 'Tippe'

aav_Wildcat = aav_Wildcat.append(aav_Tippe)
aav_Wildcat.to_csv('Average_Annual_Metrics.txt', sep = '\t', index=True)



# monthly metrics to txt

mav_Wildcat = MonthlyAverages['Wildcat'] 
mav_Wildcat['Station'] = 'Wildcat'

mav_Tippe = MonthlyAverages['Tippe'] 
mav_Tippe['Station'] = 'Tippe'

mav_Wildcat = mav_Wildcat.append(mav_Tippe)
mav_Wildcat.to_csv('Average_Monthly_Metrics.txt', sep = '\t', index=True)


