#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import csv
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib
import libpysal
import segregation
import mapclassify
from segregation.local import MultiLocationQuotient, MultiLocalDiversity, MultiLocalEntropy, MultiLocalSimpsonInteraction, MultiLocalSimpsonConcentration, LocalRelativeCentralization

def cleanlink_wac(file,fips,shp):
    """Automates the process of cleaning and linking the LODES data. Takes file(string), state + county fips(string),
       and shp(string)as inputs. shp needs double / as it uses gpd instead of pd.
    """
    #read lodes data
    lehd_df = pd.read_csv(file,compression='gzip',header=0,sep=',',quotechar='"',error_bad_lines=False)
    #convert w_geocode to string to make it easier to check
    lehd_df['w_geocode'] = lehd_df['w_geocode'].astype(str)
    # check if beginning of string matches SD county fips code
    lehd_df = lehd_df[lehd_df['w_geocode'].str.match(fips)]
    #add leading 0 to make consistent with block data(below)
    lehd_df['w_geocode'] = '0' + lehd_df['w_geocode']
    lehd_df.rename(columns={'w_geocode': 'GEOID10'}, inplace=True)

    blocks = gpd.read_file(shp)
    blocks = blocks.merge(lehd_df, how='left', on='GEOID10')
    blocks.drop(['INTPTLON10','INTPTLAT10','COUNTYFP10','STATEFP10','NAME10','TRACTCE10','BLOCKCE10','MTFCC10','FUNCSTAT10','UR10','AWATER10','UACE10','UATYP10','createdate'], axis=1, inplace=True)
    blocks.fillna(0,inplace=True)
    #Remove any blocks where there is no land area, no jobs there so why would we care about them?
    blocks = blocks[blocks.ALAND10 != 0]
    return blocks

def generate_lists(dataframe):
    df_columns = dataframe.columns.tolist()
    workerage_list = df_columns[4:7]
    workerwage_list = df_columns[7:10]
    jobs_list = df_columns[10:30]
    race_list = df_columns[30:38]
    edu_list= df_columns[38:42]
    sex_list = df_columns[42:44]
    firmage_list = df_columns[44:49]
    firmsize_list = df_columns[49:54]
    return(workerage_list,workerwage_list,jobs_list,race_list,edu_list,sex_list,firmage_list,firmsize_list)

def gen_workerage(dataframe):
    df_columns = dataframe.columns.tolist()
    workerage_list = df_columns[4:7]
    return(workerage_list)

def gen_workerwage(dataframe):
    df_columns = dataframe.columns.tolist()
    workerwage_list = df_columns[7:10]
    return(workerwage_list)

def gen_jobs(dataframe):
    df_columns = dataframe.columns.tolist()
    jobs_list = df_columns[10:30]
    return(jobs_list)

def gen_race(dataframe):
    df_columns = dataframe.columns.tolist()
    race_list = df_columns[30:38]
    return(race_list)

def gen_edu(dataframe):
    df_columns = dataframe.columns.tolist()
    edu_list = df_columns[38:42]
    return(edu_list)

def gen_sex(dataframe):
    df_columns = dataframe.columns.tolist()
    sex_list = df_columns[42:44]
    return(sex_list)

def gen_firmage(dataframe):
    df_columns = dataframe.columns.tolist()
    firmage_list = df_columns[44:49]
    return(firmage_list)

def gen_firmsize(dataframe):
    df_columns = dataframe.columns.tolist()
    firmsize_list = df_columns[49:54]
    return(firmsize_list)

def calc_lq(dataframe):
    """
    #nested list for LQ for loop
    lehd_lists = [workerage_list,workerwage_list,jobs_list,race_list,edu_list,sex_list,firmage_list,firmsize_list]
    index_list = WorkerAgeLQIndex, WorkerWageLQIndex, JobsLQIndex, RaceLQIndex, EduLQIndex, SexLQIndex, FirmAgeLQIndex,FirmSizeLQIndex
    for_list =  zip(lehd_lists,index_list)

    for lehd_lists,index_lists in for_list:
    for items in for_list:
        items = MultiLocationQuotient(SDlehd_blocks, lehd_lists[lehd_lists])
        SDlehd_blocks['LQ_' + lehd_list[lehd_lists]] = items.statistics[:,lehd_lists]
    """

    df_columns = dataframe.columns.tolist()
    workerage_list = df_columns[4:7]
    workerwage_list = df_columns[7:10]
    jobs_list = df_columns[10:30]
    race_list = df_columns[30:38]
    edu_list= df_columns[38:42]
    sex_list = df_columns[42:44]
    firmage_list = df_columns[44:49]
    firmsize_list = df_columns[49:54]

    WorkerAgeLQIndex = MultiLocationQuotient(dataframe, workerage_list)
    WorkerWageLQIndex = MultiLocationQuotient(dataframe, workerwage_list)
    JobsLQIndex = MultiLocationQuotient(dataframe, jobs_list)
    RaceLQIndex = MultiLocationQuotient(dataframe, race_list)
    EduLQIndex = MultiLocationQuotient(dataframe, edu_list)
    SexLQIndex = MultiLocationQuotient(dataframe, sex_list)
    FirmAgeLQIndex = MultiLocationQuotient(dataframe, firmage_list)
    FirmSizeLQIndex = MultiLocationQuotient(dataframe, firmsize_list)


    for items in range(len(workerage_list)):
        dataframe['LQ_' + workerage_list[items]] = WorkerWageLQIndex.statistics[:,items]

    for items in range(len(workerwage_list)):
        dataframe['LQ_' + workerwage_list[items]] = WorkerWageLQIndex.statistics[:,items]

    for items in range(len(jobs_list)):
        dataframe['LQ_' + jobs_list[items]] = JobsLQIndex.statistics[:,items]

    for items in range(len(race_list)):
        dataframe['LQ_' + race_list[items]] = RaceLQIndex.statistics[:,items]
    
    for items in range(len(edu_list)):
        dataframe['LQ_' + edu_list[items]] = EduLQIndex.statistics[:,items]
    
    for items in range(len(sex_list)):
        dataframe['LQ_' + sex_list[items]] = SexLQIndex.statistics[:,items]
    
    for items in range(len(firmage_list)):
        dataframe['LQ_' + firmage_list[items]] = FirmAgeLQIndex.statistics[:,items]
    
    for items in range(len(firmsize_list)):
        dataframe['LQ_' + firmsize_list[items]] = FirmSizeLQIndex.statistics[:,items]

def graph_codes(dataframe,code,cmap='OrRd',k=5):
    """A function that takes the dataframe, LEHD code(string), cmap(string), and k(int) as arguements. Graphs both the original
    and the LQ of the code with arguements specified. Only necessary arguments are dataframe and code. """
    #Create a cmap that's more appropriate for displaying LQ
    from pysal.viz.splot._viz_utils import shift_colormap
    lqmap = shift_colormap('YlOrBr', midpoint=1.0)
    
    #Remove 0 in LQ; more buckets for other values
    col_list = dataframe.columns.tolist()
    lqlist = [lq for lq in col_list if 'LQ_' in lq]
    for lq in lqlist:
        dataframe[lq].replace(0,'NaN')
        
    #start graphing
    fig, axs = matplotlib.pyplot.subplots(1,2, figsize=(20,10))
    axs[0].set_title(code)
    axs[1].set_title('LQ_'+ code)
    return dataframe.plot(column= code, cmap=cmap, k=k, scheme='fisherjenkssampled', figsize = (10,10), legend=True, ax=axs[0]), dataframe.plot(column= 'LQ_' + code, cmap=lqmap, k=k, scheme='fisherjenkssampled', figsize = (10,10), legend=True, ax=axs[1])

