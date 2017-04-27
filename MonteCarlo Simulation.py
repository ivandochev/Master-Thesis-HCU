import csv
import operator
import random
import timeit
import pandas as pd
import numpy as np
from sets import Set
from itertools import repeat

#SET PARAMTERS
Iters = 500
OutPath =  "D:\HCU\Master_Thesis\MonteCarloSim\ResultsVerbrauch.csv"


tic=timeit.default_timer()

#OPEN AND READ

f = open('D:\HCU\Master_Thesis\MonteCarloSim\Gebaude_2017.csv', 'r+')
#BUILD:
#Cluster [0]
#IWU_Typ [1]
#NutzflEnEv [2]
#TypCountEp [3]
#Wohnfl [4]
#TypCountEpVerb [5]
#SanStufe [6]
f1 = open('D:\HCU\Master_Thesis\MonteCarloSim\EpassVerbrauch.csv', 'r+')
#DICT:
#IWU_Type_Unique_IndexPerType_Bedarf [0]
#HEBed_sqMNutFlEnEV [1]
#HEBedSan_sqMNutFlEnEV [2]

csv_ALKIS = csv.reader(f)
csv_Epass = csv.reader(f1)

#PREPARE FUNCTIONS
def BuildRoll(Build,dict):
    if "NN_" in Build[1]:
        return [Build[0], 0]
    else:
        '''For those in the ALKIS that do not have an epass
        so the count is 0'''
        if int(float(Build[5])) == 0:
            TIndex = 0
        else:
            TIndex = random.randint(1,int(float(Build[5]))) 
        
        BuildKey = "%s_%s" % (Build[1],TIndex)
        if Build[6] == '0':
            BuildSpezBedarf = int(float(dict[BuildKey][1]))
        else:
            BuildSpezBedarf = int(float(dict[BuildKey][2]))
        Flaeche = int(float(Build[4]))
        Bedarf = BuildSpezBedarf * Flaeche
        #print Build[6]
        #print BuildKey        
        #print [Build, Bedarf, BuildSpezBedarf]
        return [Build[0], Bedarf]

        
def MonteCarlo(BList, EpDict):
    d = []
    for i in range(len(BList)):
        result = BuildRoll(BList[i],EpDict)
        d.append(result)
    labels = ['Cl','kWh']
    df = pd.DataFrame.from_records(d, columns=['Cl','kWh'])
    grouped = df.groupby('Cl').sum()
    return grouped
   
  
#PREPARE DATA AND OUTPUT
BuildList = [b for b in csv_ALKIS]
EpassDict = {Epass[0]: Epass for Epass in csv_Epass}
Out = []
for b in BuildList:
    Out.append([b[0],0])
OutDF = pd.DataFrame.from_records(Out, columns=['Cl','kWh'])
ResultDF = OutDF.groupby('Cl').sum()

#ROLL MONTE CARLO
for i in range(Iters):
    Roll = MonteCarlo(BuildList, EpassDict)
    ResultDF = pd.concat([ResultDF, Roll], axis=1, join='inner')

#print ResultDF
#WRITE OUTPUT
ResultDF.to_csv(OutPath, sep=',')

#CLOSE FILES
f.close()
f1.close()

toc=timeit.default_timer()
print "CALCULATING Time Elapsed in mins:" + str((toc - tic)/60)
