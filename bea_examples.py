import os, dotenv;
import requests, json;
import numpy as np;
import pandas as pd;


#
# Load API Key from .env file
#

dotenv.load_dotenv();

bea_api_key = os.getenv('bea_api_key');

bea_endpoint = 'https://apps.bea.gov/api/data';


#
# Fetch data for CAINC4 (Personal income and employment by major component)
#

# UserID=Your-36Character-Key&amp;
# method=GetData&amp;datasetname=Regional&amp;
# TableName=CAINC4&amp;LineCode=30&amp;
# GeoFIPS=COUNTY&amp;
# Year=2013&amp;
# ResultFormat=json&amp;

r = requests.get(
	url=bea_endpoint,
	params={
		'UserID': bea_api_key,
		'method': 'GetData',
		'datasetname':'Regional',
		'TableName':'CAINC4',
		'LineCode':'30',
		'GeoFIPS':'COUNTY',
		'Year':'2013',
		'ResultFormat':'json'
	}
);

r_json = r.json()['BEAAPI'];

r_data_name = r_json['Results']['PublicTable'];

print('Data comes from table: ', r_data_name); # Data comes from table:  CAINC4 Personal income and employment by major component



r_data = pd.DataFrame(r_json['Results']['Data']);

# >>> r_data
#
#            Code GeoFips         GeoName TimePeriod  CL_UNIT UNIT_MULT DataValue NoteRef
# 0     CAINC4-30   01001     Autauga, AL       2013  Dollars         0     35492       4
# 1     CAINC4-30   01003     Baldwin, AL       2013  Dollars         0     38828       4
# 2     CAINC4-30   01005     Barbour, AL       2013  Dollars         0     29719       4
# 3     CAINC4-30   01007        Bibb, AL       2013  Dollars         0     27225       4
# 4     CAINC4-30   01009      Blount, AL       2013  Dollars         0     30222       4
# ...         ...     ...             ...        ...      ...       ...       ...     ...
# 3135  CAINC4-30   56037  Sweetwater, WY       2013  Dollars         0     48578       4
# 3136  CAINC4-30   56039       Teton, WY       2013  Dollars         0    177810       4
# 3137  CAINC4-30   56041       Uinta, WY       2013  Dollars         0     39430       4
# 3138  CAINC4-30   56043    Washakie, WY       2013  Dollars         0     42240       4
# 3139  CAINC4-30   56045      Weston, WY       2013  Dollars         0     44118       4

# [3140 rows x 8 columns]