import sqlite3
import pandas as pd


con = sqlite3.connect('input/database.sqlite')

devices = pd.read_sql_query("select * from Player", con)
print(devices)