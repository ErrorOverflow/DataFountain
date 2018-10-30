import numpy
import pandas

file = pandas.read_csv('C:\\Users\\WML\\Documents\\train_2_dup_del.csv')
s = pandas.Series(file['current_service'], dtype='category')

print(s.count())
