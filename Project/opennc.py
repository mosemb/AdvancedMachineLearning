import netCDF4 as nc
fn ="/home/mose/Documents/Semester4/AdvancedMachineLearning/DataMl/abadan/10minutely/wdiragl38S1/wdiragl38S1_200908.nc"
ds = nc.Dataset(fn)

print (ds)


#print(ds.__dict__)


#print(ds.__dict__['start_year'])

#for dim in ds.dimensions.values():  
 #   print(dim)

print(ds.dimensions['time'])

#for var in ds.variables.values():
 #   print(var)

print(ds['time'])


