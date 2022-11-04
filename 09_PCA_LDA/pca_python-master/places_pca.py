#!/usr/bin/python
#==========================================================
# A program to perform PCA on places data
# Bilal Nizami
# UKZN, Durban, 2016
#==========================================================

## step by step PCA 
from __future__ import print_function
import numpy as np
import numpy.ma as ma
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.mlab import PCA
from sklearn.preprocessing import scale
#from sklearn.decomposition import PCA as sklearnPCA
import argparse
from math import exp

#Header and places titles
head = ['Climate', 'HousingCost', 'HlthCare', 'Crime', 'Transp', 'Educ', 'Arts', 'Recreat', 'Econ', 'CaseNum', 'Long', 'Lat', 'Pop', 'StNum']
places = ('Abilene,TX', 'Akron,OH', 'Albany,GA', 'Albany-Schenectady-Troy,NY',\
		'Albuquerque,NM', 'Alexandria,LA', 'Allentown,Bethlehem,PA-NJ', \
		'Alton,Granite-City,IL', 'Altoona,PA', 'Amarillo,TX', 'Anaheim-Santa-Ana,CA', \
		'Anchorage,AK', 'Anderson,IN', 'Anderson,SC', 'Ann-Arbor,MI', 'Anniston,AL', \
		'Appleton-Oshkosh-Neenah,WI', 'Asheville,NC', 'Athens,GA', 'Atlanta,GA', \
		'Atlantic-City,NJ', 'Augusta,GA-SC', 'Aurora-Elgin,IL', 'Austin,TX', \
		'Bakersfield,CA', 'Baltimore,MD', 'Bangor,ME', 'Baton-Rouge,LA', 'Battle-Creek,MI', \
		'Beaumont-Port-Arthur,TX', 'Beaver-County,PA', 'Bellingham,WA', 'Benton-Harbor,MI', \
		'Bergen-Passaic,NJ', 'Billings,MT', 'Biloxi-Gulfport,MS', 'Binghampton,NY', 'Birmingham,AL', \
		'Bismarck,ND', 'Bloomington,IN', 'Bloomington-Normal,IL', 'Boise-City,ID', 'Boston,MA', \
		'Boulder-Longmont,CO', 'Bradenton,FL', 'Brazoria,TX', 'Bremerton,WA', \
		'Bridgeport-Milford,CT', 'Bristol,CT', 'Brockton,MA', 'Brownsville-Harlington,TX', \
		'Bryan-College-Station,TX', 'Buffalo,NY', 'Burlington,NC', 'Burlington,VT', 'Canton,OH', \
		'Casper,WY', 'Cedar-Rapids,IA', 'Champaign-Urbana-Rantoul,IL', 'Charleston,SC', \
		'Charleston,WV', 'Charlotte-Gastonia-Rock-Hill,NC-SC', 'Charlottesville,VA', \
		'Chattanooga,TN-GA', 'Chicago,IL', 'Chico,CA', 'Cincinnati,OH-KY-IN', \
		'Clarksville-Hopkinsville,TN-KY', 'Cleveland,OH', 'Colorado-Springs,CO', \
		'Columbia,MO', 'Columbia,SC', 'Columbus,GA-AL', 'Columbus,OH', 'Corpus-Christi,TX', \
		'Cumberland,MD-WV', 'Dallas,TX', 'Danbury,CT', 'Danville,VA', \
		'Davenport-Rock-Island-Moline,IA-IL', 'Dayton-Springfield,OH', \
		'Daytona-Beach,FL', 'Decatur,IL', 'Denver,CO', 'Des-Moines,IA', 'Detroit,MI', \
		'Dothgan,AL', 'Dubuque,IA', 'Duluth,MN-WI', 'East-St.-Louis-Belleville,IL', \
		'Eau-Claire,WI', 'El-Paso,TX', 'Elkhart-Goshen,IN', 'Elmira,NY', 'Enid,OK', 'Erie,PA', \
		'Eugene-Springfield,OR', 'Evansville,IN-KY', 'Fall-River,MA-RI', 'Fargo-Moorhead,ND-MN', \
		'Fayetteville,NC', 'Fayettteville-Sprindale,AR', 'Fitchburg-Leominster,MA', 'Flint,MI', \
		'Florence,AL', 'Florence,SC', 'Fort-Collins-Lover=land,CO', \
		'Fort-Lauderdale-Hollywood-Pompano-Beach,FL', 'Fort-Myers,FL', 'Fort-Pierce,FL', \
		'Fort-Smith,AR-OK', 'Fort-Walton-Beach,FL', 'Fort-Wayne,IN', 'Forth-Arlington,TX', \
		'Fresno,CA', 'Gadsden,AL', 'Gainesville,FL', 'Galveston-Texas-City,TX', 'Gary-Hammond,IN', \
		'Glens-Falls,NY', 'Grand-Forks,ND', 'Grand-Rapids,MI', 'Great-Falls,MT', 'Greeley,CO', \
		'Green-Bay,WI', 'Greensboro-Winston-Salem-High-Point,NC', 'Greenville-Spartanburg,SC', \
		'Hagerstown,MD', 'Hamilton-Middletown,OH', 'Harrisburg-Lebanon-Carlisle,PA', \
		'Hartford,CT', 'Hickory,NC', 'Honolulu,HI', 'Houma-Thibodaux,LA', 'Houston,TX', \
		'Huntington-Ashland,WV-KY-OH', 'Huntsville,AL', 'Indianapolis,IN', 'Iowa-City,IA', \
		'Jackson,MI', 'Jackson,MS', 'Jacksonville,FL', 'Jacksonville,NC', 'Janesville-Beloit,WI', \
		'Jersey-City,NJ', 'Johnson-City-Kingsport-Bristol,TN-VA', 'Johnstown,PA', 'Joliet,IL', \
		'Joplin,MO', 'Kalamazoo,MI', 'Kankakee,IL', 'Kansas-City,KS', 'Kansas-City,MO', \
		'Kenosha,WI', 'Kileen-Temple,TX', 'Knoxville,TN', 'Kokomo,IN', 'La-Crosse,WI', \
		'Lafayette,IN', 'Lafayette,LA', 'Lake-Charles,LA', 'Lake-County,IL', \
		'Lakeland-Winter-Haven,FL', 'Lancaster,PA', 'Lansing-East-Lansing,MI', \
		'Laredo,TX', 'Las-Cruces,NM', 'Las-Vegas,NV', 'Lawrence,KS', \
		'Lawrence-Haverhill,MA-NH', 'Lawton,OK', 'Lewiston-Auburn,ME', \
		'Lexington-Fayette,KY', 'Lima,OH', 'Lincoln,NE', 'Little-Rock,North-Little-Rock,AR', \
		'Longview-Marshall,TX', 'Lorain-Elyria,OH', 'Los-Angeles,Long-Beach,CA', \
		'Louisville,KY-IN', 'Lowell,MA-NH', 'Lubbock,TX', 'Lynchburg,VA', \
		'Macon,Warner-Robbins,GA', 'Madison,WI', 'Manchester,NH', 'Mansfield,OH', \
		'McAllen-Edinburg-Mission,TX', 'Medford,OR', 'Melbourne-Titusville-Palm-Bay,FL', \
		'Memphis,TN-AR-MS', 'Miami-Hialeah,FL', 'Middlesex-Somerset,Hunterdon,NJ', \
		'Middletown,CT', 'Midland,TX', 'Milwaukee,WI', 'Minneapolis-St.-Paul,MN-WI', \
		'Mobile,AL', 'Modesto,CA', 'Monmouth-Ocean,NJ', 'Monroe,LA', 'Montgomery,AL', \
		'Muncie,IN', 'Muskegon,MI', 'Nashua,NH', 'Nashville,TN', 'Nassua-Suffolk,NY', \
		'New-Bedsford,MA', 'New-Britain,CT', 'New-Haven-Meriden,CT', \
		'New-London-Norwich,CT-RI', 'New-Orleans,LA', 'New-York,NY', \
		'Newark,NJ', 'Niagara-Falls,NY', 'Norfolk-Virginia-Beach-Newport-News,VA', \
		'Norwalk,CT', 'Oakland,CA', 'Ocala,FL', 'Odessa,TX', 'Oklahoma-City,OK', 'Olympia,WA', \
		'Omaha,NE-IA', 'Orange-County,NY', 'Orlando,FL', 'Owensboro,KY', 'Oxnard-Ventura,CA', \
		'Panama-City,FL', 'Parkerburg-Marietta,WV-OH', 'Pascagoula,MS', \
		'Pawtucket-Woonsocket-Attleboro,RI-MA', 'Pensacola,FL', 'Peoria,IL', \
		'Philadelphia,PA-NJ', 'Phoenix,AZ', 'Pine-Bluff,AR', 'Pittsburgh,PA', \
		'Pittsfield,MA', 'Portland,ME', 'Portland,OR', 'Portsmouth-Dover-Rochester,NH-ME', \
		'Poughkeepsie,NY', 'Providence,RI', 'Provo-Orem,UT', 'Pueblo,CO', 'Racine,WI', \
		'Raleigh-Durham,NC', 'Reading,PA', 'Redding,CA', 'Reno,NV', 'Richland-Kinnewick-Pasco,WA', \
		'Richmond-Petersburg,VA', 'Riverside-San-Bernardino,CA', 'Roanoke,VA', 'Rochester,MN', \
		'Rochester,NY', 'Rockford,IL', 'Sacramento,CA', 'Saginaw-Bay-City-Midland,MI', \
		'St.-Cloud,MN', 'St.-Joseph,MO', 'St.-Louis,MO-IL', 'Salem,OR', 'Salem-Glouster,MA', \
		'Salinas-Seaside-Monterey,CA', 'Salt-Lake-City-Ogden,UT', 'San-Angelo,TX', \
		'San-Antonio,TX', 'San-Diego,CA', 'San-Francisco,CA', 'San-Jose,CA', \
		'Santa-Barbara-Santa-Maria-Lompoc,CA', 'Santa-Cruz,CA', 'Santa-Rosa-Petaluma,CA', \
		'Sarasota,FL', 'Savannah,GA', 'Scranton-Wilkes-Barre,PA', 'Seattle,WA', 'Sharon,PA', \
		'Sheboygan,WI', 'Sherman-Denison,TX', 'Shreveport,LA', 'Sioux-City,IA-NE', 'Sioux-Falls,SD', \
		'South-Bend-Mishawaka,IN', 'Spokane,WA', 'Springfield,IL', 'Springfield,MA', 'Springfield,MO', \
		'Stamford,CT', 'State-College,PA', 'Steubenville-Weirton,OH-WV', 'Stockton,CA', 'Syracuse,NY', \
		'Tacoma,WA', 'Tallahassee,FL', 'Tampa-St.-Petersburg-Clearwater,FL', 'Terre-Haute,IN', \
		'Texarkana,TX-Texarkana,AR', 'Toledo,OH', 'Topeka,KS', 'Trenton,NJ', 'Tuscon,AZ', 'Tulsa,OK', \
		'Tuscaloosa,AL', 'Tyler,TX', 'Utica-Rome,NY', 'Vallejo-Fairfield-Napa,CA', 'Vancouver,WA', \
		'Victoria,TX', 'Vineland-Millville-Bridgeton,NJ', 'Visalia-Tulare-Porterville,CA', \
		'Waco,TX', 'Washington,DC-MD-VA', 'Materbury,CT', 'Waterloo-Cedar-Falls,IA', 'Wausau,WI', \
		'West-Palm-Beach-Boca-Raton-Delray-Beach,FL', 'Wheeling,WV-OH', 'Wichita,KS', \
		'Wichita-Falls,TX', 'Williamsport,PA', 'Wilmington,DE-NJ-MD', 'Wilmington,NC', \
		'Worcester,MA', 'Yakima,WA', 'York,PA', 'Youngstown-Warren,OH', 'Yuba-City,CA')
#============================
###argument passing 
parser = argparse.ArgumentParser(description='argument')
parser.add_argument('-m', type=str, nargs=1, help='data transformation method', dest='t_method',\
					choices=['log', 'unit'])
parser.add_argument('-c', type=int, nargs='+', help='coloumn number', dest='col')
parser.add_argument('-e', type=int, nargs=1, help='number of eigenvector',dest='user_eigen')
args = parser.parse_args()

col = args.col
user_eigen = args.user_eigen
t_method = args.t_method

## Read data file 
arr = []
fp = open ("places_number.txt","r")
#read line into array 
for line in fp.readlines():
    # add a new sublist
    arr.append([])
    # loop over the elemets, split by whitespace
    for i in line.split():
        # convert to integer and append to the last
        # element of the list
		arr[-1].append(float(i))
fp.close()

arr = np.array(arr)
## select user defined coloumns
##choosen coloumn 
if args.col != None:
	print ('choosen coloumns are:')
	for i in col:
		print (head[i])
else:
	print ('all the data coloumns will be used')

arr = arr[:,col]
arr1 = np.array(arr) # non transformed input data


# log or unit variance scale transformation
print ('data transformation method:')
print (t_method[0])
if t_method[0] == "log":
	arr_log = np.zeros((329,len(col)))
	for i in range(len(arr)):
		arr_log[i,:] = np.log10(arr[i,:]) 
	# check for NaNs in matrix and set it to 0 (due to log of negative numbers)	
	where_nans = np.isnan(arr_log)
	arr_log[where_nans] = 0
	arr = np.array(arr_log)
elif t_method[0] == 'unit':
	# scale to unit variance 
	arr_scale = []
	arr_scale = scale(arr, axis=0, with_mean=True, with_std=True, copy=True)
	arr = arr_scale	
	

#print (arr)

#++++++++++++++++++++++=
# plot 3 dimentions of the matrix, 
# just change the index of vectors to plot different dimentions

#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(111)
#ax.plot(arr[:,0],arr[:,3], marker='o', linestyle='None')
#plt.show()

#place_pca = PCA(arr)
#print(place_pca.fracs)
#print (place_pca.Y)
#plt.plot(place_pca.Y[:,0],place_pca.Y[:,3], marker='^', linestyle='None')

#plt.plot(place_pca.project(arr[1,:]))
#plt.show()

#=====================================================
# computing the mean vectors for headers. needed for scatter matrix
mean_cli = np.mean(arr[:,0])
mean_hou = np.mean(arr[:,1])
mean_heal = np.mean(arr[:,2])
mean_crime = np.mean(arr[:,3])
mean_trans = np.mean(arr[:,4])
mean_edu = np.mean(arr[:,5])
mean_art = np.mean(arr[:,6])
mean_rec = np.mean(arr[:,7])
mean_econ = np.mean(arr[:,8])
#mean_casen = np.mean(arr[:,9])
#mean_long = np.mean(arr[:,10])
#mean_lat = np.mean(arr[:,11])
#mean_pop = np.mean(arr[:,12])
#mean_stn = np.mean(arr[:,13])

#print (arr.mean(0))

#===============================================
# covariance matrix of selected coloumns
cov_mat = np.cov(arr, rowvar=False)
#cov_mat = np.cov([arr[:,0],arr[:,1],arr[:,2],arr[:,3],arr[:,4],arr[:,5],arr[:,6],arr[:,7],arr[:,8]])

#===================================================
# eigenvector and eigenvalues

arr_eval, arr_evec = np.linalg.eig(cov_mat)

#=============================
# sanity check of calculated eigenvector and eigen values 
# it must be cov matrix * eigen vector = eigen vector * eigen value
for i in range(len(arr_eval)):
		eigv = arr_evec[:,i].reshape(1,len(col)).T
		np.testing.assert_array_almost_equal(cov_mat.dot(eigv), arr_eval[i]*eigv, decimal=3, err_msg='', verbose=True)

#=============================================
# sort the eigenvales
e_p = []
for i in range(len(arr_eval)):
	eig_pairs = [np.abs(arr_eval[i]), arr_evec[:,i]]
	e_p.append(eig_pairs)
#print (e_p[2])
e_p.sort(key=lambda x: x[0], reverse=True)

# sorted eigenvalues and variation explained
print ('sorted eigenvalues')
tot_var = 0
for i in e_p:
	tot_var +=i[0]
variation = []
cum = []
j = 0
eigv = []
for i in e_p:
	print (i[0])
	eigv.append(i[0])
	variation.append(i[0]/tot_var)
	print ("variation explained:",variation[j]*100)
	cum = np.cumsum(variation)
	print ('cumulative: ', cum[j]*100 )
	j +=1
	
## scree plot of variation 
plt.plot(cum, eigv, marker = 'o')
plt.xlabel('Cumulative variation')
plt.ylabel('Eigenvalue')
plt.title('Scree plot')
plt.show()

# Choose PC (take top two eigenvalues)
 
mat_w = np.hstack((e_p[0][1].reshape(len(col),1), e_p[1][1].reshape(len(col),1)))
#print (mat_w.T)
#print (np.shape(arr))

#=======================================================
#principal component scores/ Z scores
eigenvec = []
eigenvec = e_p[0][1]
j = 0
z_score = []

for i in eigenvec:
	col_mean = pow(10,arr[:,j]).mean(axis=0)
	#print (col_mean)
	print (i)
	#print ((col_mean-pow(10,arr[:,j])))
	#print (i*(col_mean-pow(10,arr[:,j])))
	z_score.append(i*(col_mean-pow(10,arr[:,j])))
	j+=1
z_score = np.array(z_score)
print (z_score)
#print (np.sum(z_score, axis=1))
#print (sum(e_p[0][1].reshape(len(col),1)*arr[:,0]))
#z_score = 



#========================================================
# transform the input data into choosen pc
arr_transformed = mat_w.T.dot(arr.T)
#print (np.shape(arr_transformed))
#print (arr_transformed)

plt.plot (arr_transformed[0,:], arr_transformed[1,:], marker = 'o', linestyle='None')
plt.title('PCA projection')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

### pca by numpy in built method

#mlab_pca = PCA(arr1)
#print(mlab_pca.Y)	
#print('PC axes in terms of the measurement axes scaled by the standard deviations:\n', mlab_pca.Wt)
#plt.plot(mlab_pca.Y[0:329,0],mlab_pca.Y[0:329,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
#plt.plot(mlab_pca.Y[20:40,0], mlab_pca.Y[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

#plt.xlabel('x_values')
#plt.ylabel('y_values')
#plt.xlim([-4,4])
#plt.ylim([-4,4])
#plt.legend()
#plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

#plt.show()


###from sklearn.decomposition

#sklearn_pca = sklearnPCA(n_components=3)
#sklearn_transf = sklearn_pca.fit_transform(arr1)
#print (sklearn_transf.T)
#plt.plot(sklearn_transf[:,0],sklearn_transf[:,1], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
#plt.plot(sklearn_transf[20:40,0], sklearn_transf[20:40,1], '^', markersize=7, color='red', alpha=0.5, label='class2')

#plt.xlabel('x_values')
#plt.ylabel('y_values')
#plt.xlim([-4,4])
#plt.ylim([-4,4])
#plt.legend()
#plt.title('Transformed samples with class labels from matplotlib.mlab.PCA()')

#plt.show()