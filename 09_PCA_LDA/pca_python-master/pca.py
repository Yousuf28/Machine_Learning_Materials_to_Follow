#!/usr/bin/python
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

np.random.seed(1) # random seed to generate same matrix 

# A reader pointed out that Python 2.7 would raise a
# "ValueError: object of too small depth for desired array".
# This can be avoided by choosing a smaller random seed, e.g. 1
# or by completely omitting this line, since I just used the random seed for
# consistency.

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

mu_vec2 = np.array([1,1,1])
cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

#print(class1_sample)
#print(class2_sample)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()

sample_pca = np.concatenate([class1_sample,class2_sample],1)
assert sample_pca.shape == (3,40), "The matrix doesn't have dimention of 3X40"

print('sample', sample_pca)
#print ('\n above is the input matrix\n\n\n')

## mean vector for each row

mean_x = np.mean(sample_pca[0,])
mean_y = np.mean(sample_pca[1,])
mean_z = np.mean(sample_pca[2,])
mean_vec = np.array([mean_x,mean_y,mean_z])

#print(mean_vec)

#print(mean_y)


#===========================
# scatter matrix
#

#print (range(sample_pca.shape[0]))
#sc_matrix = np.zeros((3,3))    
#for i in range(sample_pca.shape[1]):
#    sc_matrix += (sample_pca[:,i].reshape(3,1) - mean_vec).dot((sample_pca[:,i].reshape(3,1)-mean_vec).T)  
#print (sc_matrix)    

#=========
# Covariance matrix

cova_mat = np.cov([sample_pca[0,:], sample_pca[1,:], sample_pca[2,:]])
#print (cova_mat)

#============================
# eigenvalue and eigenvector

my_eigen, my_eigvec = np.linalg.eig(cova_mat)
#print (my_eigen, my_eigvec)
#print (cova_mat.dot(my_eigvec))
#print ('\n\n\n')
#print (my_eigen.dot(my_eigvec))
#print (my_eigvec[:,0].reshape(1,3).T)

#=====================================
# plot the eigenvalue

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

ax.plot(sample_pca[0,:], sample_pca[1,:], sample_pca[2,:], 'o', markersize=8, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
for v in my_eigvec.T:
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('Eigenvectors')

plt.show()

# Combine eigenvalue, eigenvector in a tuples
#eig_pairs = [(np.abs(my_eigen[i]), my_eigvec[:,i]) for i in range(len(my_eigen))]

#eig_pairs = []
e_p = []
for i in range(len(my_eigen)):
 eig_pairs = [np.abs(my_eigen[i]), my_eigvec[:,i]]
 e_p.append(eig_pairs)

e_p.sort(key=lambda x: x[0], reverse=True)
#print (eig_pairs)
#print (e_p) 
#for i in e_p:
#	print(i[0])
	
#Eigenvectors matrix with two top eigenvalue
eigen_matrix = np.hstack((e_p[0][1].reshape(3,1),e_p[1][1].reshape(3,1)))
#print (eigen_matrix)

# transform the orignal matrix with eigen_matrix
transf = eigen_matrix.T.dot(sample_pca)
print(transf)
 
plt.plot(transf[0,0:20], transf[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transf[0,20:40], transf[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')

plt.show()
    
