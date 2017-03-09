
# coding: utf-8

# ## Shear bands:
# 
# This notebook explores shear band emergence. The models are based on those described in  
# 
# Spiegelman, Marc, Dave A. May, and Cian R. Wilson. "On the solvability of incompressible Stokes with viscoplastic rheologies in geodynamics." Geochemistry, Geophysics, Geosystems (2016).
# 	
# Kaus, Boris JP. "Factors that control the angle of shear bands in geodynamic numerical models of brittle deformation." Tectonophysics 484.1 (2010): 36-47.
# 
# 
# Lemiale, V., et al. "Shear banding analysis of plastic models formulated for incompressible viscous flows." Physics of the Earth and Planetary Interiors 171.1 (2008): 177-186.
# 
# 
# Moresi, L., and H-B. Mühlhaus. "Anisotropic viscous models of large-deformation Mohr–Coulomb failure." Philosophical Magazine 86.21-22 (2006): 3287-3305.
#     
#     
# ## Scaling
# 
# For this problem, 
# 
# * we scale velocities by $U_0$, the imposed boundary velocity  (m/s)
# * viscosities by $10^{22}$ Pa s, and 
# * stresses/pressures by $\eta_0 U_0/H$, where H is the layer depth (m). 
# 
# ### NOTES
# 

# In[1]:

import numpy as np
import underworld as uw
import math
from underworld import function as fn
import glucifer

import os
import sys
import natsort
import shutil
from easydict import EasyDict as edict
import operator
import pint
import time
import operator

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# In[2]:

#####
#Stubborn version number conflicts - For now...
#####
try:
    natsort.natsort = natsort.natsorted
except:
    natsort.natsort = natsort.natsort


# ## Some necessary functions

# In[3]:

#In case NN swarm interpolation is required

from scipy.spatial import cKDTree as kdTree

def nn_evaluation(fromSwarm, toSwarm, n=1, weighted=False):
    """
    This function provides nearest neighbour information for uw swarms, 
    given the "toSwarm", this function returns the indices of the n nearest neighbours in "fromSwarm"
    it also returns the inverse-distance if weighted=True. 
    
    The function works in parallel.
    
    The arrays come out a bit differently when used in nearest neighbour form
    (n = 1), or IDW: (n > 1). The examples belowe show how to fill out a swarm variable in each case. 
    
    
    Usage n == 1:
    ------------
    ix, weights = nn_evaluation(swarm, data, n=1, weighted=False)
    toSwarmVar.data[:][:,0] = np.average(fromSwarmVar[ix][:,0], weights=weights)
    
    Usage n > 1:
    ------------
    ix, weights = nn_evaluation(swarm, data, n=2, weighted=False)
    toSwarmVar.data[:][:,0] =  np.average(fromSwarmVar[ix][:,:,0], weights=weights, axis=1)
    
    """
    
    
    if len(toSwarm) > 0: #this is required for safety in parallel
        
        #this should avoid building the tree again when this function is called multiple times.
        try:
            tree = fromSwarm.tree
            #print(1)
        except:
            #print(2)
            fromSwarm.tree = kdTree(fromSwarm.particleCoordinates.data)
            tree = fromSwarm.tree
        d, ix = tree.query(toSwarm, n)
        if n == 1:
            weights = np.ones(toSwarm.shape[0])
        elif not weighted:
            weights = np.ones((toSwarm.shape[0], n))*(1./n)
        else:
            weights = (1./d[:])/(1./d[:]).sum(axis=1)[:,None]
        return ix,  weights 
    else:
        return [], []


# In[4]:

def eig2d(sigma):

    """
    Input: sigma, symmetric tensor, numpy array of length 3, with xx yy xy compenents
    
    Output:
    
    s1: first major stress fms will be the most extensive
    s2: second major stress the least extensive, most compressive
    deg: angle to the first major stress axis (most extensive in degrees anticlockwise from horizontal axis - x)
    """
 

    s11=sigma[0]
    #s12=sigma[2]/2.  #(engineering strain/stress)
    s12=sigma[2]
    s22=sigma[1]

    fac = 28.64788975654116; #90/pi - 2theta conversion

    x1 = (s11 + s22)/2.0;
    x2 = (s11 - s22)/2.0;
    R = x2 * x2 + s12 * s12;
    
    #Get the stresses 
    if(R > 0.0):         #if shear stress is not zero
        R = np.sqrt(R);
        s1 = x1 + R;
        s2 = x1 - R;
    else:
        s1 = x1;        #if shear stress is zero
        s2 = x1;

    if(x2 != 0.0):   
        deg = fac * math.atan2(s12,x2); #Return the arc tangent (measured in radians) of y/x.
    elif s12 <= 0.0:
        deg= -45.0;
    else:
        deg=  45.0;
    return s1, s2, deg


# Model name and directories
# -----

# In[5]:

############
#Model letter and number
############


#Model letter identifier default
Model = "T"

#Model number identifier default:
ModNum = 0

#Any isolated letter / integer command line args are interpreted as Model/ModelNum

if len(sys.argv) == 1:
    ModNum = ModNum 
elif sys.argv[1] == '-f': #
    ModNum = ModNum 
else:
    for farg in sys.argv[1:]:
        if not '=' in farg: #then Assume it's a not a paramter argument
            try:
                ModNum = int(farg) #try to convert everingthing to a float, else remains string
            except ValueError:
                Model  = farg


# In[6]:

###########
#Standard output directory setup
###########

outputPath = "results" + "/" +  str(Model) + "/" + str(ModNum) + "/" 
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
outputFile = 'results_model' + Model + '_' + str(ModNum) + '.dat'

if uw.rank()==0:
    # make directories if they don't exist
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    if not os.path.isdir(dbPath):
        os.makedirs(dbPath)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)

        
comm.Barrier() #Barrier here so no procs run the check in the next cell too early


# ## Set parameter dictionaries
# 
# * Parameters are stored in dictionaries. 
# * If starting from checkpoint, parameters are loaded using pickle
# * If params are passed in as flags to the script, they overwrite 
# 

# In[ ]:




# In[7]:

###########
#Parameter / settings dictionaries get saved&loaded using pickle
###########
 
dp = edict({}) #dimensional parameters
sf = edict({}) #scaling factors
ndp = edict({}) #dimensionless paramters
md = edict({}) #model paramters, flags etc
#od = edict({}) #output frequencies
 


dict_list = [dp, sf, ndp, md]
dict_names = ['dp.pkl', 'sf.pkl', 'ndp.pkl', 'md.pkl']



# In[8]:

###########
#Store the physical parameters, scale factors and dimensionless paramters in easyDicts
###########

#dp : dimensional paramters
dp = edict({})
dp.depth=30*1e3                #domain depth
dp.asthenosphere=dp.depth/4   #level from bottom of model,
dp.eta1=1e24
dp.eta2=1e21
dp.etaMin=1e18
dp.U0=0.0025/(3600*24*365)     #m/s 
dp.rho=2700.                   #kg/m3
dp.g=9.81
dp.cohesion=100e6              #
dp.fa=30.                      #friction angle degrees
dp.a=1.                        #fraction of the dynamic pressure to allow in the model
dp.notchWidth = dp.depth/16.


#md : Modelling choices and Physics switches
md = edict({})        
md.refineMesh=False
md.stickyAir=False
md.aspectRatio=4.
md.res=64
md.ppc=25
md.tol=1e-10
md.maxIts=50
md.perturb=0 # 0 for material heterogeneity, 1 for cohesion weakening
md.directorPhi=45. #if this is , angle is same as maximum shear 
md.directorV=1e-5 #if this is false , angle is taken as tan(fa) / 2.
md.directorSigma=1e-5 #standard deviation in fault orientation


# In[9]:

###########
#If command line args are given, overwrite
#Note that this assumes that params as commans line args/
#only append to the 'dimensional' and 'model' dictionary (not the non-dimensional)
###########    


###########
#If extra arguments are provided to the script" eg:
### >>> uw.py 2 dp.arg1=1 dp.arg2=foo dp.arg3=3.0
###
###This would assign ModNum = 2, all other values go into the dp dictionary, under key names provided
###
###Two operators are searched for, = & *=
###
###If =, parameter is re-assigned to givn value
###If *=, parameter is multipled by given value
###
### >>> uw.py 2 dp.arg1=1 dp.arg2=foo dp.arg3*=3.0
###########

for farg in sys.argv[1:]:
    try:
        (dicitem,val) = farg.split("=") #Split on equals operator
        (dic,arg) = dicitem.split(".") #colon notation
        if '*=' in farg:
            (dicitem,val) = farg.split("*=") #If in-place multiplication, split on '*='
            (dic,arg) = dicitem.split(".")
            
        if val == 'True': 
            val = True
        elif val == 'False':     #First check if args are boolean
            val = False
        else:
            try:
                val = float(val) #next try to convert  to a float,
            except ValueError:
                pass             #otherwise leave as string
        #Update the dictionary
        if farg.startswith('dp'):
            if '*=' in farg:
                dp[arg] = dp[arg]*val #multiply parameter by given factor
            else:
                dp[arg] = val    #or reassign parameter by given value
        if farg.startswith('md'):
            if '*=' in farg:
                md[arg] = md[arg]*val #multiply parameter by given factor
            else:
                md[arg] = val    #or reassign parameter by given value
                
    except:
        pass
            

comm.barrier()


# In[10]:

#In this code block we map the dimensional paramters to dimensionless, through scaling factors

#sf : scaling factors

sf = edict({})
sf.LS = 30*1e3
sf.eta0 = 1e22
sf.vel = 0.0025/(3600*24*365)
sf.stress = (sf.eta0*dp.U0)/sf.LS
sf.density = sf.LS**3
sf.g = 9.81
sf.rho = (sf.eta0*dp.U0)/(sf.LS**2*sf.g)


#ndp : non dimensional parameters
ndp = edict({})
ndp.depth = dp.depth/sf.LS
ndp.U0 = dp.U0/sf.vel
ndp.asthenosphere = dp.asthenosphere/sf.LS
ndp.eta1 = dp.eta1/sf.eta0
ndp.eta2 = dp.eta2/sf.eta0
ndp.etaMin = dp.etaMin/sf.eta0
ndp.cohesion = (dp.cohesion/sf.stress)
ndp.fa = math.tan(np.radians(dp.fa)) #friction coefficient
ndp.g = dp.g/sf.g
ndp.rho = dp.rho/sf.rho
ndp.notchWidth = dp.notchWidth/sf.LS
ndp.a = dp.a



ndp.fa, ndp.cohesion


# Create mesh and finite element variables
# ------
# 
# Note: the use of a pressure-sensitive rheology suggests that it is important to use a Q2/dQ1 element 

# In[11]:

minX  = -0.5*md.aspectRatio
maxX  =  0.5*md.aspectRatio
maxY  = ndp.depth
meshV =  1.0

if md.stickyAir:
    maxY  = 1.1


resY = int(md.res)
resX = int(resY*md.aspectRatio)

elementType="Q2/dPc1"  # This is enough for a test but not to use the code in anger

mesh = uw.mesh.FeMesh_Cartesian( elementType = (elementType), 
                                 elementRes  = ( resX, resY), 
                                 minCoord    = ( minX, 0.), 
                                 maxCoord    = ( maxX, maxY),
                                 periodic    = [False, False]  ) 



velocityField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=mesh.dim )
pressureField    = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )

velocityField.data[:] = [0.,0.]
pressureField.data[:] = 0.


# ### Boundary conditions
# 
# Pure shear with moving  walls — all boundaries are zero traction with 

# In[12]:

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
base   = mesh.specialSets["MinJ_VertexSet"]
top    = mesh.specialSets["MaxJ_VertexSet"]

allWalls = iWalls + jWalls

velocityBCs = uw.conditions.DirichletCondition( variable        = velocityField, 
                                                indexSetsPerDof = (iWalls, base) )

for index in mesh.specialSets["MinI_VertexSet"]:
    velocityField.data[index] = [ndp.U0, 0.]
for index in mesh.specialSets["MaxI_VertexSet"]:
    velocityField.data[index] = [ -ndp.U0, 0.]
    


# ### Setup the material swarm and passive tracers
# 
# The material swarm is used for tracking deformation and history dependence of the rheology
# 
# Passive swarms can track all sorts of things but lack all the machinery for integration and re-population

# In[13]:

swarm  = uw.swarm.Swarm( mesh=mesh )
swarmLayout = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=swarm, particlesPerCell=int(md.ppc) )
swarm.populate_using_layout( layout=swarmLayout )

# create pop control object
pop_control = uw.swarm.PopulationControl(swarm)

surfaceSwarm = uw.swarm.Swarm( mesh=mesh )


# ### Create a particle advection system
# 
# Note that we need to set up one advector systems for each particle swarm (our global swarm and a separate one if we add passive tracers).

# In[14]:

advector        = uw.systems.SwarmAdvector( swarm=swarm,            velocityField=velocityField, order=2 )
advector2       = uw.systems.SwarmAdvector( swarm=surfaceSwarm,     velocityField=velocityField, order=2 )


# ### Add swarm variables
# 
# We are using a single material with a single rheology. We need to track the plastic strain in order to have some manner of strain-related softening (e.g. of the cohesion or the friction coefficient). For visualisation of swarm data we need an actual swarm variable and not just the computation.
# 
# Other variables are used to track deformation in the shear band etc.
# 
# **NOTE**:  Underworld needs all the swarm variables defined before they are initialised or there will be / can be memory problems (at least it complains about them !). That means we need to add the monitoring variables now, even if we don't always need them.

# In[15]:

# Tracking different materials

materialVariable = swarm.add_variable( dataType="int", count=1 )

directorVector   = swarm.add_variable( dataType="double", count=2)



# plastic deformation for weakening

plasticStrain  = swarm.add_variable( dataType="double",  count=1 )



# passive markers at the surface

surfacePoints = np.zeros((1000,2))
surfacePoints[:,0] = np.linspace(minX+0.01, maxX-0.01, 1000)
surfacePoints[:,1] = 1.0 #

surfaceSwarm.add_particles_with_coordinates( surfacePoints )
yvelsurfVar = surfaceSwarm.add_variable( dataType="double", count=1)


# ### Initialise swarm variables
# 

# In[16]:

yvelsurfVar.data[...] = (0.)
materialVariable.data[...] = 0

plasticStrain.data[...] = 0


# ### Material distribution in the domain.
# 
# 

# In[17]:

# Initialise the 'materialVariable' data to represent different materials. 
material1 = 1 # viscoplastic
material0 = 0 # accommodation layer a.k.a. Sticky Air
material2 = 2 # Under layer 


materialVariable.data[:] = 0.

# The particle coordinates will be the input to the function evaluate (see final line in this cell).
# We get proxy for this now using the input() function.

coord = fn.input()

# Setup the conditions list for the following conditional function. Where the
# z coordinate (coordinate[1]) is less than the perturbation, set to lightIndex.



#notchWidth = (1./32.) * md.notch_fac

notchCond = operator.and_(coord[1] < ndp.asthenosphere + ndp.notchWidth, operator.and_(coord[0] < ndp.notchWidth, coord[0] > -1.*ndp.notchWidth )  )

mu = ndp.notchWidth
sig =  0.25*ndp.notchWidth
gausFn1 = ndp.notchWidth*fn.math.exp(-1.*(coord[0] - mu)**2/(2 * sig**2)) + ndp.asthenosphere
mu = -1.*ndp.notchWidth
gausFn2 = ndp.notchWidth*fn.math.exp(-1.*(coord[0] - mu)**2/(2 * sig**2)) + ndp.asthenosphere

conditions = [ (       coord[1] > 1.0 , material0 ), #air
               (       coord[1] < ndp.asthenosphere , material2 ), #asthenosphere
               (       coord[1] < gausFn1 , material2 ), #asthenosphere
               (       coord[1] < gausFn2 , material2 ), #asthenosphere       

               (       notchCond , material2 ),
               (       True ,           material1 ) ]  #visco-plastic

# The actual function evaluation. Here the conditional function is evaluated at the location
# of each swarm particle. The results are then written to the materialVariable swarm variable.

if md.perturb == 0:
    materialVariable.data[:] = fn.branching.conditional( conditions ).evaluate(swarm)
    
else: 
    #in this case just build the asphenosphere
    materialVariable.data[:] = material1
    materialVariable.data[np.where(swarm.particleCoordinates.data[:,1] < ndp.asthenosphere)] = material2


# In[19]:

figMat = glucifer.Figure( figsize=(1200,400), boundingBox=((-2.0, 0.0, 0.0), (2.0, 1.0, 0.0)) )
figMat.append( glucifer.objects.Points(swarm,materialVariable, pointSize=2.0) )
figMat.append( glucifer.objects.Mesh(mesh))
#figMat.show()


# ## Buoyancy forces
# 
# In this example, no buoyancy forces are included in the Stokes system, the Pressures that appear are dynamic (p'). We add the appropriate lithostatic component to the Drucker-Prager yield criterion.

# In[20]:

lithPressureFn = ndp.rho* (1. - coord[1])


# ## Background Rheology

# In[21]:

strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateFn)


# In[22]:

# plastic strain - weaken a region at the base close to the boundary (a weak seed but through cohesion softening)

def gaussian(xx, centre, width):
    return ( np.exp( -(xx - centre)**2 / width ))

def boundary(xx, minX, maxX, width, power):
    zz = (xx - minX) / (maxX - minX)
    return (np.tanh(zz*width) + np.tanh((1.0-zz)*width) - math.tanh(width))**power

# weight = boundary(swarm.particleCoordinates.data[:,1], 10, 4) 

if md.perturb != 0: #build a heterogenity into the cohesion, through the accumulated plastic strain term

    #plasticStrain.data[:] = 0.5 * np.random.rand(*plasticStrain.data.shape[:])
    plasticStrain.data[:] = np.random.normal(loc=0.5, scale=0.05,size = plasticStrain.data.shape[:])
    plasticStrain.data[:,0] *= gaussian(swarm.particleCoordinates.data[:,0], 0.0, ndp.notchWidth) 
    plasticStrain.data[:,0] *= gaussian(swarm.particleCoordinates.data[:,1], ndp.asthenosphere, ndp.notchWidth/2.) 
    plasticStrain.data[:,0] *= boundary(swarm.particleCoordinates.data[:,0], minX, maxX, 10.0, 2)
    plasticStrain.data[:,0][np.where(swarm.particleCoordinates.data[:,1] < ndp.asthenosphere)] = 0.

# 



# In[23]:

# Friction - in this form it can also be made to weaken with strain

cohesion0       = fn.misc.constant(ndp.cohesion)
cohesionInf = 0.25*cohesion0
refStrain = 0.5

# Drucker-Prager yield criterion

weakenedCohesion = cohesion0+ (cohesionInf - cohesion0)*fn.misc.min(1., plasticStrain/refStrain) 

yieldStressFn   = weakenedCohesion  + ndp.fa *(lithPressureFn + ndp.a*pressureField ) 

#yieldStressFn   = cohesionFn + ndp.fa *(lithPressureFn + ndp.a*fn.misc.max(fn.misc.constant(0.), pressureField) ) #in this case only positive dynamic pressures


# first define strain rate tensor

strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
strainRate_2ndInvariantFn = fn.tensor.second_invariant(strainRateFn)

# now compute a viscosity assuming yielding

#min_viscosity = visc0  # same as the air ... 

yieldingViscosityFn =  0.5 * yieldStressFn / (strainRate_2ndInvariantFn+1.0e-18)

#viscosityFn = fn.exception.SafeMaths( fn.misc.max(fn.misc.min(yieldingViscosityFn, 
#                                                              backgroundViscosityFn), 
#                                                  min_viscosity))


compositeviscosityFn = fn.exception.SafeMaths( fn.misc.max(
                                              1./((1./yieldingViscosityFn) + (1./ndp.eta1))
                                             , ndp.etaMin))


# In[24]:

viscosityMap = { material0: ndp.etaMin, material1:compositeviscosityFn, material2:ndp.eta2 }

viscosityFn  = fn.branching.map( fn_key = materialVariable, 
                                           mapping = viscosityMap )

backgroundviscosityMap = { material0: ndp.etaMin, material1:ndp.eta1, material2:ndp.eta2 }

backgroundViscosityFn  = fn.branching.map( fn_key = materialVariable, 
                                           mapping = backgroundviscosityMap )


# ## Initial Stokes solve to compute the stresses
# 

# In[ ]:




# In[25]:

stokes = uw.systems.Stokes(    velocityField = velocityField, 
                               pressureField = pressureField,
                               conditions    = velocityBCs,
                               fn_viscosity  = viscosityFn)

solver = uw.systems.Solver( stokes )


# In[26]:

solver.solve(nonLinearIterate=True, nonLinearMaxIterations=1)


# ## Principal stress orientations directly

# In[27]:

strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )
LFn = velocityField.fn_gradient 


# In[28]:

ssrS = strainRateFn.evaluate(swarm)
ssrM = strainRateFn.evaluate(mesh)


# In[29]:

principalAnglesM  = np.apply_along_axis(eig2d, 1, ssrM[:, :])[:,2]


# In[30]:

#eig2d returns the most extensive axis, we want most compressive

principalAnglesM -= 90.


# In[31]:

principalStress   = uw.mesh.MeshVariable( mesh, mesh.dim )

principalStress.data[:,0] = np.cos(np.radians(principalAnglesM))
principalStress.data[:,1] = np.sin(np.radians(principalAnglesM))


# ## Eigenvector approach

# In[32]:

eig1  = uw.mesh.MeshVariable( mesh, mesh.dim )
eig2  = uw.mesh.MeshVariable( mesh, mesh.dim )

#eig1  = swarm.add_variable( dataType="double",  count=2 )
#eig2  = swarm.add_variable( dataType="double",  count=2 )



for ti, val in enumerate(eig1.data):
    eigVals, eigVex= np.linalg.eig(np.array([[ssrM[ti][0],ssrM[ti][2]],[ssrM[ti][2],ssrM[ti][1]]]))
    pi = np.argmax(eigVals) #index of most extensive eigenvalue
    eig1.data[ti] = eigVex[pi] 
    eig2.data[ti] = eigVex[abs(pi - 1)]  #index of other eigenvalue - 2D assumption


# In[ ]:




# In[34]:

figP = glucifer.Figure( figsize=(1600,400), boundingBox=((-2.0, 0.0, 0.0), (2.0, 1.0, 0.0)) )

figP.append( glucifer.objects.VectorArrows(mesh, principalStress , arrowHead=0.2, scaling=.1, resolutionI=64, resolutionJ=16) )
#figP.append( glucifer.objects.VectorArrows(mesh, eig2, arrowHead=0.2, scaling=.1, resolutionI=64, resolutionJ=16) )


#figP.show()


# In[ ]:




# ## Director Vector for TI rheology

# In[35]:

#Grab principal angles on the Swarm
principalAngles  = np.apply_along_axis(eig2d, 1, ssrS[:, :])[:,2]

#eig2d returns the most extensive axis, we want most compressive

principalAngles -= 90.


# In[36]:

aOrient = principalAngles + (45. - dp.fa/2.)
bOrient = principalAngles - (45. - dp.fa/2.)
finalOrient = np.zeros(aOrient.shape)


# In[37]:

an = np.zeros((aOrient.shape[0], 2))
bn = np.zeros((bOrient.shape[0], 2))

an[:,0] = np.cos(np.radians(aOrient))
an[:,1] = np.sin(np.radians(aOrient))
#                           
bn[:,0] = np.cos(np.radians(bOrient))
bn[:,1] = np.sin(np.radians(bOrient))


# In[38]:

Lij = LFn.evaluate(swarm)


# In[39]:

#define the matrix product for $\dot n_i = L_{ji} n_j$

def Ldotn(L,n):
    x,y =  L[:,0]*n[:,0] + L[:,1]*n[:,1], L[:,2]*n[:,0] + L[:,3]*n[:,1]
    return np.column_stack((x, y))


# In[40]:

#apply L_{ji} n_j

aAction = Ldotn(Lij,an)
bAction = Ldotn(Lij,bn)


# In[41]:

#The slip plane most closely aligned to the sense of shear is the one with the smaller norm |\dot n_i|

amask = np.linalg.norm(aAction, axis=1) < np.linalg.norm(bAction, axis=1)

finalOrient[:] = bOrient[:]
finalOrient[amask] = aOrient[amask]


# In[42]:

#in this step we find the normal vector to the shear band
directorVector.data[:,0] = np.sin(np.radians(finalOrient))
directorVector.data[:,1] = -1.*np.cos(np.radians(finalOrient))


# In[45]:

ix, weights = nn_evaluation(swarm, mesh.data, n=1, weighted=False)

meshDirector = uw.mesh.MeshVariable( mesh, mesh.dim )
meshDirector.data[:] = directorVector.data[ix]


figDir = glucifer.Figure( figsize=(1600,400), boundingBox=((-2.0, 0.0, 0.0), (2.0, 1.0, 0.0)) )

figDir.append( glucifer.objects.VectorArrows(mesh, meshDirector, arrowHead=0.01, scaling=.1, resolutionI=32, resolutionJ=8) )
figDir.append( glucifer.objects.VectorArrows(mesh, principalStress,  arrowHead=0.2, scaling=.1, resolutionI=32, resolutionJ=8) )
#figDir.append( glucifer.objects.VectorArrows(mesh, eig1, arrowHead=0.2, scaling=.1, resolutionI=32, resolutionJ=8) )


#figDir.show()


# ## Transversely isotropic rheology

# In[46]:

#resolved strain rates

edotn_SFn = (        directorVector[0]**2 * strainRateFn[0]  +
                        2.0 * directorVector[1]    * strainRateFn[2] * directorVector[0] +
                              directorVector[1]**2 * strainRateFn[1]
                    )

edots_SFn = (  directorVector[0] *  directorVector[1] *(strainRateFn[1] - strainRateFn[0]) +
                        strainRateFn[2] * (directorVector[0]**2 - directorVector[1]**2)
                     )



# In[47]:

##Transversely isotropic rheology


cohesion0       = fn.misc.constant(ndp.cohesion)
cohesionFn = cohesion0

#can try several variants on this - 
#weakenedCohesion = cohesion0+ (cohesionInf - cohesion0)*fn.misc.min(1., plasticStrain/refStrain) 
#yieldStressFn   = weakenedCohesion  + ndp.fa *(lithPressureFn + ndp.a*pressureField ) 

viscosity2Fn = ( (ndp.fa *(lithPressureFn + ndp.a*pressureField )  + weakenedCohesion ) / (2.*fn.math.abs(edots_SFn) + 1.0e-15))


delviscosity2fn = fn.misc.min(ndp.eta1 - ndp.etaMin, fn.misc.max(0.0, 
                     ndp.eta1 - viscosity2Fn))


delviscosity2Map    = { 0: 0.0, 
                     1: delviscosity2fn,  #only material a has stress-limting TI rheology
                     2: 0.                   
                   }

secondViscosityFn  = fn.branching.map( fn_key  = materialVariable, 
                                       mapping = delviscosity2Map )


# In[ ]:




# System setup
# -----
# 
# Setup a Stokes equation system and connect a solver up to it.  
# 
# In this example, no buoyancy forces are considered. However, to establish an appropriate pressure gradient in the material, it would normally be useful to map density from material properties and create a buoyancy force.

# In[48]:

stokes = uw.systems.Stokes( velocityField  = velocityField, 
                               pressureField  = pressureField,
                               conditions     = velocityBCs,
                               fn_viscosity   = backgroundViscosityFn, 
                              _fn_viscosity2  = secondViscosityFn,
                              _fn_director    = directorVector)
solver = uw.systems.Solver( stokes )

# "mumps" is a good alternative for "lu" but  # use "lu" direct solve and large penalty (if running in serial)

if(uw.nProcs()==1):
    solver.set_inner_method("lu")
    solver.set_penalty(1.0e7)
    solver.options.scr.ksp_type="cg"
    solver.options.scr.ksp_rtol = 1.0e-4

else:
    solver.set_inner_method("mumps")
    solver.set_penalty(1.0e7)
    solver.options.scr.ksp_type="cg"
    solver.options.scr.ksp_rtol = 1.0e-4



# In[49]:

#solver.solve( nonLinearIterate=True, nonLinearMaxIterations=20)


# In[50]:

#solver._stokesSLE._cself.curResidual


# ## Manual Picard iteration

# In[51]:

prevVelocityField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=mesh.dim )

prevPressureField    = uw.mesh.MeshVariable( mesh=mesh.subMesh,         nodeDofCount=1)


prevVelocityField.data[:] = (0., 0.)
prevPressureField.data[:] = 0.


# In[52]:

def volumeint(Fn = 1., rFn=1.):
    return uw.utils.Integral( Fn*rFn,  mesh )


# In[53]:

md.maxIts = 5


# In[54]:

surfaceArea = uw.utils.Integral(fn=1.0,mesh=mesh, integrationType='surface', surfaceIndexSet=top)
surfacePressureIntegral = uw.utils.Integral(fn=pressureField, mesh=mesh, integrationType='surface', surfaceIndexSet=top)

(area,) = surfaceArea.evaluate()


# In[61]:

#The underworld Picard interation applies the following residual (SystemLinearEquations.c)

#/* Calculate Residual */
#      VecAXPY( previousVector, -1.0, currentVector );
#      VecNorm( previousVector, NORM_2, &prevVecNorm );
#      VecNorm( currentVector, NORM_2, &currVecNorm );
#      residual = ((double)prevVecNorm) / ((double)currVecNorm);



count = 0


res1Vals = []
res2Vals = []
res3Vals = []

for i in range(int(md.maxIts)):
    
    prevVelocityField.data[:] = velocityField.data.copy()
    prevPressureField.data[:] = pressureField.data[:] 

    
    solver.solve( nonLinearIterate=False)
    
    #remove drift in the pressure
    (p0,) = surfacePressureIntegral.evaluate() 
    pressureField.data[:] -= p0 / area
    
    #Update the dynamic pressure variable
    #dynPressureField.data[:] = pressureField.data[:] - lithPressureFn.evaluate(mesh.subMesh)
    
    
    ####
    #Calculate a range of norms to assess convergence
    ####
    
    #L2 norm of current velocity
    v2 = fn.math.dot(velocityField,  velocityField)
    _Vr = volumeint(v2)
    velL2 = np.sqrt(_Vr.evaluate()[0])
    
    
    #L2 norm of delta velocity
    
    delV = velocityField - prevVelocityField
    v2 = fn.math.dot(delV,  delV)
    _Vr = volumeint(v2)
    delvelL2 = np.sqrt(_Vr.evaluate()[0])
    
    
    #L2 norm of current dynamic pressure
    p2 = fn.math.dot(pressureField, pressureField)
    _Pr = volumeint(p2)
    pL2 = np.sqrt(_Pr.evaluate()[0])
    
    
    #L2 norm of delta dynamic pressure
    delP = pressureField - prevPressureField
    p2 = fn.math.dot(delP,  delP)
    _Pr = volumeint(p2)
    delpL2 = np.sqrt(_Pr.evaluate()[0])
    
    #Full norm of the primal variables
    
    x2 = fn.math.dot(velocityField,  velocityField) + fn.math.dot(pressureField, pressureField)
    _Xr = volumeint(x2)
    xL2 = np.sqrt(_Xr.evaluate()[0])
    
    #Full norm of the change in primal variables
    
    delV = velocityField - prevVelocityField
    delP = pressureField - prevPressureField
    x2 = fn.math.dot(delV,  delV) + fn.math.dot(delP, delP)
    _Xr = volumeint(x2)
    delxL2 = np.sqrt(_Xr.evaluate()[0])
    
    
    
    res1 = abs(delvelL2 /velL2)
    res1Vals .append(res1)
    
    res2 = abs(delpL2 /pL2)
    res2Vals .append(res2)
    
    res3 = abs(delxL2 /xL2)
    res3Vals .append(res3)

    
    count +=1
    print(res1, res2, res3)
    print(count)
    
    
    #Converged stopping condition
    if res1 < md.tol:
        break


# ## Figures

# In[62]:

#dp.fa


# In[63]:

figSinv = glucifer.Figure( figsize=(1600,400), boundingBox=((-2.0, 0.0, 0.0), (2.0, 1.0, 0.0)) )

#figSinv .append( glucifer.objects.Points(swarm,plasticStrain, pointSize=2.0) )

figSinv .append( glucifer.objects.Points(swarm,strainRate_2ndInvariantFn, pointSize=2.0, valueRange=[2e-1, 1.5]) )
figSinv.append( glucifer.objects.VectorArrows(mesh, -1.*meshDirector, arrowHead=0.2, scaling=.075, resolutionI=32, resolutionJ=8) )
#figSinv.append( glucifer.objects.VectorArrows(mesh, sigma1, arrowHead=0.0, scaling=.1, resolutionI=32, resolutionJ=8) )


#figSinv.show()
#figSinv.save_image('ti.png')


# In[64]:

figVisc = glucifer.Figure( figsize=(1600,400), boundingBox=((-2.0, 0.0, 0.0), (2.0, 1.0, 0.0)) )

figVisc.append( glucifer.objects.VectorArrows(mesh, velocityField, arrowHead=0.25, scaling=.075, resolutionI=32, resolutionJ=8) )
#figVisc.append( glucifer.objects.Points(swarm, viscosityFn, pointSize=2.0, logScale=True ,valueRange=[1., 1000]) )

figVisc.append( glucifer.objects.Points(swarm, backgroundViscosityFn - secondViscosityFn, pointSize=2.0, logScale=True, valueRange=[5., 100.] ) )

#figVisc.show()


# In[ ]:




# In[58]:

#viscosityFn.evaluate([0.5, 0.5])
#velocityField.evaluate([0.5, 0.5])


# In[65]:

figPres= glucifer.Figure( figsize=(1600,400), boundingBox=((-2.0, 0.0, 0.0), (2.0, 1.0, 0.0)) )

figPres.append( glucifer.objects.Points(swarm, pressureField,pointSize=2.0, valueRange=[-5.,25.] ))

#figPres.draw.label(r'$\sin (x)$', (0.2,0.7,0))
#figPres.append( glucifer.objects.Mesh(mesh,opacity=0.2))

#figPres.show()


# In[ ]:




# In[127]:

figSinv.save_image(imagePath + "figSinv.png")

figVisc.save_image(imagePath +  "figVisc.png")

figPres.save_image(imagePath + "figPres.png")


# In[128]:

velocityField.save(filePath + "vel.h5")
pressureField.save(filePath + "pressure.h5")


# In[129]:

yvelsurfVar.data[...] = velocityField[1].evaluate(surfaceSwarm)
yvelsurfVar.save(filePath + "yvelsurf.h5")


# ## Save points to determine shear band angle

# In[153]:

eii_mean = uw.utils.Integral(strainRate_2ndInvariantFn,mesh).evaluate()[0]/4.

eii_std = uw.utils.Integral(fn.math.sqrt(0.25*(strainRate_2ndInvariantFn - eii_mean)**2.), mesh).evaluate()[0]


# In[176]:

#grab all of material points that are above 2 sigma of mean strain rate invariant



xv, yv = np.meshgrid(
        np.linspace(mesh.minCoord[0], mesh.maxCoord[0], mesh.elementRes[0]), 
        np.linspace(mesh.minCoord[1], mesh.maxCoord[1], mesh.elementRes[1]))

meshGlobs = np.row_stack((xv.flatten(), yv.flatten())).T


#Calculate the 2-sigma value of the strain rate invariant function (
#we use this a definition for a shear band)
eII_sig = eii_mean  + 1.5*eii_std


# In[177]:

shearbandswarm  = uw.swarm.Swarm( mesh=mesh, particleEscape=True )
shearbandswarmlayout  = uw.swarm.layouts.GlobalSpaceFillerLayout( swarm=shearbandswarm , particlesPerCell=int(md.ppc/16.) )
shearbandswarm.populate_using_layout( layout=shearbandswarmlayout )


# In[178]:

#shearbandswarm.particleGlobalCount


# In[179]:

np.unique(strainRate_2ndInvariantFn.evaluate(shearbandswarm) < eII_sig)


# In[180]:

with shearbandswarm.deform_swarm():
    mask = np.where(strainRate_2ndInvariantFn.evaluate(shearbandswarm) < eII_sig)
    shearbandswarm.particleCoordinates.data[mask[0]]= (1e20, 1e20)

shearbandswarm.update_particle_owners()    

with shearbandswarm.deform_swarm():
    mask = np.where((shearbandswarm.particleCoordinates.data[:,1] < ndp.asthenosphere + ndp.notchWidth) | 
                    (shearbandswarm.particleCoordinates.data[:,1] >  1. - ndp.notchWidth) |
                   (shearbandswarm.particleCoordinates.data[:,0] <  minX/1.5))
    shearbandswarm.particleCoordinates.data[mask]= (1e20, 1e20)

shearbandswarm.update_particle_owners()


with shearbandswarm.deform_swarm():
    mask = np.where(shearbandswarm.particleCoordinates.data[:,0] > -2.*ndp.notchWidth)
    shearbandswarm.particleCoordinates.data[mask]= (1e20, 1e20)
                    
shearbandswarm.update_particle_owners()


# In[181]:

shearbandswarm.save(filePath + 'swarm.h5')



# ## Calculate and save some metrics

# In[182]:

#We'll create a function that based on the strain rate 2-sigma value. 
#Use this to estimate thickness and average pressure within the shear band

conds = [ ( (strainRate_2ndInvariantFn >  eII_sig) & (coord[1] > ndp.asthenosphere + ndp.notchWidth), 1.),
            (                                           True , 0.) ]


conds2 = [ ( (strainRate_2ndInvariantFn <  eII_sig) & (coord[1] > ndp.asthenosphere + ndp.notchWidth), 1.),
            (                                           True , 0.) ]


# lets also integrate just one eighth of sphere surface
_2sigRest= fn.branching.conditional( conds ) 

_out2sigRest= fn.branching.conditional( conds2 ) 


# In[183]:

sqrtv2 = fn.math.sqrt(fn.math.dot(velocityField,velocityField))
#sqrtv2x = fn.math.sqrt(fn.math.dot(velocityField[0],velocityField[0]))
vd = 4.*backgroundViscosityFn*strainRate_2ndInvariantFn # there's an extra factor of 2, which is necessary because the of factor of 0.5 in the UW second invariant 


_rmsint = uw.utils.Integral(sqrtv2, mesh)

#_rmsSurf = uw.utils.Integral(sqrtv2x, mesh, integrationType='Surface',surfaceIndexSet=mesh.specialSets["MaxJ_VertexSet"])

_viscMM = fn.view.min_max(backgroundViscosityFn)
dummyFn = _viscMM.evaluate(swarm)

_eiiMM = fn.view.min_max(strainRate_2ndInvariantFn)
dummyFn = _eiiMM.evaluate(swarm)

#Area and pressure integrals inside / outside shear band
_shearArea = uw.utils.Integral(_2sigRest, mesh)
_shearPressure = uw.utils.Integral(_2sigRest*pressureField, mesh)

_backgroundArea = uw.utils.Integral(_out2sigRest, mesh)
_backgroundPressure = uw.utils.Integral(_out2sigRest*pressureField, mesh)

#dissipation 

_vdint  = uw.utils.Integral(vd,mesh)
_shearVd  = uw.utils.Integral(vd*_2sigRest,mesh)
_backgroundVd  = uw.utils.Integral(vd*_out2sigRest,mesh)


#dynamic pressure min / max

#dynPressureField.data[:] = pressureField.data[:] - lithPressureFn.evaluate(mesh.subMesh)
_press = fn.view.min_max(pressureField)
dummyFn = _press.evaluate(swarm)


# In[ ]:




# In[184]:

#_viscMM.min_global(), _viscMM.max_global()
#_eiiMM.min_global(), _eiiMM.max_global()

rmsint = _rmsint.evaluate()[0]

shearArea = _shearArea.evaluate()[0]
shearPressure = _shearPressure.evaluate()[0]

backgroundArea = _backgroundArea.evaluate()[0]
backgroundPressure = _backgroundPressure.evaluate()[0]

vdint = _vdint.evaluate()[0]
shearVd = _shearVd.evaluate()[0]
backgroundVd = _backgroundVd.evaluate()[0]


viscmin = _viscMM.min_global()
viscmax = _viscMM.max_global()
eiimin = _eiiMM.min_global()
eiimax = _eiiMM.max_global()

pressmax = _press.max_global()
pressmin = _press.min_global()


# In[ ]:




# 
# ## scratch

# In[185]:

import h5py


# In[186]:

fname = filePath + 'swarm.h5'

if uw.rank()==0:
    with h5py.File(fname,'r') as hf:
        #print('List of arrays in this file: \n', hf.keys())
        data = hf.get('data')
        np_data = np.array(data)

    sbx =  np_data[:,0]
    sby =  np_data[:,1]

    #sbx =  shearbandswarm.particleCoordinates.data[:,0]
    #sby =  shearbandswarm.particleCoordinates.data[:,1]

    z = np.polyfit(sbx, sby, 1)
    p = np.poly1d(z)
    
    #newcoords = np.column_stack((sbx, p(sbx)))
    angle = math.atan(z[0])*(180./math.pi)
    45. - dp.fa
    
comm.barrier()




if rank==0:
    dydx = p[1]
    const = p[0]
else:
    dydx = 1.
    const = 0.

# share value of dydx
comm.barrier()
dydx = comm.bcast(dydx, root = 0)
const = comm.bcast(const, root = 0)
comm.barrier()



print(dydx, const)


# In[187]:

xs = np.linspace(0, -1., 100)
newcoords = np.column_stack((xs, dydx*xs + const )) 


swarmCustom = uw.swarm.Swarm(mesh)
swarmCustom.add_particles_with_coordinates(newcoords )


# In[188]:

figTest2 = glucifer.Figure( figsize=(1600,400), boundingBox=((-2.0, 0.0, 0.0), (2.0, 1.0, 0.0)) )
figTest2.append( glucifer.objects.Points(shearbandswarm, pointSize=2.0, colourBar=False) )

figTest2.append( glucifer.objects.Points(swarmCustom , pointSize=4.0,colourBar=False) )

figTest2.append( glucifer.objects.Points(swarm,strainRate_2ndInvariantFn, pointSize=3.0, valueRange=[1e-3, 2.]) )


#figTest2.show()

figTest2.save_image(imagePath +  "figTest2.png")


# In[ ]:




# In[206]:

import csv

if uw.rank()==0:

    someVals = [rmsint, shearArea ,shearPressure, 
                backgroundArea, backgroundPressure, viscmin, viscmax, eiimin, eiimax, angle,vdint, shearVd, backgroundVd, pressmin, pressmax  ] 

    with open(os.path.join(outputPath, 'metrics.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(someVals)
    with open(os.path.join(outputPath, 'solver.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow(res1Vals)
        writer.writerow(res2Vals)
        writer.writerow(res3Vals)
    with open(os.path.join(outputPath, 'params.csv'), 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        writer.writerow([dp[i] for i in sorted(dp.keys())]) #this makes sure the params are written in order of the sorted keys (i.e an order we can reproduce)
        writer.writerow([ndp[i] for i in sorted(ndp.keys())])
        writer.writerow([md[i] for i in sorted(md.keys())])


# test = np.array([0.5, 0.5])
# 
# ys = np.linspace(0, 1, 10)
# xs = np.zeros(10)
# 
# points = np.column_stack((xs, ys))
# 
# 
# ix, weights = nn_evaluation(swarm, points, n=1, weighted=False)
# ix, weights
# 
# visc = viscosityFn.evaluate(swarm)[ix]
# 
# 
# %pylab inline
# plt.scatter(ys, visc)

# In[207]:

#angle


# In[208]:

#45 - dp.fa/2


# In[ ]:



