import numpy as np


def neighbour1Matrix(markerLine, k= False):

    """
    comment
    """

    #get the particle coordinates, on the order that the kdTree quuery naturally returns them

    #markerLine.rebuild() 
    all_particle_coords = markerLine.kdtree.data
    queryOut = markerLine.kdtree.query(all_particle_coords, k=markerLine.swarm.particleCoordinates.data.shape[0] )
    ids = queryOut[1]
    coords = all_particle_coords[ids]


    #build the matrix of neighbour -adjacency
    AN = np.zeros((all_particle_coords.shape[0],all_particle_coords.shape[0] ))

    #First add the nearest neighbour
    AN[ids[:,0],ids[np.arange(len(AN)), 1]] =  1

    return AN

def neighbour2Matrix(markerLine, k= False):

    """
    comment
    """

    #get the particle coordinates, on the order that the kdTree quuery naturally returns them

    all_particle_coords = markerLine.kdtree.data
    queryOut = markerLine.kdtree.query(all_particle_coords, k=markerLine.kdtree.data.shape[0] )
    ids = queryOut[1]
    coords = all_particle_coords[ids]

    #Now, make a vector array using tile
    pvector = all_particle_coords[ids[:,0]]
    pcoords = np.tile(pvector, (all_particle_coords.shape[0],1,1)).swapaxes(0,1)
    vectors = np.subtract(coords, pcoords)

    #Now we have to compute the inner product pair for the the nearest neighbour and all successive neighbours (we want to find one that is negative)

    #these are the x, y components of the nearest neighbours
    nnXVector = np.tile(vectors[:,1,0], (all_particle_coords.shape[0],1,1)).T.reshape(all_particle_coords.shape[0], all_particle_coords.shape[0])
    nnYVector = np.tile(vectors[:,1,1], (all_particle_coords.shape[0],1,1)).T.reshape(all_particle_coords.shape[0], all_particle_coords.shape[0])

    #now make the dot products
    xInnerCompare = (vectors[:,:,0] * nnXVector)
    yInnerCompare = (vectors[:,:,1] * nnYVector)
    dotProdCompare = xInnerCompare + yInnerCompare

    #find the first point for which the position vector has a negative dot product with the nearest neighbour

    negDots = dotProdCompare < 0.

    #Here's where we limit the search
    if k:
        negDots[:,k:] = False


    #this should the the column of the first negative entry. To see which particle this corresponds to
    #cols = np.argmax(negDots[:,2:], axis = 1) + 2
    cols = np.argmax(negDots[:,:], axis = 1)
    #if cols is zero, it means no obtuse neighbour was found - likely an end particle.
    #For now, set to first column (we'll delete this later)
    cols[cols == 0] = 0


    answer = ids[np.arange(all_particle_coords.shape[0]),cols]


    #build the matrix of neighbour -adjacency
    A0 = np.zeros((all_particle_coords.shape[0],all_particle_coords.shape[0] ))

    #now add the first subsequent neighbour that is obtuse to the first
    A0[ids[:,0],answer] =  1

    #Now remove diagonals - these were any particles where a nearest obtuse neighbour couldn't be found
    #diagIds = np.array(zip(np.arange(markerLine.kdtree.data.shape[0]), np.arange(markerLine.kdtree.data.shape[0])))
    #A0[diagIds[:,0], diagIds[:,1]] = 0
    np.fill_diagonal(A0, 0)


    return A0


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def neighbour2Matrix2(markerLine, angle = 70., k= 7):
    all_particle_coords = markerLine.kdtree.data
    queryOut = markerLine.kdtree.query(all_particle_coords, k=markerLine.kdtree.data.shape[0] )

    dists = queryOut[0]
    ids = queryOut[1]
    coords = all_particle_coords[ids]

    #build the angles matrix
    R = neighboursAngleMatrix(markerLine)

    #here we penalise distances as their angle increases from 180. - this wieghing value is pretty arbitary, derived from trial-and-error
    fac = 10*dists[:,1].mean()
    weightedDists = dists + fac*(1. - gaussian(R, np.pi, sig=np.deg2rad(angle)))

    #get the ids of the smallest distance after the weighting has been applied
    cols = np.abs(weightedDists)[:,0:k].argmin(axis = 1)
    cols[np.where(cols==1)] = 0 #if any 1s have appeared make them zeros. (zeros will mean couln't find a neighbour)
    N2ids =  ids[np.arange(len(cols)),cols]


    #build the matrix of neighbour -adjacency
    A0 = np.zeros((all_particle_coords.shape[0],all_particle_coords.shape[0] ))

    #now add the subsequent neighbour we have identified though distance - angle weighting
    A0[np.arange(len(cols)),N2ids] =  1

    #zero the diagonals - these rows / partcles have no second neighbour
    np.fill_diagonal(A0, 0)

    return A0

def laplacianMatrix(markerLine, A):
    """
    """

    dims = markerLine.kdtree.data.shape[1]

    #Get neighbours
    all_particle_coords = markerLine.kdtree.data
    #queryOut = markerLine.kdtree.query(all_particle_coords, k=dims + 1)
    #neighbours = queryOut[1][:,1:]


    #create 2s on the diagonal
    L = 2.*np.eye(all_particle_coords.shape[0])

    #set all neighbours to -1
    L[A == 1] = -1
    #Find rows that only have one neighbour (these are probably/hopefully endpoints)
    mask = np.where(A.sum(axis=1) == 1)
    #Set those rows to zero
    L[mask,:]  = 0
    #And set the diagonal back to 2. (The Laplacian operator should just return the particle position)
    #L[mask,mask] = 2

    return 0.5*L #right?


def pairDistanceMatrix(markerLine):
    """
    """
    partx = markerLine.kdtree.data[:,0]
    party = markerLine.kdtree.data[:,1]
    dx = np.subtract.outer(partx , partx )
    dy = np.subtract.outer(party, party)
    distanceMatrix = np.hypot(dx, dy)

    return distanceMatrix


def neighboursAngleMatrix(markerLine):

    """
    """

    all_particle_coords = markerLine.kdtree.data
    queryOut = markerLine.kdtree.query(all_particle_coords, k=markerLine.kdtree.data.shape[0] )
    ids = queryOut[1]
    coords = all_particle_coords[ids]

    #print("Shape coords", str(coords.shape))


    pvector = all_particle_coords[ids[:,0]]
    pcoords = np.tile(pvector, (all_particle_coords.shape[0],1,1)).swapaxes(0,1)
    vectors = np.subtract(coords, pcoords)

    #these are the x, y components of the nearest neighbours
    nnXVector = np.tile(vectors[:,1,0], (all_particle_coords.shape[0],1,1)).T.reshape(all_particle_coords.shape[0], all_particle_coords.shape[0])
    nnYVector = np.tile(vectors[:,1,1], (all_particle_coords.shape[0],1,1)).T.reshape(all_particle_coords.shape[0], all_particle_coords.shape[0])

    #now make the dot products
    xInnerCompare = (vectors[:,:,0] * nnXVector)
    yInnerCompare = (vectors[:,:,1] * nnYVector)
    dotProdCompare = xInnerCompare + yInnerCompare


    nearNeigbourNorm = np.linalg.norm(np.tile(vectors[:,1], (all_particle_coords.shape[0],1,1)), axis = 2)
    otherNbsNorms = np.linalg.norm(vectors, axis = 2)
    normMult = nearNeigbourNorm.T*otherNbsNorms #Tranpose here because of sloppiness in above line
    cosThetas = np.divide(dotProdCompare,normMult)
    angles = np.arccos(cosThetas)
    pi2mask = angles>np.pi
    angles[pi2mask] -= np.pi

    #assume that nans are effectively zero
    return np.nan_to_num(angles)
