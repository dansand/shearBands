
import numpy as np
import underworld as uw

from scipy.spatial import cKDTree as kdTree

class markerLine2D(object):
    """
    All the bits and pieces needed to define a marker surface (in 2D) from a string of points
    """


    def __init__(self, mesh, velocityField, pointsX, pointsY, fthickness, fID, insidePt=(0.0,0.0)):


        # Marker swarms are probably sparse, and on most procs will have nothing to do
        # if there are no particles (or not enough to compute what we need)
        # then set this flag and return appropriately. This can be checked once the swarm is
        # populated.

        self.empty = False

        # Should do some checking first

        self.mesh = mesh
        self.velocity = velocityField
        self.thickness = fthickness
        self.ID = fID
        self.insidePt = insidePt
        self.director = None

        # Set up the swarm and variables on all procs

        self.swarm = uw.swarm.Swarm( mesh=self.mesh, particleEscape=True )
        self.director = self.swarm.add_variable( dataType="double", count=2)
        self._swarm_advector = uw.systems.SwarmAdvector( swarm=self.swarm,
                                                         velocityField=self.velocity, order=2 )

        self.swarm.add_particles_with_coordinates(np.stack((pointsX, pointsY)).T)
        self.director.data[...] = 0.0

        self._update_kdtree()
        self._update_surface_normals()

        return



    def add_points(self, pointsX, pointsY):

        self.swarm.add_particles_with_coordinates(np.stack((pointsX, pointsY)).T)

        self.rebuild()


    def rebuild(self):

        self._update_kdtree()
        self._update_surface_normals()

        return


    def _update_kdtree(self):

        self.empty = False
        self.swarm.shadow_particles_fetch()

        dims = self.swarm.particleCoordinates.data.shape[1]

        pc = np.append(self.swarm.particleCoordinates.data,
                       self.swarm.particleCoordinates.data_shadow)

        all_particle_coords = pc.reshape(-1,dims)

        if len(all_particle_coords) < 3:
            self.empty = True
            self.kdtree = lambda x: float('inf')
        else:
            self.kdtree = kdTree(all_particle_coords)

        return


    def advection(self, dt):
        """
        Update marker swarm particles as material points and rebuild data structures
        """
        self._swarm_advector.integrate( dt, update_owners=True)
        self.swarm.shadow_particles_fetch()

        self._update_kdtree()
        self._update_surface_normals()

        uw.barrier()

        return



    def compute_marker_proximity(self, coords, distance=None):
        """
        Build a mask of values for points within the influence zone.
        """
        #can be important for parallel
        self.swarm.shadow_particles_fetch()


        if not distance:
            distance = self.thickness

        if self.empty:
            return np.empty((0,1)), np.empty(0, dtype="int")

        d, p   = self.kdtree.query( coords, distance_upper_bound=distance )

        fpts = np.where( np.isinf(d) == False )[0]

        proximity = np.zeros((coords.shape[0],1))
        proximity[fpts] = self.ID

        return proximity, fpts


    def compute_normals(self, coords, thickness=None):



        # make sure this is called by all procs including those
        # which have an empty self

        self.swarm.shadow_particles_fetch()

        if thickness==None:
            thickness = self.thickness

        # Nx, Ny = _points_to_normals(self)

        if self.empty:
            return np.empty((0,2)), np.empty(0, dtype="int")

        d, p   = self.kdtree.query( coords, distance_upper_bound=thickness )

        fpts = np.where( np.isinf(d) == False )[0]
        director = np.zeros_like(coords)

        if uw.nProcs() == 1 or self.director.data_shadow.shape[0] == 0:
            fdirector = self.director.data
            #print('1')
        elif self.director.data.shape[0] == 0:
            fdirector = self.director.data_shadow
            #print('2')
        else:
            fdirector = np.concatenate((self.director.data,
                                    self.director.data_shadow))
            #print('3')

        director[fpts] = fdirector[p[fpts]]

        return director, fpts


    def compute_signed_distance(self, coords, distance=None):

        # make sure this is called by all procs including those
        # which have an empty self

        #can be important for parallel
        self.swarm.shadow_particles_fetch()

        if not distance:
            distance = self.thickness

        #print(self.director.data.shape, self.director.data_shadow.shape)

        if self.empty:
            return np.empty((0,1)), np.empty(0, dtype="int")

        if uw.nProcs() == 1 or self.director.data_shadow.shape[0] == 0:
            fdirector = self.director.data
            print('1')
        elif self.director.data.shape[0] == 0:
            fdirector = self.director.data_shadow
            print('2')
        else:
            fdirector = np.concatenate((self.director.data,
                                    self.director.data_shadow))
            print('3')

        d, p  = self.kdtree.query( coords, distance_upper_bound=distance )

        fpts = np.where( np.isinf(d) == False )[0]

        director = np.zeros_like(coords)  # Let it be zero outside the region of interest
        director = fdirector[p[fpts]]

        #print('dir. min', np.linalg.norm(director, axis = 1).min())

        vector = coords[fpts] - self.kdtree.data[p[fpts]]


        dist = np.linalg.norm(vector, axis = 1)

        signed_distance = np.empty((coords.shape[0],1))
        signed_distance[...] = np.inf

        sd = np.einsum('ij,ij->i', vector, director)
        signed_distance[fpts,0] = sd[:]
        #signed_distance[:,0] = d[...]

        #return signed_distance, fpts
        #signed_distance[fpts,0] = dist[:]
        return signed_distance , fpts


    def _update_surface_normals(self):
        """
        Rebuilds the normals for the string of points
        """

        # This is the case if there are too few points to
        # compute normals so there can be values to remove

        #can be important for parallel
        self.swarm.shadow_particles_fetch()

        if self.empty:
            self.director.data[...] = 0.0
        else:

            particle_coords = self.swarm.particleCoordinates.data

            #these will hold the normal vector compenents
            Nx = np.empty(self.swarm.particleLocalCount)
            Ny = np.empty(self.swarm.particleLocalCount)

            for i, xy in enumerate(particle_coords):
                r, neighbours = self.kdtree.query(particle_coords[i], k=3)

                # neighbour points are neighbours[1] and neighbours[2]

                XY1 = self.kdtree.data[neighbours[1]]
                XY2 = self.kdtree.data[neighbours[2]]

                dXY = XY2 - XY1

                Nx[i] =  dXY[1]
                Ny[i] = -dXY[0]

                if (self.insidePt):
                    sign = np.sign((self.insidePt[0] - xy[0]) * Nx[i] +
                                   (self.insidePt[1] - xy[1]) * Ny[i])
                    Nx[i] *= sign
                    Ny[i] *= sign


            for i in range(0, self.swarm.particleLocalCount):
                scale = 1.0 / np.sqrt(Nx[i]**2 + Ny[i]**2)
                Nx[i] *= scale
                Ny[i] *= scale


            self.director.data[:,0] = Nx[:]
            self.director.data[:,1] = Ny[:]

        return
