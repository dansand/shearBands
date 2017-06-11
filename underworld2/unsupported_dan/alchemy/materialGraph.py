import operator
import uuid
import underworld as uw
from underworld import function as fn


class MatGraph(object):

    node_dict_factory = dict
    adjlist_dict_factory = dict
    edge_attr_dict_factory = dict

    def __init__(self, data=None, **attr):


        self.node_dict_factory = ndf = self.node_dict_factory
        self.adjlist_dict_factory = self.adjlist_dict_factory
        self.edge_attr_dict_factory = self.edge_attr_dict_factory

        self.graph = {}   # dictionary for graph attributes
        self.node = ndf()  # empty node attribute dict
        self.adj = ndf()  # empty adjacency dict

        self.pred = ndf()  # predecessor
        self.succ = self.adj  # successor
        # attempt to load graph with data
        if data is not None:
            convert.to_networkx_graph(data, create_using=self)
        # load graph attributes (must be after convert)
        self.graph.update(attr)
        self.edge = self.adj
        self.edge = self.adj

    def is_directed(self):
        """Return True if graph is directed, False otherwise."""
        return True

    @property
    def name(self):
        return self.graph.get('name', '')

    @name.setter
    def name(self, s):
        self.graph['name'] = s


    def __iter__(self):
        """Iterate over the nodes. Use the expression 'for n in G'.

        Returns
        -------
        niter : iterator
            An iterator over all nodes in the graph.

        Examples
        --------
        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_path([0,1,2,3])
        """
        return iter(self.node)

    def __contains__(self, n):
        """Return True if n is a node, False otherwise. Use the expression
        'n in G'.

        Examples
        --------
        >>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
        >>> G.add_path([0,1,2,3])
        >>> 1 in G
        True
        """
        try:
            return n in self.node
        except TypeError:
            return False

    def __len__(self):
        """Return the number of nodes. Use the expression 'len(G)'.

        """
        return len(self.node)

    def __getitem__(self, n):
        """Return a dict of neighbors of node n.  Use the expression 'G[n]'.
        ** This allows the indexing form  G[n]

        """
        return self.adj[n]




    def add_node(self, n, attr_dict=None, **attr):
        """Add a single node n and update node attributes.

        """
        # set up attribute dict
        if attr_dict is None:
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise ValueError(\
                    "The attr_dict argument must be a dictionary.")
        if n not in self.succ:
            self.succ[n] = self.adjlist_dict_factory()
            self.pred[n] = self.adjlist_dict_factory()
            self.node[n] = attr_dict
        else: # update attr even if node already exists
            self.node[n].update(attr_dict)

    def add_nodes_from(self, nodes, **attr):
        """Add multiple nodes."""
        for n in nodes:
            # keep all this inside try/except because
            # CPython throws TypeError on n not in self.succ,
            # while pre-2.7.5 ironpython throws on self.succ[n]
            try:
                if n not in self.succ:
                    self.succ[n] = self.adjlist_dict_factory()
                    self.pred[n] = self.adjlist_dict_factory()
                    self.node[n] = attr.copy()
                else:
                    self.node[n].update(attr)
            except TypeError:
                nn,ndict = n
                if nn not in self.succ:
                    self.succ[nn] = self.adjlist_dict_factory()
                    self.pred[nn] = self.adjlist_dict_factory()
                    newdict = attr.copy()
                    newdict.update(ndict)
                    self.node[nn] = newdict
                else:
                    olddict = self.node[nn]
                    olddict.update(attr)
                    olddict.update(ndict)


    def remove_node(self, n):

        """Remove node n.


        """
        try:
            nbrs=self.succ[n]
            del self.node[n]
        except KeyError: # ValueError if n not in self
            raise ValueError("The node %s is not in the digraph."%(n,))
        for u in nbrs:
            del self.pred[u][n] # remove all edges n-u in digraph
        del self.succ[n]          # remove node from succ
        for u in self.pred[n]:
            del self.succ[u][n] # remove all edges n-u in digraph
        del self.pred[n]          # remove node from pred

    def remove_nodes_from(self, nodes):

        """Remove multiple nodes.

        """
        adj = self.adj
        for n in nodes:
            try:
                del self.node[n]
                for u in list(adj[n].keys()):   # keys() handles self-loops
                    del adj[u][n]  # (allows mutation of dict in loop)
                del adj[n]
            except KeyError:
                pass

    def nodes(self, data=False):

        return list(self.nodes_iter(data=data))


    def nodes_iter(self, data=False):

        if data:
            return iter(self.node.items())
        return iter(self.node)


    def add_edge(self, u, v, attr_dict=None, **attr):
        """Add an edge between u and v.

        """
        # set up attribute dictionary
        if attr_dict is None:
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise TypeError(\
                    "The attr_dict argument must be a dictionary.")
        # add nodes
        if u not in self.succ:
            self.succ[u]= self.adjlist_dict_factory()
            self.pred[u]= self.adjlist_dict_factory()
            self.node[u] = {}
        if v not in self.succ:
            self.succ[v]= self.adjlist_dict_factory()
            self.pred[v]= self.adjlist_dict_factory()
            self.node[v] = {}
        # add the edge
        datadict=self.adj[u].get(v,self.edge_attr_dict_factory())
        datadict.update(attr_dict)
        self.succ[u][v]=datadict
        self.pred[v][u]=datadict

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        """Add all the edges in ebunch.

        """
        # set up attribute dict
        if attr_dict is None:
            attr_dict=attr
        else:
            try:
                attr_dict.update(attr)
            except AttributeError:
                raise TypeError(\
                    "The attr_dict argument must be a dict.")
        # process ebunch
        for e in ebunch:
            ne = len(e)
            if ne==3:
                u,v,dd = e
                assert hasattr(dd,"update")
            elif ne==2:
                u,v = e
                dd = {}
            else:
                raise ValueError(\
                    "Edge tuple %s must be a 2-tuple or 3-tuple."%(e,))
            if u not in self.succ:
                self.succ[u] = self.adjlist_dict_factory()
                self.pred[u] = self.adjlist_dict_factory()
                self.node[u] = {}
            if v not in self.succ:
                self.succ[v] = self.adjlist_dict_factory()
                self.pred[v] = self.adjlist_dict_factory()
                self.node[v] = {}
            datadict=self.adj[u].get(v,self.edge_attr_dict_factory())
            datadict.update(attr_dict)
            datadict.update(dd)
            self.succ[u][v] = datadict
            self.pred[v][u] = datadict


    def remove_edge(self, u, v):
        """Remove the edge between u and v.

        """
        try:
            del self.succ[u][v]
            del self.pred[v][u]
        except KeyError:
            raise ValueError("The edge %s-%s not in graph."%(u,v))

    def remove_edges_from(self, ebunch):
        """Remove all edges specified in ebunch.

        """
        for e in ebunch:
            (u,v)=e[:2]  # ignore edge data
            if u in self.succ and v in self.succ[u]:
                del self.succ[u][v]
                del self.pred[v][u]


    def edges(self, nbunch=None, data=False, default=None):
        """Return a list of edges.

        """
        return list(self.edges_iter(nbunch, data, default))

    def edges_iter(self, nbunch=None, data=False, default=None):
        """Return an iterator over the edges.

        """
        if nbunch is None:
            nodes_nbrs=self.adj.items()
        else:
            nodes_nbrs=((n,self.adj[n]) for n in self.nbunch_iter(nbunch))
        if data is True:
            for n,nbrs in nodes_nbrs:
                for nbr,ddict in nbrs.items():
                    yield (n,nbr,ddict)
        elif data is not False:
            for n,nbrs in nodes_nbrs:
                for nbr,ddict in nbrs.items():
                    d=ddict[data] if data in ddict else default
                    yield (n,nbr,d)
        else:
            for n,nbrs in nodes_nbrs:
                for nbr in nbrs:
                    yield (n,nbr)

    # alias out_edges to edges
    #out_edges_iter=edges_iter
    #out_edges=list(self.edges_iter(nbunch, data, default))

    #################
     #Underworld-related stuff
    ################


    def add_transition(self, nodes, function, FnOperator, value, combineby = 'and'):


        """Add a material trnasition between node[0], node[1].

        Parameters
        ----------
        nodes : a list of 2 numbers (integers) representing the material indexes
        function : uw2 function
        FnOperator : logical operator from operator package (e.g operator.lt, operator.gt)

        Examples
        --------
        >>>DG = MatGraph() #Setup the MatGraph object
        >>>material_list = [1,2,3,4]
        >>>DG.add_nodes_from(material_list)
        >>>DG.add_transition((1,2), xFn, operator.lt, 0.5)


        Notes
        -----
        ...
        """

        #only greater than or less than comparisons are supported for conditons
        if not operator.or_(FnOperator.__name__ == ('lt'), FnOperator.__name__ == ('gt')):
            raise AssertionError("FnOperator must be either operator.lt or operator.gt", FnOperator)

        firstEdge = True
        try:
            self[nodes[0]][nodes[1]] #see if the node exists
            #get names of previous condition dict:
            prevdname = self[nodes[0]][nodes[1]].keys()[0]
            firstEdge = False
        except:
            self.add_node(nodes[0])
            self.add_node(nodes[1])
            self.add_edges_from([nodes])
        #create a random name for dictionary (we need to have a different key for each condition on the graph edge)
        dname = uuid.uuid4()
        self[nodes[0]][nodes[1]][dname] = {}
        self[nodes[0]][nodes[1]][dname]['function'] = function
        self[nodes[0]][nodes[1]][dname]['operator'] =  FnOperator
        self[nodes[0]][nodes[1]][dname]['value'] =  value
        self[nodes[0]][nodes[1]][dname]['combineby'] = 'and'
        if combineby == 'or':
            self[nodes[0]][nodes[1]][dname]['combineby'] =  'or'
        if not firstEdge:
            assert self[nodes[0]][nodes[1]][dname]['combineby'] == self[nodes[0]][nodes[1]][prevdname]['combineby'], "if the graph has multiple conditions on an edge, provided 'combineby' string must be identical to avoid ambiguity."


    def build_condition_list(self, materialVariable):

        """Add a material trnasition between node[0], node[1].

        Parameters
        ----------
        materialVariable : uw2 swarm variable
            the variable containing the material indexes for the model


        Examples
        --------
        >>>DG.build_condition_list(materialVariable)
        >>>materialVariable.data[:] = fn.branching.conditional(DG.condition_list).evaluate(swarm)

        Notes
        -----
        ...
        """

        self.condition_list = [] #empty the condition list
        dm = 1e-6
        for node in self.nodes(): #Loop through nodes of graph
            for otherNode in self[node].keys(): #loop through all egdes from a given node
                #if node < otherNode:
                #this returns true for all particles with materialIndex == node (direct comparison isn't supported)
                checkFrom = operator.and_((materialVariable > (node - dm) ),
                           (materialVariable < (node + dm) ))
                condIt = 0
                for cond in self[node][otherNode].keys(): #loop through all conditions attached to the graph edge
                    op = self[node][otherNode][cond]['operator']    #
                    fun = self[node][otherNode][cond]['function']   #{extract function, operator, value}
                    val = self[node][otherNode][cond]['value']      #
                    condExp = op(fun, val)  #Now provide the function & value to the operator, return result as a variable
                    if condIt == 0:
                        totCond = condExp #if this is the first condition, assign to totCond
                    else: #if this is the NOT first condition, combine conditin with previous totCond (using AND or OR)
                        if self[node][otherNode].values()[0]['combineby'] == 'or':
                            totCond = operator.or_(totCond, condExp)
                        else:
                            totCond = operator.and_(totCond, condExp)
                    condIt += 1

                #When we pass this on to fn.branching.conditional, we only want to apply it to paticles where
                # matIndex == node, which occurs where checkFrom == True, 1
                combCond = operator.and_(totCond, checkFrom)
                #combCond = totCond
                self.condition_list.append(((combCond), otherNode))
        self.condition_list.append((True ,          materialVariable)) #if no conditions are true, return current matId
