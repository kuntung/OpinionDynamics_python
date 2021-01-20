import leidenalg as la
import igraph as ig

if __name__ == '__main__':
    g = ig.Graph.Barabasi(n=20, m=3)
    c = g.clusters()
    print(c)
    partition = la.find_partition(g, la.ModularityVertexPartition)
    ig.plot(partition).show()
