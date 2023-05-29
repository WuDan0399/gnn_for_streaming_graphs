from os.path import join

from torch_geometric import datasets
from torch_geometric import utils

from utils import *


def FakeHiddenStateLength(feature_length: int, nlayers: int):
    decay_coeff = 0.8
    min_len: int = 50
    hidden_len = [feature_length]
    for i in range(1, nlayers):
        hidden_len.append(max(int(hidden_len[i - 1] * decay_coeff), min_len))
    return np.array(hidden_len)


def FirstLayer(degree, total_degree, num_effected_end, feature_length):
    # Bytes to be fetched for one vertex
    byte_per_vertex = feature_length * data_type_length
    # probability to effect a vertex adding\deleting a random edge, sorted by id
    prob = degree / total_degree
    # number of Bytes to be fetched for aggregation if a vertex changes its feature, sorted by id.
    memory_access = byte_per_vertex * degree
    computation_aggregation = feature_length * degree
    # fetch history and src's feature.
    optimized_memory_access_per_add = feature_length * (history_length + 1)
    # Expected number of memory access for one edge change.
    expected_memory_access_per_change = num_effected_end * \
        np.sum(prob * memory_access) / 1024 / 1024
    total_memory_access = np.sum(memory_access) / 1024 / 1024 / 1024
    return optimized_memory_access_per_add, expected_memory_access_per_change, total_memory_access


@measure_time
def EdgelistToGraph(edges):
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        if v not in graph:
            graph[v] = []
        graph[u].append(v)
        # graph[v].append(u) # edge directed
    return graph


def WaveFront(graph, visited, prev_wavefront):
    # similar to BFS, get the next level unvisited vertices
    curr_wavefront = set()
    for v in prev_wavefront:
        for dst in graph[v]:
            if dst not in visited:
                visited.add(dst)
                curr_wavefront.add(dst)
    return curr_wavefront


# def Whole(hidden_state_length, n_vertex: int):
#     # Recompute the whole graph
#     # Assume 20% locality (less message fetched)
#     # est_fetch_times = max(1, 0.8 * (edge_index.shape[0] / n_vertex))
#     # print("Estimated average number of times to fetch a whole graph: {}".format(est_fetch_times))
#     # fetched_numbers_whole = (est_fetch_times * n_vertex) * hidden_state_length
#     fetched_numbers_whole = n_vertex * hidden_state_length
#     return fetched_numbers_whole


def NonOverlapped(graph, sample_edges, hidden_state_length, nlayers: int, num_effected_end: int):
    # Recomputed the affected area, process each edge independently
    fetched_vertices = np.zeros((nlayers, sample_edges.shape[0]))
    for it_edge, edge in enumerate(sample_edges):
        affected = set(edge[1:]) if num_effected_end == 1 else set(edge)
        visited = set(affected)
        prev_wavefront = affected
        for it_layer in range(0, nlayers):
            curr_wavefront = WaveFront(graph, visited, prev_wavefront)
            prev_wavefront = curr_wavefront
            # affected for next layer, also fetched for this layer.
            affected = affected.union(curr_wavefront)
            fetched_vertices[it_layer, it_edge] = len(affected)
    return np.sum(fetched_vertices, axis=1)


@measure_time
def Mergable(graph, sample_edges:np.ndarray, nlayers: int, num_effected_end: int):
    ## Verified with nx.bfs_tree
    ## Sample verification code
    # tmp_graph = nx.DiGraph()
    # tmp_graph.add_edges_from(np.transpose(data.edge_index.numpy()))
    # theoretical_affected2['conv2'] = len(affected_nodes_by_layer(tmp_graph, sample_nodes, 2))
    ##

    # Recomputed the affected area for all changes
    fetched_vertices_each_layer = {}

    for edge in sample_edges:
        affected = set(edge[1:]) if num_effected_end == 1 else set(edge)
        visited = set(affected)
        prev_wavefront = affected
        for it_layer in range(0, nlayers):
            if it_layer not in fetched_vertices_each_layer:
                fetched_vertices_each_layer[it_layer] = set()
            curr_wavefront = WaveFront(graph, visited, prev_wavefront)
            prev_wavefront = curr_wavefront
            # affected for next layer, also fetched for this layer.
            affected = affected.union(curr_wavefront)
            fetched_vertices_each_layer[it_layer] = fetched_vertices_each_layer[it_layer].union(
                affected)

    num_fetched_vertices_each_layer = [
        len(fetched_vertices_each_layer[layer]) for layer in range(nlayers)]
    return num_fetched_vertices_each_layer


def Plot(merged_affected, affected, whole, nlayers, title, file_title):
    # set width of bar
    barWidth = 0.25

    fig, ax = plt.subplots(layout='constrained')

    # Set position of bar on X axis
    br = np.arange(nlayers)

    datas = [merged_affected, affected, whole]
    labels = ['merged affected area', 'affected area', 'whole']
    bar_labels = [["{:.2e}".format(x) for x in data] for data in datas]

    # Make the plot
    for i, data, label, bar_label in zip(range(len(labels)), datas, labels, bar_labels):
        offset = barWidth * i
        rects = ax.bar(br + offset, data, barWidth, label=label)
        ax.bar_label(rects, bar_label, padding=3)
        # rects1 = ax.bar(br1, merged_affected, color='r', width=barWidth,
        #     edgecolor='grey', label='merged affected area')

    # Adding Xticks
    ax.set_yscale('log')
    plt.xlabel('Layer', fontweight='bold', fontsize=15)
    plt.ylabel('Data Fetched (MB)', fontweight='bold', fontsize=15)
    plt.xticks([r + barWidth for r in range(nlayers)], range(1, nlayers + 1))
    plt.title(title)

    plt.legend()
    plt.savefig(join("", file_title + ".pdf"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = general_parser(parser)
    dataset = load_dataset(args)

    ##### Configurations #####
    history_length = 3
    data_type_length = 4  # suppose floating point 4B
    batch_ratio = 0.005  # 0.5% of the total edge change.
    ##### Configurations #####

    # # Add parser to select dataset
    # parser = argparse.ArgumentParser(
    #     description="Indicate the dataset through argument.")
    # parser.add_argument("-d", "--dataset", nargs='?', default="cora",
    #                     help="select dataset. (cora/yelp/amazon/full)")
    # parser.add_argument("-l", "--nlayers", nargs='?', default=10,
    #                     type=int, help="number of layers")
    # parser.add_argument("-i", "--initial", nargs='?', default=70.0, type=float,
    #                     help="percentage of edges loaded at the begining. [0.0, 100.0]")
    # parser.add_argument("-pb", "--perbatch", nargs='?', default=0.5, type=float,
    #                     help="percentage of edges loaded per batch. [0.0, 100.0]")
    # parser.add_argument("-nb", "--numbatch", nargs='?',
    #                     default=50, type=int, help="number of batches")
    # args = parser.parse_args()


    feature_length, n_vertex, num_effected_end = 0, 0, 0
    if args.dataset == "cora":
        edge_index = datasets.CitationFull(
            "./datasets/CitationFull", "Cora").get(0).edge_index
        feature_length = 8710
        n_vertex = 19793
    elif args.dataset == "yelp":
        edge_index = datasets.Yelp("./datasets/Yelp").get(0).edge_index
        feature_length = 300
        n_vertex = 716847
    elif args.dataset == "amazon":
        edge_index = datasets.AmazonProducts(
            "./datasets/Amazon").get(0).edge_index
        feature_length = 200
        n_vertex = 1569960

    # # test input
    # edge_index = datasets.CitationFull("./datasets/CitationFull", "Cora").get(0).edge_index
    # feature_length = 8710
    # n_vertex = 19793
    # args.nlayers = 10

    file_name = f"expected_comp_mem_{args.dataset}.txt"
    folder = f"results"
    with open(join(folder, file_name), 'w') as f:
        if utils.is_undirected(edge_index):
            print("Undirected Graph")
            id, degree = np.unique(
                edge_index[0], return_counts=True)  # undirected graph
            total_degree = edge_index.shape[1]
            total_edge = total_degree / 2
            num_effected_end = 2
        else:
            print("Directed Graph")
            # in degree for directed graph
            id, degree = np.unique(edge_index[1], return_counts=True)
            # out_id, in_degree = np.unique(graph.edge_index[1], return_counts=True)
            total_degree = edge_index.shape[1]
            total_edge = total_degree
            num_effected_end = 1

        edge_index = np.transpose(edge_index.numpy())  # to shape (nedges, 2)
        batch_size = int(batch_ratio * total_edge)
        hidden_state_length = FakeHiddenStateLength(
            feature_length, args.nlayers)
        f.write("Length of input in each layer: \n")
        for i in range(0, args.nlayers):
            f.write("{}: {} \n".format(i + 1, hidden_state_length[i]))
        sample_edges = edge_index[np.random.choice(edge_index.shape[0], batch_size, replace=False)]

        # Treat each change independently, no data reuse among changes. Calculate cost for recompute the whole graph
        graph = EdgelistToGraph(edge_index)

        fetched_vertex_whole = n_vertex

        fetched_vertex_batch_affected = NonOverlapped(graph, sample_edges,
                                                      hidden_state_length, args.nlayers,
                                                      num_effected_end)

        # Treat batch of changes together, full data reuse among changes
        fetched_vertex_batch_affected_merged = Mergable(graph, sample_edges, args.nlayers,
                                                        num_effected_end)

        fetch_data_whole = fetched_vertex_whole * hidden_state_length *\
            data_type_length / 1024 / 1024 / 1024  # in GB
        fetch_data_batch_affected = fetched_vertex_batch_affected * hidden_state_length * \
            data_type_length / 1024 / 1024  # in MB
        fetched_data_batch_affected_merged = fetched_vertex_batch_affected_merged * hidden_state_length * \
            data_type_length / 1024 / 1024  # in MB

        f.write("[Vertices] Merged Affected, Affected, Whole\n")
        f.write(f"Initially affected number of edges \ vertices: {len(sample_edges)}, {len(set(sample_edges.flatten().tolist()))}\n")
        for i in range(0, args.nlayers):
            f.write("{}: {}, {}, {}\n".format(i + 1, fetched_vertex_batch_affected_merged[i],
                                              int(fetched_vertex_batch_affected[i]), fetched_vertex_whole))
        f.write("[Data] Merged Affected(MB), Affected(MB), Whole(GB)\n")
        for i in range(0, args.nlayers):
            f.write("{}: {}, {}, {}\n".format(i + 1, fetched_data_batch_affected_merged[i],
                                                    fetch_data_batch_affected[i], fetch_data_whole[i]))
        f.write("[Ratio] Affected/Merged Affected, Whole/Merged Affected\n")
        for i in range(0, args.nlayers):
            f.write("{}: 1, {}, {}\n".format(i + 1,
                                             fetch_data_batch_affected[i] /
                                             fetched_data_batch_affected_merged[i],
                                             fetch_data_whole[i] / fetched_data_batch_affected_merged[i]))

        # est_fetch_times = max(1, 0.8 * (edge_index.shape[0] / n_vertex))
        # f.write("Estimated average number of times to fetch a whole graph: {}".format(est_fetch_times))

        title = "{}: {}-layer GCN".format(args.dataset, args.nlayers)
        file_title = "{}_{}_layer".format(args.dataset, args.nlayers)
        Plot(fetched_data_batch_affected_merged, fetch_data_batch_affected, fetch_data_whole * 1024, args.nlayers,
             title, file_title)
