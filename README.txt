# Preparation:
1. conda environment: baseline and modified.
        baseline: change aggr parameter in __init__ function to "min" for SAGEConv and 'max' for GINConv
        modifed: a. change aggr parameter in __init__ function same as baseline
                 b. add save_int parameter in __init__ to save intermediate values
                 c. add code to save and return intermediate values when not training and save_int==True



GCN.py [baselinePyG]
    # patience epochs interval #
    Full graph GCN training.

GCN_neighbor_loader.py [baselinePyG]
    # patience epochs interval #
    1. Minibatch GCN training, if no trained model, sampling with 10 neighbors
    2. Timing of each phase during inference, if trained model exists, graph sampled with interval

GCN_timing_original.py  [baselinePyG]
    # range perbatch stream(if affected) #
    1. Time measure for full graph inference
    2. Time measure for affected part inference (repeat 10 times)

GCN_get_intermediate_result.py [gnnEnv]
    # perbatch, interval, save_int, stream #
    Get intermediate result for 1)sampled graph with interval size and 2)graph with 'perbatch' edges missing.

my_GCN.py [gnnEnv]
    # perbatch, save_int, stream #
    Use incremental update to get the result. Use --save_int if the intermediate result is not saved previously.


ignite.py [gnnEnv]
    Base framework for ignite method

ignite_gcn.py [gnnEnv]
    # perbatch, save_int, stream, model #
    Customized for GCN, extending base ignite framwork.

ignite_sage.py [gnnEnv] 
    # perbatch, save_int, stream, model #
    Customized for SAGE, extending base ignite framwork. No sampling implemented.

ignite_gcn.py [gnnEnv]
    # binary, perbatch, save_int, stream, model #
    Customized for GIN, extending base ignite framwork.

pureGIN.py [baselinePyG]
    # binary, dataset, aggr #



test.py
    Functional Verification and Tests

utils.py
    task management, helper functions, almost all shared functions

GCN_batch_accuracy.py
    Relation between accuracy and batch size in streaming graph.

GCN_inf_dynamic.py
    Compare the real affected area with theoretical ones. Get output of each layer.

GCN_timing_inc.py
    Waiting to be implemented

pureGCN.py
    Pure GCN with no changes\printing\data saving.

theoretical_est.py
    Theoretical affected area v.s. full graph with change of batch size of streaming graph.