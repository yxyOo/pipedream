rm optimizer_graph_hierarchical.log
rm convert_graph_to_model.log
# python optimizer_graph_hierarchical.py -f ../profiler/translation/profiles/gnmt/graph.txt -n 4 \
#     --straight_pipeline --activation_compression_ratio 1 -o gnmt_search_partitioned 2>&1 >>  optimizer_graph_hierarchical.log
python optimizer_graph_hierarchical.py -f ../profiler/translation/profiles/gnmt_large/graph.txt -n 4 \
    --straight_pipeline --activation_compression_ratio 1 -o gnmt_large_repart_linear 2>&1 >>  optimizer_graph_hierarchical.log
python convert_graph_to_model.py -f gnmt_large_repart_linear/gpus=4.txt -n gnmt_large_repart_linear \
    -a gnmt_large_repart_linear -o ../runtime/translation/models/gnmt_large_repart_linear/gpus=4 --stage_to_num_ranks 0:1,1:1,2:1,3:1 2>&1 >> convert_graph_to_model.log
