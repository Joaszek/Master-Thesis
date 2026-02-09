from Elliptic2Explorer import Elliptic2Explorer

"""Example usage with all analysis functions."""
print("\n" + "=" * 80)
print("ELLIPTIC2 COMPLETE ANALYSIS - POLARS (FIXED VERSION)")
print("=" * 80)

explorer = Elliptic2Explorer(data_path="./dataset/archive")

# Load with sampling (1% for speed)
print("\n[1/6] Loading data...")
explorer.load_data_sample(sample_fraction=0.1)

# Analyze background graph
print("\n[2/6] Analyzing background graph...")
bg_analysis = explorer.analyze_background_graph()

# Analyze subgraphs
print("\n[3/6] Analyzing subgraphs...")
sg_analysis = explorer.analyze_subgraphs()

# Analyze features (FIXED)
print("\n[4/6] Analyzing features...")
feat_analysis = explorer.analyze_features_FIXED(sample_size=10000)

# Estimate computational requirements
print("\n[5/6] Estimating computational requirements...")
comp_analysis = explorer.estimate_computational_requirements()

    # Save report
print("\n[6/6] Saving report...")
report_file = explorer.save_report()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)

# Summary
print("\nSummary:")
print(f"  Nodes (estimated): ~{bg_analysis.get('n_nodes', 0) * 100:,}")
print(f"  Edges (estimated): ~{bg_analysis.get('n_edges', 0) * 100:,}")
print(f"  Subgraphs: {sg_analysis.get('n_subgraphs', 0):,}")
print(f"  Node features: {feat_analysis.get('node_features', {}).get('count', 0)}")
print(f"  Edge features: {feat_analysis.get('edge_features', {}).get('count', 0)}")
print(f"  Total features: {feat_analysis.get('total_features', 0)}")
print(f"  Memory (Polars): ~{comp_analysis.get('memory_gb', {}).get('total_polars', 0):.1f} GB")
print(f"\n  Report: {report_file}")
