# Persistent Homology Analysis of arXiv Collaboration Networks

## Overview

This project applies **topological data analysis (TDA)**, specifically persistent homology, to study collaboration networks across different mathematical and computer science fields on arXiv. The goal is to identify topological differences in collaboration patterns between fields, building on recent insights about higher-order structures in scientific collaboration networks.

While traditional network analysis focuses on pairwise relationships and standard graph metrics, persistent homology can detect higher-dimensional topological features—such as loops, voids, and other complex structures—that may reveal deeper organizational patterns in how researchers collaborate within different scientific domains.

## Research Questions

- **Do different mathematical and computer science fields exhibit distinct topological signatures in their collaboration patterns?**
- **How do higher-order topological features (beyond pairwise connections) differ between theoretical and applied domains?**
- **Can persistent homology features predict field membership or collaboration success better than traditional network metrics?**

## Motivation

Recent work by Yang & Wang (2024) demonstrated that higher-order structures in local collaboration networks are associated with individual scientific productivity. This suggests that the topological organization of collaboration networks contains meaningful information about research dynamics that goes beyond what traditional network analysis can capture.

Mathematical research is organized into numerous subfields with distinct collaborative norms, making it an ideal testbed for investigating whether topological differences in collaboration patterns are genuine structural phenomena or simply artifacts of network size and growth.

## Methodology Overview

### 1. **Data Foundation**
- **Source**: arXiv papers and author collaboration networks
- **Scope**: Mathematics and computer science publications
- **Time Range**: [To be specified based on data availability]
- **Size**: ~121,000 papers with robust author disambiguation

### 2. **Network Construction**
- Build collaboration networks by mathematical subfield (arXiv categories)
- Apply conservative author disambiguation pipeline for high precision
- Generate category-specific networks with minimum size thresholds for robust analysis

### 3. **Topological Analysis**
- **Distance Matrix Construction**: Convert collaboration networks to metric spaces
- **Filtration**: Build filtered simplicial complexes using various distance thresholds
- **Persistence Computation**: Calculate persistent homology in dimensions 0, 1, and potentially 2
- **Feature Extraction**: Derive topological features from persistence diagrams

### 4. **Comparative Analysis**
- Statistical comparison of persistence features across fields
- Integration with traditional network metrics for validation
- Machine learning classification using topological features

## Technical Approach

### Core TDA Pipeline

```python
# Simplified workflow
networks = build_collaboration_networks_by_category(papers, authors)
for category, network in networks.items():
    distance_matrix = build_distance_matrix(network)
    persistence = compute_persistent_homology(distance_matrix)
    features = extract_persistence_features(persistence)
    results[category] = features
```

### Key Dependencies
- **TDA**: `gudhi`, `scikit-tda`, `persim`, `ripser`
- **Network Analysis**: `networkx`
- **Data Processing**: `pandas`, `numpy`, `scipy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`, `umap-learn`

### Scalability Considerations
- Use sparse Rips or witness complexes for large networks (>5000 nodes)
- Implement batch processing for memory constraints
- Parallel computation across categories using `joblib`

## Project Structure

```
tda-collaboration-networks/
├── data/
│   ├── raw/                    # Source data from arXiv
│   ├── processed/             # Cleaned and categorized networks
│   └── persistence/           # Computed persistence diagrams and features
├── src/
│   ├── data_preprocessing.py   # Data cleaning and network construction
│   ├── author_disambiguation.py # Author name disambiguation pipeline
│   ├── tda_analysis.py        # Persistent homology computation
│   ├── feature_extraction.py  # Topological feature extraction
│   ├── visualization.py       # Plotting and visualization tools
│   └── statistical_analysis.py # Comparative statistical tests
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_persistence_computation.ipynb
│   ├── 03_topological_comparison.ipynb
│   └── 04_results_interpretation.ipynb
├── results/
│   ├── figures/               # Generated plots and visualizations
│   ├── tables/                # Statistical results and comparisons
│   └── persistence_diagrams/ # Saved persistence diagrams
├── config/
│   └── analysis_config.yaml   # Analysis parameters and settings
└── docs/
    ├── methodology.md         # Detailed methodological documentation
    └── results_summary.md     # Key findings and interpretations
```

## Getting Started

### Prerequisites
```bash
# Install core TDA libraries
pip install gudhi scikit-tda persim ripser

# Install supporting libraries
pip install networkx pandas numpy scipy matplotlib seaborn plotly
pip install scikit-learn umap-learn joblib
```

### Quick Start
1. **Clone and setup**:
   ```bash
   git clone https://github.com/brian-hepler-phd/tda-collaboration-networks.git
   cd tda-collaboration-networks
   pip install -r requirements.txt
   ```

2. **Run initial analysis**:
   ```bash
   python src/data_preprocessing.py --config config/analysis_config.yaml
   python src/tda_analysis.py --categories math.AG,math.AP,cs.LG
   ```

3. **Explore results**:
   ```bash
   jupyter notebook notebooks/01_exploratory_analysis.ipynb
   ```

## Expected Outcomes

### Immediate Deliverables
- **Persistence diagrams** for major arXiv categories
- **Topological feature comparisons** between mathematical subfields
- **Classification accuracy** using persistence features vs. traditional metrics

### Research Contributions
- First comprehensive topological analysis of collaboration patterns across mathematical disciplines
- Methodological framework for applying TDA to scientific collaboration networks
- Empirical evidence for (or against) topological differences in disciplinary collaboration patterns

### Potential Extensions
- **Temporal Analysis**: Evolution of topological features over time
- **Productivity Prediction**: Using topological features to predict research outcomes
- **Cross-Disciplinary Analysis**: Extending beyond mathematics to other scientific domains

## Attribution and Prior Work

This project builds upon the network analysis framework developed in [MRC-Network-Analysis](https://github.com/brian-hepler-phd/MRC-Network-Analysis). The core author disambiguation pipeline, network construction methods, and data processing infrastructure were originally developed for:

> Hepler, B. (2024). *Multi-scale Analysis of Research Collaboration Networks in Mathematics*. [Publication details]

The methodological foundation combines established techniques from:
- **Network Science**: Author disambiguation and collaboration network construction
- **Topological Data Analysis**: Persistent homology and topological feature extraction  
- **Scientometrics**: Multi-scale analysis of research collaboration patterns

## References

1. **Yang, W., & Wang, Y. (2024)**. "Higher-order structures of local collaboration networks are associated with individual scientific productivity." *EPJ Data Science*, 13(1), 1-22.

2. **Carlsson, G. (2009)**. "Topology and data." *Bulletin of the American Mathematical Society*, 46(2), 255-308.

3. **Patania, A., et al. (2017)**. "The shape of collaborations." *EPJ Data Science*, 6(1), 1-16.

4. **Edelsbrunner, H., & Harer, J. (2010)**. *Computational topology: an introduction*. American Mathematical Society.

---

## Contact

**Brian Hepler**  
hepler.brian@gmail.com  
ORCID:  0000-0002-8037-930X