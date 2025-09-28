import pandas as pd
import json
import logging
import networkx as nx
import itertools
import ast
from community import community_louvain
from pathlib import Path
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """Configuration for Phase 4: Parallel Trends Validation."""
    
    # --- FILE PATHS ---
    CLASSIFICATION_FILE = 'data/cleaned/cs_topics_info.csv'
    PAPERS_TOPICS_FILE = 'results/topics/document_topics_20250914_174642.csv'
    AUTHORS_FILE = 'data/cleaned/author_topic_networks_disambiguated_v4.csv'
    NETWORK_METRICS_FILE = 'results/collaboration_analysis/topic_analysis_10metrics_fixed_20250914_201257.json'
    OUTPUT_DIR = Path('results/shock_analysis/parallel_trends/')

    # --- COLUMN NAMES ---
    PAPERS_ID_COL = 'id'
    PAPERS_DATE_COL = 'published_date'
    AUTHORS_ID_COL = 'id'
    AUTHORS_TOPIC_COL = 'topic'
    AUTHORS_AUTHORS_COL = 'authors_parsed'
    CLASSIFICATIONS_TOPIC_COL = 'Topic'

    # --- ANALYSIS PARAMETERS ---
    INTERVENTION_DATE = pd.to_datetime('2022-11-30')
    KEY_METRICS_TO_TEST = ['modularity', 'coreness_ratio']

def load_precomputed_metrics(config):
    """Load pre-computed network metrics from JSON file."""
    logging.info("Loading pre-computed network metrics...")
    
    with open(config.NETWORK_METRICS_FILE, 'r') as f:
        metrics_data = json.load(f)
    
    # Convert to DataFrame
    metrics_list = []
    for key, topic_metrics in metrics_data.items():
        metrics_list.append(topic_metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    logging.info(f"Loaded metrics for {len(metrics_df)} topics")
    
    return metrics_df

def parse_author_string(author_str):
    """Parse author string from the authors_parsed column."""
    if not isinstance(author_str, str): 
        return []
    try:
        author_list = ast.literal_eval(author_str)
        full_names = [" ".join(part for part in name_parts if part).strip() for name_parts in author_list]
        return [name for name in full_names if name]
    except (ValueError, SyntaxError): 
        return []

def calculate_modularity(G):
    """Calculate modularity using community detection - CORRECTED VERSION."""
    if len(G) < 3 or G.number_of_edges() == 0:
        return 0.0
        
    try:
        # Try to use community detection
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            return community_louvain.modularity(partition, G)
        except ImportError:
            # Fallback to NetworkX community detection
            communities = list(nx.community.greedy_modularity_communities(G))
            return nx.community.modularity(G, communities)
    except:
        return 0.0

def calculate_coreness_ratio(G):
    """Calculate coreness as ratio of max k-core nodes to total nodes - CORRECTED VERSION."""
    if len(G) < 3:
        return 0.0
    
    try:
        core_numbers = nx.core_number(G)
        max_core = max(core_numbers.values()) if core_numbers else 0
        core_nodes = [n for n, k in core_numbers.items() if k == max_core]
        return len(core_nodes) / len(G)
    except:
        return 0.0

def calculate_network_metrics(G):
    """Calculate network metrics for a given graph - CORRECTED VERSION."""
    if G.number_of_nodes() == 0: 
        return None
    
    # Use largest connected component if not fully connected
    if not nx.is_connected(G):
        largest_cc_nodes = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc_nodes)
    
    # Calculate modularity using corrected function
    modularity_score = calculate_modularity(G)
    
    # Calculate coreness ratio using corrected function
    coreness_ratio = calculate_coreness_ratio(G)
    
    # Calculate density
    density = nx.density(G)
    
    return {
        'modularity': modularity_score,
        'coreness_ratio': coreness_ratio,
        'size': G.number_of_nodes(),
        'density': density
    }

def build_topic_period_panel(config):
    """Build topic-period panel with pre/post network metrics for DiD analysis."""
    logging.info("Building topic-period panel for DiD analysis...")
    
    # Load data
    classifications = pd.read_csv(config.CLASSIFICATION_FILE, low_memory=False)
    papers_dates = pd.read_csv(config.PAPERS_TOPICS_FILE, usecols=[config.PAPERS_ID_COL, config.PAPERS_DATE_COL], low_memory=False)
    authors_data = pd.read_csv(config.AUTHORS_FILE, usecols=[config.AUTHORS_ID_COL, config.AUTHORS_AUTHORS_COL, config.AUTHORS_TOPIC_COL], low_memory=False)
    
    # Rename columns for consistency
    papers_dates.rename(columns={config.PAPERS_ID_COL: 'paper_id', config.PAPERS_DATE_COL: 'date'}, inplace=True)
    authors_data.rename(columns={config.AUTHORS_ID_COL: 'paper_id', config.AUTHORS_AUTHORS_COL: 'authors_str', config.AUTHORS_TOPIC_COL: 'topic_id'}, inplace=True)
    classifications.rename(columns={config.CLASSIFICATIONS_TOPIC_COL: 'topic_id'}, inplace=True)
    
    # Merge data
    master_df = pd.merge(authors_data, papers_dates, on='paper_id', how='inner')
    master_df = pd.merge(master_df, classifications[['topic_id', 'Classification']], on='topic_id', how='inner')
    
    # Filter to LLM_RELATED and CONTROL groups
    master_df = master_df[master_df['Classification'].isin(['LLM_RELATED', 'CONTROL'])]
    master_df['date'] = pd.to_datetime(master_df['date'])
    master_df['authors'] = master_df['authors_str'].apply(parse_author_string)
    
    # Create period indicator
    master_df['period'] = master_df['date'].apply(
        lambda x: 'pre' if x < config.INTERVENTION_DATE else 'post'
    )
    
    all_metrics = []
    viable_topics = classifications['topic_id'].unique()
    
    logging.info(f"Processing {len(viable_topics)} topics...")
    
    for i, topic_id in enumerate(viable_topics):
        if i % 500 == 0:
            logging.info(f"  Processed {i}/{len(viable_topics)} topics...")
            
        topic_papers = master_df[master_df['topic_id'] == topic_id]
        
        # Process both pre and post periods
        for period in ['pre', 'post']:
            period_papers = topic_papers[topic_papers['period'] == period]
            
            if len(period_papers) == 0:
                continue
                
            # Build collaboration network for this topic-period
            G = nx.Graph()
            for author_list in period_papers['authors']:
                if len(author_list) >= 2:
                    for author1, author2 in itertools.combinations(author_list, 2):
                        if G.has_edge(author1, author2):
                            G[author1][author2]['weight'] += 1
                        else:
                            G.add_edge(author1, author2, weight=1)
            
            # Calculate metrics if network is large enough - REMOVED SIZE THRESHOLD
            # This allows consistency with cross-sectional analysis
            if G.number_of_nodes() >= 3:  # Minimum for meaningful metrics
                metrics = calculate_network_metrics(G)
                if metrics:
                    metrics['topic_id'] = topic_id
                    metrics['period'] = period
                    metrics['Classification'] = topic_papers['Classification'].iloc[0]
                    metrics['papers_in_period'] = len(period_papers)
                    all_metrics.append(metrics)
    
    # Convert to DataFrame
    panel_df = pd.DataFrame(all_metrics)
    
    if panel_df.empty:
        logging.error("No valid topic-period observations created!")
        return None
    
    # Add group variable for compatibility
    panel_df['group'] = panel_df['Classification'].map({'LLM_RELATED': 'treat', 'CONTROL': 'control'})
    
    logging.info(f"Built panel with {len(panel_df)} topic-period observations")
    logging.info(f"Covering {panel_df['topic_id'].nunique()} unique topics")
    
    # Show period distribution
    period_counts = panel_df.groupby(['period', 'group']).size().unstack(fill_value=0)
    logging.info(f"Period distribution:\n{period_counts}")
    
    # DIAGNOSTIC: Show baseline metric values
    logging.info("\n=== BASELINE METRIC DIAGNOSTICS ===")
    pre_period = panel_df[panel_df['period'] == 'pre']
    for metric in ['modularity', 'coreness_ratio']:
        if metric in pre_period.columns:
            mean_val = pre_period[metric].mean()
            std_val = pre_period[metric].std()
            min_val = pre_period[metric].min()
            max_val = pre_period[metric].max()
            logging.info(f"{metric}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
    
    return panel_df

def save_topic_period_panel(config):
    """Build and save the topic-period panel dataset."""
    panel_df = build_topic_period_panel(config)
    
    if panel_df is not None:
        # Save to the location expected by run_did_analysis.py
        output_path = Path('results/shock_analysis/topic_period_metrics.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        panel_df.to_csv(output_path, index=False)
        logging.info(f"Saved topic-period panel to {output_path}")
        
        # Also save a backup in data/cleaned/
        backup_path = Path('data/cleaned/topic_period_metrics.csv') 
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        panel_df.to_csv(backup_path, index=False)
        logging.info(f"Backup saved to {backup_path}")
        
        return panel_df
    else:
        logging.error("Failed to create topic-period panel!")
        return None

def build_quarterly_panel_with_precomputed(config):
    """Build quarterly panel using pre-computed network metrics and quarterly paper counts."""
    logging.info("Building quarterly panel with pre-computed metrics...")
    
    # Load data
    classifications = pd.read_csv(config.CLASSIFICATION_FILE, low_memory=False)
    papers_dates = pd.read_csv(config.PAPERS_TOPICS_FILE, usecols=[config.PAPERS_ID_COL, config.PAPERS_DATE_COL], low_memory=False)
    authors_data = pd.read_csv(config.AUTHORS_FILE, usecols=[config.AUTHORS_ID_COL, config.AUTHORS_TOPIC_COL], low_memory=False)
    
    # Load pre-computed metrics
    metrics_df = load_precomputed_metrics(config)
    
    # Rename columns for consistency
    papers_dates.rename(columns={config.PAPERS_ID_COL: 'paper_id', config.PAPERS_DATE_COL: 'date'}, inplace=True)
    authors_data.rename(columns={config.AUTHORS_ID_COL: 'paper_id', config.AUTHORS_TOPIC_COL: 'topic_id'}, inplace=True)
    classifications.rename(columns={config.CLASSIFICATIONS_TOPIC_COL: 'topic_id'}, inplace=True)
    
    # Merge data
    master_df = pd.merge(authors_data, papers_dates, on='paper_id', how='inner')
    master_df = pd.merge(master_df, classifications[['topic_id', 'Classification']], on='topic_id', how='inner')
    
    # Filter to LLM_RELATED and CONTROL groups
    master_df = master_df[master_df['Classification'].isin(['LLM_RELATED', 'CONTROL'])]
    master_df['date'] = pd.to_datetime(master_df['date'])
    master_df['quarter'] = master_df['date'].dt.to_period('Q')
    
    # Create quarterly paper counts per topic
    quarterly_counts = master_df.groupby(['quarter', 'topic_id', 'Classification']).size().reset_index(name='papers_in_quarter')
    
    # Merge with pre-computed metrics
    panel_df = pd.merge(quarterly_counts, metrics_df[['topic_id'] + config.KEY_METRICS_TO_TEST], on='topic_id', how='inner')
    
    # Filter topics that have metrics (these met the minimum network size requirement)
    valid_topics = set(metrics_df['topic_id'])
    panel_df = panel_df[panel_df['topic_id'].isin(valid_topics)]
    
    # Add time variables
    min_quarter = panel_df['quarter'].min()
    panel_df['quarter_num'] = (panel_df['quarter'].dt.year - min_quarter.year) * 4 + (panel_df['quarter'].dt.quarter - min_quarter.quarter)
    
    # Create binary treatment variable
    panel_df['treat'] = (panel_df['Classification'] == 'LLM_RELATED').astype(int)
    panel_df['group'] = panel_df['Classification'].map({'LLM_RELATED': 'treat', 'CONTROL': 'control'})
    
    logging.info(f"Built panel with {len(panel_df)} topic-quarter observations")
    logging.info(f"Covering {panel_df['topic_id'].nunique()} unique topics")
    logging.info(f"Time range: {panel_df['quarter'].min()} to {panel_df['quarter'].max()}")
    
    return panel_df

def plot_publication_trends(config):
    """Plot average quarterly publications per topic by group over time."""
    logging.info("--- Plotting publication trends by group ---")
    
    # Load the same data as the main analysis
    classifications = pd.read_csv(config.CLASSIFICATION_FILE, low_memory=False)
    papers_dates = pd.read_csv(config.PAPERS_TOPICS_FILE, usecols=[config.PAPERS_ID_COL, config.PAPERS_DATE_COL], low_memory=False)
    authors_data = pd.read_csv(config.AUTHORS_FILE, usecols=[config.AUTHORS_ID_COL, config.AUTHORS_TOPIC_COL], low_memory=False)
    
    # Load metrics to get the valid topic set
    metrics_df = load_precomputed_metrics(config)
    valid_topics = set(metrics_df['topic_id'])

    # Rename columns for consistency
    papers_dates.rename(columns={config.PAPERS_ID_COL: 'paper_id', config.PAPERS_DATE_COL: 'date'}, inplace=True)
    authors_data.rename(columns={config.AUTHORS_ID_COL: 'paper_id', config.AUTHORS_TOPIC_COL: 'topic_id'}, inplace=True)
    classifications.rename(columns={config.CLASSIFICATIONS_TOPIC_COL: 'topic_id'}, inplace=True)
    
    # Merge data
    master_df = pd.merge(authors_data, papers_dates, on='paper_id', how='inner')
    master_df = pd.merge(master_df, classifications[['topic_id', 'Classification']], on='topic_id', how='inner')
    
    # Filter to LLM_RELATED and CONTROL groups AND topics with valid metrics
    master_df = master_df[master_df['Classification'].isin(['LLM_RELATED', 'CONTROL'])]
    master_df = master_df[master_df['topic_id'].isin(valid_topics)]  # Key addition: only topics with network metrics
    master_df['date'] = pd.to_datetime(master_df['date'])
    
    # Create quarterly aggregations
    master_df['year_quarter'] = master_df['date'].dt.to_period('Q')
    
    # Get topic counts for context
    topic_counts = classifications[
        (classifications['Classification'].isin(['LLM_RELATED', 'CONTROL'])) & 
        (classifications['topic_id'].isin(valid_topics))
    ]['Classification'].value_counts()
    logging.info(f"Topic counts (with network metrics): {dict(topic_counts)}")
    
    # Count papers by quarter, topic, and group
    quarterly_topic_counts = master_df.groupby(['year_quarter', 'topic_id', 'Classification']).size().reset_index(name='paper_count')
    quarterly_averages = quarterly_topic_counts.groupby(['year_quarter', 'Classification'])['paper_count'].mean().reset_index()
    quarterly_averages.rename(columns={'paper_count': 'avg_papers_per_topic'}, inplace=True)
    quarterly_averages['year_quarter_ts'] = quarterly_averages['year_quarter'].dt.to_timestamp()
    quarterly_averages['group'] = quarterly_averages['Classification'].map({'LLM_RELATED': 'treat', 'CONTROL': 'control'})
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    sns.lineplot(data=quarterly_averages, x='year_quarter_ts', y='avg_papers_per_topic', hue='group', 
                marker='o', markersize=5, ax=ax, palette={'treat': 'red', 'control': 'blue'}, linewidth=2.5)
    
    ax.axvline(x=config.INTERVENTION_DATE, color='black', linestyle='--', alpha=0.7, linewidth=2)
    ax.text(config.INTERVENTION_DATE, ax.get_ylim()[1] * 0.9, 'ChatGPT Release\n(Nov 30, 2022)', 
            horizontalalignment='center', verticalalignment='top', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax.set_title('Quarterly Average Publications per Topic by Group\n(Topics with Valid Collaboration Networks)', fontsize=14, weight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Average Papers per Topic per Quarter', fontsize=11)
    
    control_count = topic_counts.get('CONTROL', 0)
    treat_count = topic_counts.get('LLM_RELATED', 0)
    ax.legend(title='Group', labels=[f'Control ({control_count} topics)', f'Treatment ({treat_count} topics)'])
    
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    
    # Save plot
    plot_path = config.OUTPUT_DIR / 'publication_trends_per_topic_with_networks.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved publication trends plot to {plot_path}")
    
    return quarterly_averages

def test_and_plot_trends(panel_df, metric, config):
    """Generate plot and run statistical test for parallel trends."""
    logging.info(f"--- Validating parallel trends for: {metric.upper()} ---")

    # Filter to pre-treatment period for parallel trends test
    pre_treatment_df = panel_df[panel_df['quarter'].dt.to_timestamp() < config.INTERVENTION_DATE].copy()
    
    if len(pre_treatment_df) == 0:
        logging.warning(f"No pre-treatment data for {metric}")
        return None

    # 1. Visualization
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plot_data = pre_treatment_df.copy()
    plot_data['quarter_ts'] = plot_data['quarter'].dt.to_timestamp()
    
    sns.lineplot(data=plot_data, x='quarter_ts', y=metric, hue='group', errorbar='ci', ax=ax, 
                palette={'treat': 'red', 'control': 'blue'})
    
    ax.set_title(f'Pre-Treatment Trend for: {metric.replace("_", " ").title()}', fontsize=16, weight='bold')
    ax.set_xlabel('Quarter', fontsize=12)
    ax.set_ylabel(f'Average {metric.replace("_", " ").title()}', fontsize=12)
    ax.legend(title='Group')
    plt.tight_layout()
    
    plot_path = config.OUTPUT_DIR / f'parallel_trends_{metric}.png'
    plt.savefig(plot_path)
    plt.close(fig)
    logging.info(f"Saved plot to {plot_path}")

    # 2. Statistical Test (pre-treatment data only)
    formula = f"{metric} ~ quarter_num * treat"
    try:
        model = smf.ols(formula, data=pre_treatment_df.dropna(subset=[metric])).fit(
            cov_type='cluster', cov_kwds={'groups': pre_treatment_df['topic_id']}
        )
        interaction_p_value = model.pvalues['quarter_num:treat']
        
        print(f"\nStatistical Test for {metric}:")
        print(model.summary().tables[1])
        print(f"\nP-value for interaction (quarter_num:treat): {interaction_p_value:.4f}")
        
        if interaction_p_value < 0.05:
            print("  -> WARNING: Parallel trends assumption may be violated (p < 0.05).")
        else:
            print("  -> SUCCESS: Parallel trends assumption holds (p >= 0.05).")
        
        return {
            'metric': metric,
            'interaction_coef': model.params['quarter_num:treat'],
            'p_value': interaction_p_value,
            'is_violated': interaction_p_value < 0.05
        }
    except Exception as e:
        logging.error(f"Could not fit model for {metric}. Error: {e}")
        return None

def main():
    config = Config()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build panel using pre-computed metrics
    quarterly_panel = build_quarterly_panel_with_precomputed(config)
    
    if quarterly_panel.empty:
        logging.error("Failed to build quarterly panel. No data to analyze.")
        return

    # Plot publication trends (using same topic subset as panel)
    quarterly_data = plot_publication_trends(config)
    
    # Test parallel trends for network metrics
    results = []
    for metric in config.KEY_METRICS_TO_TEST:
        if metric in quarterly_panel.columns and quarterly_panel[metric].notna().any():
            result = test_and_plot_trends(quarterly_panel, metric, config)
            if result:
                results.append(result)
        else:
            logging.warning(f"Metric '{metric}' not found or contains all NaNs. Skipping.")
    
    # Save results
    summary_df = pd.DataFrame(results)
    summary_path = config.OUTPUT_DIR / 'parallel_trends_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    quarterly_data_path = config.OUTPUT_DIR / 'quarterly_publication_trends_per_topic.csv'
    quarterly_data.to_csv(quarterly_data_path, index=False)

    logging.info("Building topic-period panel for DiD analysis...")
    save_topic_period_panel(config)
    
    logging.info(f"Phase 4 complete. Summary of tests saved to {summary_path}")
    logging.info(f"Quarterly publication trends saved to {quarterly_data_path}")

if __name__ == '__main__':
    main()