import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import logging
from pathlib import Path
from datetime import datetime

def calculate_effect_sizes(df, metric, treatment_coef):
    """Calculate effect sizes for DiD results."""
    
    # Calculate baseline statistics (control group, pre-period)
    baseline = df[(df['treat'] == 0) & (df['post'] == 0)][metric]
    baseline_mean = baseline.mean()
    baseline_std = baseline.std()
    
    # Calculate overall standard deviation for Cohen's d
    overall_std = df[metric].std()
    
    # Effect sizes
    cohens_d = treatment_coef / overall_std
    percent_change = (treatment_coef / baseline_mean) * 100
    std_units = treatment_coef / baseline_std
    
    return {
        'baseline_mean': baseline_mean,
        'baseline_std': baseline_std,
        'overall_std': overall_std,
        'cohens_d': cohens_d,
        'percent_change': percent_change,
        'std_units': std_units
    }

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size."""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def get_significance_stars(p_value):
    """Returns significance stars for a p-value."""
    if p_value < 0.01: return '***'
    if p_value < 0.05: return '**'
    if p_value < 0.1: return '*'
    return ''

def run_enhanced_did_analysis(config):
    """
    Enhanced DiD analysis with effect size calculations.
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading panel data from {config.PANEL_DATA_FILE}")
    try:
        df = pd.read_csv(config.PANEL_DATA_FILE)
    except FileNotFoundError:
        logging.error(f"FATAL: Input file not found at {config.PANEL_DATA_FILE}. Please run Phase 3 first.")
        return

    # Prepare data for regression
    df['treat'] = (df['group'] == 'treat').astype(int)
    df['post'] = (df['period'] == 'post').astype(int)
    df['log_size'] = np.log(df['size'] + 1)
    
    all_results = []
    effect_sizes_results = []

    for metric in config.OUTCOME_METRICS:
        logging.info(f"--- Running DiD analysis for: {metric.upper()} ---")
        
        df_metric = df.dropna(subset=[metric])

        # Model 1: Basic DiD
        formula1 = f"{metric} ~ treat * post"
        model1 = smf.ols(formula1, data=df_metric).fit(cov_type='cluster', cov_kwds={'groups': df_metric['topic_id']})
        
        # Model 2: DiD with Controls
        formula2 = f"{metric} ~ treat * post + log_size + density"
        model2 = smf.ols(formula2, data=df_metric).fit(cov_type='cluster', cov_kwds={'groups': df_metric['topic_id']})
        
        # Model 3: Weighted DiD
        model3 = smf.wls(formula1, data=df_metric, weights=df_metric['size']).fit(cov_type='cluster', cov_kwds={'groups': df_metric['topic_id']})

        # Calculate effect sizes for each model
        models_info = [
            (model1, "Basic"), 
            (model2, "With Controls"), 
            (model3, "Weighted")
        ]
        
        for model, name in models_info:
            did_coef = model.params.get('treat:post', np.nan)
            did_se = model.bse.get('treat:post', np.nan)
            did_pvalue = model.pvalues.get('treat:post', np.nan)
            
            # Calculate effect sizes
            if not np.isnan(did_coef):
                effect_sizes = calculate_effect_sizes(df_metric, metric, did_coef)
                
                effect_sizes_results.append({
                    'Outcome Variable': metric,
                    'Model': name,
                    'Treatment Coef': did_coef,
                    'Baseline Mean': effect_sizes['baseline_mean'],
                    'Cohen\'s d': effect_sizes['cohens_d'],
                    'Cohen\'s d Interpretation': interpret_cohens_d(effect_sizes['cohens_d']),
                    'Percent Change': effect_sizes['percent_change'],
                    'Std Units': effect_sizes['std_units']
                })
            
            all_results.append({
                'Outcome Variable': metric,
                'Model': name,
                'Interaction Coef (β3)': f"{did_coef:.4f}",
                'Std. Error': f"({did_se:.4f})",
                'P-Value': f"{did_pvalue:.4f}",
                'Significance': get_significance_stars(did_pvalue),
                'N': int(model.nobs)
            })

    # --- Format Main Results Table ---
    results_df = pd.DataFrame(all_results)
    
    display_df = results_df.copy()
    display_df['β3 (Std. Err.)'] = display_df.apply(
        lambda row: f"{row['Interaction Coef (β3)']}{row['Significance']}\n{row['Std. Error']}",
        axis=1
    )
    
    main_table = display_df.pivot_table(
        index='Model',
        columns='Outcome Variable',
        values='β3 (Std. Err.)',
        aggfunc='first'
    )
    
    model_order = ["Basic", "With Controls", "Weighted"]
    main_table = main_table.reindex(model_order)
    main_table = main_table[config.OUTCOME_METRICS]

    n_obs = display_df.pivot_table(index='Model', columns='Outcome Variable', values='N', aggfunc='first').iloc[0]
    main_table.loc['Observations'] = n_obs.astype(int)
    main_table.loc['Topic-Clustered SE'] = "Yes"
    
    # --- Create Effect Sizes Table ---
    effect_sizes_df = pd.DataFrame(effect_sizes_results)
    
    # Format effect sizes table
    es_display = effect_sizes_df.copy()
    es_display['Effect Size Summary'] = es_display.apply(
        lambda row: f"d={row['Cohen\'s d']:.3f} ({row['Cohen\'s d Interpretation']})\n{row['Percent Change']:.1f}% change",
        axis=1
    )
    
    effect_sizes_table = es_display.pivot_table(
        index='Model',
        columns='Outcome Variable',
        values='Effect Size Summary',
        aggfunc='first'
    )
    effect_sizes_table = effect_sizes_table.reindex(model_order)
    effect_sizes_table = effect_sizes_table[config.OUTCOME_METRICS]
    
    # Add baseline means row
    baselines = effect_sizes_df.drop_duplicates('Outcome Variable')[['Outcome Variable', 'Baseline Mean']].set_index('Outcome Variable')['Baseline Mean']
    baseline_row = {metric: f"Baseline: {baselines[metric]:.3f}" for metric in config.OUTCOME_METRICS}
    effect_sizes_table.loc['Baseline Mean'] = baseline_row

    # --- Display Results ---
    logging.info("--- Main DiD Results ---")
    print(main_table.to_string())
    print("\n" + "="*80)
    print("--- Effect Sizes ---")
    print(effect_sizes_table.to_string())

    # --- Save Results ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save main results
    csv_path = config.OUTPUT_DIR / f'did_main_results_{timestamp}.csv'
    txt_path = config.OUTPUT_DIR / f'did_main_results_{timestamp}.txt'
    
    main_table.to_csv(csv_path)
    
    # Save effect sizes
    es_csv_path = config.OUTPUT_DIR / f'did_effect_sizes_{timestamp}.csv'
    effect_sizes_df.to_csv(es_csv_path, index=False)
    
    # Combined text output
    with open(txt_path, 'w') as f:
        f.write("=========================================================\n")
        f.write("Phase 5: Main Difference-in-Differences Analysis Results\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=========================================================\n\n")
        f.write("MAIN RESULTS:\n")
        f.write(main_table.to_string())
        f.write("\n\n")
        f.write("EFFECT SIZES:\n")
        f.write(effect_sizes_table.to_string())
        f.write("\n\n")
        f.write("DETAILED EFFECT SIZE CALCULATIONS:\n")
        f.write(effect_sizes_df.to_string(index=False))
        
    logging.info(f"Phase 5 complete. Results saved to {csv_path} and {txt_path}")
    logging.info(f"Effect sizes saved to {es_csv_path}")
    
    return main_table, effect_sizes_df

# Update the Config class to work with the enhanced function
class Config:
    """Configuration for Phase 5: Main DiD Analysis."""
    
    PANEL_DATA_FILE = 'results/shock_analysis/topic_period_metrics.csv'
    OUTPUT_DIR = Path('results/shock_analysis/main_did/')
    OUTCOME_METRICS = ['modularity', 'coreness_ratio']
    
    HYPOTHESES = {
        'modularity': 'β3 > 0 (more modular)',
        'coreness_ratio': 'β3 < 0 (less hierarchical)'
    }

if __name__ == '__main__':
    config = Config()
    run_enhanced_did_analysis(config)