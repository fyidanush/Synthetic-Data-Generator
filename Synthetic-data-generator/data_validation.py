from scipy.stats import ks_2samp

def compare_distributions(real_data, synthetic_data):
    results = {}
    for col in real_data.columns:
        stat, p_value = ks_2samp(real_data[col], synthetic_data[col])
        results[col] = {"KS Statistic": stat, "P-value": p_value}
    return results