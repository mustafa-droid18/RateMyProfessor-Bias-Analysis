# Data Manipulation and Statistics
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ks_1samp, norm, levene, ttest_ind, zscore

# Machine Learning and Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, classification_report,  roc_curve, auc
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from imblearn.over_sampling import SMOTE

# Visualization
import matplotlib.pyplot as plt

# Set parameters
np.random.seed(15456214)

threshold = 5  # Minimum number of ratings for reliable analysis
damping_factor = 10  # Conservative choice to stabilize adjustments
alpha=0.005

def Basic_Processing():
    # File paths
    num_file = "rmpCapstoneNum.csv"
    qual_file = "rmpCapstoneQual.csv"
    tags_file = "rmpCapstoneTags.csv"

    # Headers for each file
    num_headers = [
        "Average Rating", "Average Difficulty", "Number of Ratings",
        "Received a Pepper?", "Proportion Retake", "Online Ratings",
        "Male", "Female"
    ]
    qual_headers = ["Major/Field", "University", "US State"]
    tags_headers = [
        "Tough Grader", "Good Feedback", "Respected", "Lots to Read", "Participation Matters",
        "Don't Skip Class", "Lots of Homework", "Inspirational", "Pop Quizzes!", "Accessible",
        "So Many Papers", "Clear Grading", "Hilarious", "Test Heavy", "Graded by Few Things",
        "Amazing Lectures", "Caring", "Extra Credit", "Group Projects", "Lecture Heavy"
    ]

    # Adding headers and saving updated files
    # rmpCapstoneNum.csv
    num_data = pd.read_csv(num_file, header=None)
    num_data.columns = num_headers
    num_data.to_csv(num_file, index=False)

    # rmpCapstoneQual.csv
    qual_data = pd.read_csv(qual_file, header=None)
    qual_data.columns = qual_headers
    qual_data.to_csv(qual_file, index=False)

    # rmpCapstoneTags.csv
    tags_data = pd.read_csv(tags_file, header=None)
    tags_data.columns = tags_headers
    tags_data.to_csv(tags_file, index=False)

    print("Headers added to all files successfully.")


def Preprocessing_Q1_3(damping_factor=10, threshold=5):
    # Load the dataset
    num_data_path = 'rmpCapstoneNum.csv'
    num_data = pd.read_csv(num_data_path)

    # Rename columns for better readability
    num_data.rename(columns={
        'Male': 'Male',
        'Female': 'Female',
        'Average Rating': 'Average_Rating'
    }, inplace=True)

    # Filter rows with unambiguous gender and sufficient ratings
    filtered_data = num_data[
        ((num_data['Male'] == 1) & (num_data['Female'] == 0)) |
        ((num_data['Male'] == 0) & (num_data['Female'] == 1))
    ]
    filtered_data = filtered_data[filtered_data['Number of Ratings'] >= threshold]

    # Calculate the global mean rating across all professors
    global_mean_rating = filtered_data['Average_Rating'].mean()

    # Compute Bayesian Adjusted Average Ratings
    filtered_data['Adjusted_Average'] = filtered_data.apply(
        lambda row: (
            (row['Average_Rating'] * row['Number of Ratings'] +
            global_mean_rating * damping_factor) /
            (row['Number of Ratings'] + damping_factor)
        ), axis=1
    )

    return filtered_data

def Preprocessing_Q5_6(damping_factor=10, threshold=5):
    #Load the numerical dataset
    num_df = pd.read_csv("rmpCapstoneNum.csv")  # Adjust file path as needed

    # Filter for unambiguous gender and at least 5 ratings
    filtered_num_df = num_df[
        ((num_df['Male'] == 1) & (num_df['Female'] == 0)) |
        ((num_df['Male'] == 0) & (num_df['Female'] == 1))
    ]
    filtered_num_df = filtered_num_df[filtered_num_df['Number of Ratings'] >= 5]

    # Calculate global mean for Bayesian adjustment
    global_mean_difficulty = filtered_num_df['Average Difficulty'].mean()
    damping_factor = 10  # Adjust damping factor if needed for sensitivity

    # Apply Bayesian adjustment to difficulty ratings
    filtered_num_df['Adjusted Difficulty'] = filtered_num_df.apply(
        lambda row: (
            (row['Average Difficulty'] * row['Number of Ratings'] + global_mean_difficulty * damping_factor) /
            (row['Number of Ratings'] + damping_factor)
        ), axis=1
    )
    
    return filtered_num_df

def Preprocessing_Q10_11():

    numerical_file = "rmpCapstoneNum.csv"
    qualitative_file = "rmpCapstoneQual.csv"
    tags_file = "rmpCapstoneTags.csv"

    # Headers
    num_headers = [
        "Average Rating", "Average Difficulty", "Number of Ratings",
        "Received a Pepper?", "Proportion Retake", "Online Ratings",
        "Male", "Female"
    ]
    qual_headers = ["Major/Field", "University", "US State"]
    tags_headers = [
        "Tough Grader", "Good Feedback", "Respected", "Lots to Read", "Participation Matters",
        "Don't Skip Class", "Lots of Homework", "Inspirational", "Pop Quizzes!", "Accessible",
        "So Many Papers", "Clear Grading", "Hilarious", "Test Heavy", "Graded by Few Things",
        "Amazing Lectures", "Caring", "Extra Credit", "Group Projects", "Lecture Heavy"
    ]

    # Load datasets
    # Treat blank cells as missing
    numerical_data = pd.read_csv(numerical_file, na_values=["", " "])
    qualitative_data = pd.read_csv(qualitative_file, na_values=["", " "])
    tags_data = pd.read_csv(tags_file, na_values=["", " "])

    # Missing values
    print("Missing values in qualitative data (before handling):")
    print(qualitative_data.isnull().sum())

    # Dropping missing values in qualitative data
    qualitative_data.dropna(inplace=True)

    # Combining
    comb_df = pd.concat([numerical_data, qualitative_data, tags_data], axis=1)

    ### Handling Missing Data ###

    # Numerical Columns: Median for counts, Mean for ratings/difficulty
    numerical_cols = num_headers[:6] + tags_headers
    for col in numerical_cols:
        if comb_df[col].isnull().sum() > 0:
            if "Number of Ratings" in col or "Online Ratings" in col:
                comb_df[col] = comb_df[col].fillna(comb_df[col].median())
            else:
                comb_df[col] = comb_df[col].fillna(comb_df[col].mean())

    # Gender/Binary Columns: Add an "NA" Category
    gender_cols = ["Male", "Female", "Received a Pepper?"]
    for col in gender_cols:
        comb_df[col] = comb_df[col].astype("category")  # Convert to categorical
        comb_df[col] = comb_df[col].cat.add_categories("NA")  # Add NA category
        comb_df[col] = comb_df[col].fillna("NA")

    # Typecasting - error resolution
    comb_df = comb_df.astype({
        "Average Rating": float,
        "Average Difficulty": float,
        "Number of Ratings": int,
        "Proportion Retake": float,
        "Online Ratings": int
    })

    # Checking missing vals
    print("Missing values after handling:")
    print(comb_df.isnull().sum())

    # Save preprocessed data
    comb_df.to_csv("preprocessed_combined_data.csv", index=False)

    return None

def Question_1(alpha=0.005, damping_factor=10, threshold=5):
    """
    Evaluates gender bias in professor ratings using statistical and visualization techniques.
    
    This function performs the following tasks:
    1. Filters and preprocesses data to include only professors with unambiguous gender and a 
       minimum number of ratings (controlled by the `threshold` parameter).
    2. Computes Bayesian adjusted average ratings using the specified `damping_factor`.
    3. Conducts statistical tests (Kolmogorov-Smirnov and Mann-Whitney U) to compare ratings 
       between male and female professors.
    4. Visualizes the distribution of ratings and differences using histograms and boxplots.
    5. Analyzes subgroups of professors based on the presence or absence of a "pepper" rating.
    6. Analyzes subset of professors who are considered as Tough Graders and Bad Feedback

    Parameters:
        alpha (float): Significance level for hypothesis testing (default: 0.005).
        damping_factor (int): Damping factor for Bayesian adjustment (default: 10).
        threshold (int): Minimum number of ratings required to include a professor in the analysis (default: 5).
    """
    filtered_data=Preprocessing_Q1_3()

    # --Question 1 Code and Answer-- 

    # Separate data by gender
    male_ratings = filtered_data[filtered_data['Male'] == 1]['Adjusted_Average']
    female_ratings = filtered_data[filtered_data['Female'] == 1]['Adjusted_Average']

    # Calculate means and medians
    mean_male = male_ratings.mean()
    mean_female = female_ratings.mean()
    median_male = np.median(male_ratings)
    median_female = np.median(female_ratings)

    # Plot histograms and boxplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram for adjusted average ratings by gender
    axes[0].hist(male_ratings, bins=20, alpha=0.5, label=f'Male Professors\nMedian: {median_male:}', density=False)
    axes[0].hist(female_ratings, bins=20, alpha=0.5, label=f'Female Professors\nMedian: {median_female:}', density=False)
    axes[0].axvline(median_male, color='blue', linestyle='-', label='Median (Male Professors)')
    axes[0].axvline(median_female, color='orange', linestyle='-', label='Median (Female Professors)')
    axes[0].set_xlabel('Adjusted Average Rating')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Adjusted Average Ratings by Gender')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Boxplot for adjusted average ratings by gender
    axes[1].boxplot(
        [male_ratings, female_ratings],
        labels=[
            f'Male Professors\nMean: {mean_male:}',
            f'Female Professors\nMean: {mean_female:}'
        ],
        patch_artist=True
    )
    axes[1].set_ylabel('Adjusted Average Rating')
    axes[1].set_title('Boxplot of Adjusted Average Ratings by Gender')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust spacing and show the plots
    plt.tight_layout()
    plt.show()

    # Standardize the data (z-score normalization)
    male_ratings_standardized = zscore(male_ratings)
    female_ratings_standardized = zscore(female_ratings)

    # Perform KS test for normality
    male_ks = ks_1samp(male_ratings_standardized, cdf=norm.cdf, alternative='two-sided')
    female_ks = ks_1samp(female_ratings_standardized, cdf=norm.cdf, alternative='two-sided')

    # Print normality test results
    print("\n### Normality Test Results (Kolmogorov-Smirnov) ###")
    print(f"Male Ratings: Test Statistic = {male_ks.statistic:}, P-value = {male_ks.pvalue:}")
    print(f"Female Ratings: Test Statistic = {female_ks.statistic:}, P-value = {female_ks.pvalue:}")
    if male_ks.pvalue < alpha:
        print("Male Ratings: Distribution is NOT normal.")
    else:
        print("Male Ratings: Distribution is normal.")
    if female_ks.pvalue < alpha:
        print("Female Ratings: Distribution is NOT normal.")
    else:
        print("Female Ratings: Distribution is normal.")

    # Perform the Mann-Whitney U Test
    mannwhitney_test = mannwhitneyu(male_ratings, female_ratings, alternative='greater')

    # Print Mann-Whitney U test results
    print("\n### Mann-Whitney U Test Results ###")
    print(f"U Statistic: {mannwhitney_test.statistic:}")
    print(f"P-value: {mannwhitney_test.pvalue:}")
    if mannwhitney_test.pvalue < alpha:
        print("Result: Significant. Evidence of pro-male bias in ratings.")
    else:
        print("Result: Not Significant. No evidence of pro-male bias in ratings.")

    # Filter data for professors who did not receive a pepper
    no_pepper_data = filtered_data[filtered_data['Received a Pepper?'] == 0]
    male_no_pepper = no_pepper_data[no_pepper_data['Male'] == 1]['Adjusted_Average']
    female_no_pepper = no_pepper_data[no_pepper_data['Female'] == 1]['Adjusted_Average']

    # Perform Mann-Whitney U Test for professors without a pepper
    mannwhitney_no_pepper = mannwhitneyu(male_no_pepper, female_no_pepper, alternative='greater')

    # Print results for professors without a pepper
    print("\n### Mann-Whitney U Test Results for Professors WITHOUT a Pepper ###")
    print(f"U Statistic: {mannwhitney_no_pepper.statistic:}")
    print(f"P-value: {mannwhitney_no_pepper.pvalue:}")
    if mannwhitney_no_pepper.pvalue < alpha:
        print("Result: Significant. Evidence of pro-male bias for professors without a pepper.")
    else:
        print("Result: Not Significant. No evidence of pro-male bias for professors without a pepper.")

    # Filter data for professors who received a pepper
    pepper_data = filtered_data[filtered_data['Received a Pepper?'] == 1]
    male_pepper = pepper_data[pepper_data['Male'] == 1]['Adjusted_Average']
    female_pepper = pepper_data[pepper_data['Female'] == 1]['Adjusted_Average']

    # Perform Mann-Whitney U Test for professors with a pepper
    mannwhitney_pepper = mannwhitneyu(male_pepper, female_pepper, alternative='greater')

    # Print results for professors with a pepper
    print("\n### Mann-Whitney U Test Results for Professors WITH a Pepper ###")
    print(f"U Statistic: {mannwhitney_pepper.statistic:}")
    print(f"P-value: {mannwhitney_pepper.pvalue:}")
    if mannwhitney_pepper.pvalue < alpha:
        print("Result: Significant. Evidence of pro-male bias for professors with a pepper.")
    else:
        print("Result: Not Significant. No evidence of pro-male bias for professors with a pepper.")

    # Print mean and median adjusted ratings for all categories
    print("\n### Adjusted Ratings Summary ###")
    print(f"Mean Male (No Pepper): {male_no_pepper.mean():}")
    print(f"Mean Female (No Pepper): {female_no_pepper.mean():}")
    print(f"Mean Male (With Pepper): {male_pepper.mean():}")
    print(f"Mean Female (With Pepper): {female_pepper.mean():}")
    print(f"Median Male (No Pepper): {male_no_pepper.median():}")
    print(f"Median Female (No Pepper): {female_no_pepper.median():}")
    print(f"Median Male (With Pepper): {male_pepper.median():}")
    print(f"Median Female (With Pepper): {female_pepper.median():}")

    # Calculate means and medians for boxplot and histogram
    means = [
        male_no_pepper.mean(), female_no_pepper.mean(),
        male_pepper.mean(), female_pepper.mean()
    ]
    medians = [
        male_no_pepper.median(), female_no_pepper.median(),
        male_pepper.median(), female_pepper.median()
    ]

    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Histogram for Adjusted Average Ratings by gender and pepper status
    axes[0].hist(male_no_pepper, bins=20, alpha=0.5, label=f'Male (No Pepper)\nMedian: {medians[0]:}', density=False)
    axes[0].hist(female_no_pepper, bins=20, alpha=0.5, label=f'Female (No Pepper)\nMedian: {medians[1]:}', density=False)
    axes[0].hist(male_pepper, bins=20, alpha=0.5, label=f'Male (With Pepper)\nMedian: {medians[2]:}', density=False)
    axes[0].hist(female_pepper, bins=20, alpha=0.5, label=f'Female (With Pepper)\nMedian: {medians[3]:}', density=False)

    # Add median lines to the histogram
    groups = ['Male (No Pepper)', 'Female (No Pepper)', 'Male (With Pepper)', 'Female (With Pepper)']
    colors = ['blue', 'orange', 'green', 'red']
    for median, group, color in zip(medians, groups, colors):
        axes[0].axvline(median, color=color, linestyle='-', label=f'Median ({group})')

    axes[0].set_xlabel('Adjusted Average Rating')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Adjusted Ratings by Gender and Pepper Status')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Boxplot for Adjusted Average Ratings by gender and pepper status
    axes[1].boxplot(
        [male_no_pepper, female_no_pepper, male_pepper, female_pepper],
        labels=[
            f'Male (No Pepper)\nMean: {means[0]:}',
            f'Female (No Pepper)\nMean: {means[1]:}',
            f'Male (With Pepper)\nMean: {means[2]:}',
            f'Female (With Pepper)\nMean: {means[3]:}'
        ],
        patch_artist=True
    )
    axes[1].set_ylabel('Adjusted Average Rating')
    axes[1].set_title('Boxplot of Adjusted Ratings by Gender and Pepper Status')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust spacing and display the plots
    plt.tight_layout()
    plt.show()

    # File paths
    num_data_path = 'rmpCapstoneNum.csv'
    tags_data_path = 'rmpCapstoneTags.csv'

    # Load datasets
    num_data = pd.read_csv(num_data_path)
    tags_data = pd.read_csv(tags_data_path)

    # Rename columns in numerical data for consistency
    num_data.rename(columns={
        'Male': 'Male',
        'Female': 'Female',
        'Average Rating': 'Average_Rating'
    }, inplace=True)

    # Normalize the tags dataset (z-score normalization)
    tags_data['Tough Grader Z-Score'] = (tags_data['Tough Grader'] - tags_data['Tough Grader'].mean()) / tags_data['Tough Grader'].std()
    tags_data['Good Feedback Z-Score'] = (tags_data['Good Feedback'] - tags_data['Good Feedback'].mean()) / tags_data['Good Feedback'].std()

    # Compute percentiles for Tough Grader and Good Feedback
    tough_grader_75th = tags_data['Tough Grader'].quantile(0.75)
    good_feedback_25th = tags_data['Good Feedback'].quantile(0.25)

    # Filter professors based on percentile thresholds
    tags_filtered = tags_data[
        (tags_data['Tough Grader'] >= tough_grader_75th) & 
        (tags_data['Good Feedback'] <= good_feedback_25th)
    ]

    # Merge tags data with numerical data
    filtered_data = num_data.merge(tags_filtered, left_index=True, right_index=True, how='inner')

    # Apply filters for minimum ratings and unambiguous gender
    filtered_data = filtered_data[
        (filtered_data['Number of Ratings'] >= threshold) & 
        (
            ((filtered_data['Male'] == 1) & (filtered_data['Female'] == 0)) |
            ((filtered_data['Male'] == 0) & (filtered_data['Female'] == 1))
        )
    ]

    # Compute gender-specific global means for Bayesian adjustment
    global_mean_male = filtered_data[filtered_data['Male'] == 1]['Average_Rating'].mean()
    global_mean_female = filtered_data[filtered_data['Female'] == 1]['Average_Rating'].mean()

    # Recompute adjusted average ratings for the subset using Bayesian adjustment
    filtered_data['Adjusted_Average'] = filtered_data.apply(
        lambda row: (
            (row['Average_Rating'] * row['Number of Ratings'] +
            (global_mean_male if row['Male'] == 1 else global_mean_female) * damping_factor) /
            (row['Number of Ratings'] + damping_factor)
        ), axis=1
    )

    # Separate adjusted ratings by gender for the subset
    male_ratings_tg_subset = filtered_data[filtered_data['Male'] == 1]['Adjusted_Average']
    female_ratings_tg_subset = filtered_data[filtered_data['Female'] == 1]['Adjusted_Average']

    # Perform Mann-Whitney U Test for gender bias in this subset
    mannwhitney_tg_subset = mannwhitneyu(male_ratings_tg_subset, female_ratings_tg_subset, alternative='greater')

    # Define significance level
    alpha = 0.005

    # Display results
    print("\n### Results for Tough Grader and Bad Feedback Subset ###")
    print(f"U Statistic: {mannwhitney_tg_subset.statistic:}")
    print(f"P-value: {mannwhitney_tg_subset.pvalue:}")
    print(f"Mean Male Adjusted Ratings: {male_ratings_tg_subset.mean():}")
    print(f"Median Male Adjusted Ratings: {male_ratings_tg_subset.median():}")
    print(f"Mean Female Adjusted Ratings: {female_ratings_tg_subset.mean():}")
    print(f"Median Female Adjusted Ratings: {female_ratings_tg_subset.median():}")

    # Check significance and interpret results
    if mannwhitney_tg_subset.pvalue < alpha:
        print("Result: Significant. There is evidence of pro-male bias.")
    else:
        print("Result: Not Significant. No evidence of pro-male bias.")

    return None

def Question_2(alpha=0.005):
    # --Question 2 Code and Answer-- 
    
    filtered_data=Preprocessing_Q1_3()

    # Separate data by gender
    male_ratings = filtered_data[filtered_data['Male'] == 1]['Adjusted_Average']
    female_ratings = filtered_data[filtered_data['Female'] == 1]['Adjusted_Average']

    # Calculate variance and standard deviation for male and female ratings
    male_variance = male_ratings.var()
    female_variance = female_ratings.var()
    male_std = male_ratings.std()
    female_std = female_ratings.std()

    # Display variance and standard deviation results
    print("\n### Variance and Standard Deviation ###")
    print(f"Male Variance: {male_variance:}, Male Std Dev: {male_std:}")
    print(f"Female Variance: {female_variance:}, Female Std Dev: {female_std:}")

    # Perform Levene's test for equality of variances
    levene_test = levene(male_ratings, female_ratings, center='median')

    # Display results of Levene's test
    print("\n### Levene’s Test for Equality of Variances ###")
    print(f"Test Statistic: {levene_test.statistic:}")
    print(f"P-value: {levene_test.pvalue:}")

    # Interpret Levene's test results
    alpha = 0.005
    if levene_test.pvalue < alpha:
        print("Result: The variances are significantly different between genders.")
    else:
        print("Result: No significant difference in variances between genders.")

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram for adjusted average ratings by gender
    axes[0].hist(male_ratings, bins=20, alpha=0.5, label='Male Professors', density=False)
    axes[0].hist(female_ratings, bins=20, alpha=0.5, label='Female Professors', density=False)
    axes[0].set_xlabel('Adjusted Average Rating')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Adjusted Average Ratings by Gender')
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Boxplot for adjusted average ratings by gender
    axes[1].boxplot(
        [male_ratings, female_ratings],
        labels=['Male Professors', 'Female Professors'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue', color='blue'),
        medianprops=dict(color='red'),
    )
    axes[1].set_ylabel('Adjusted Average Rating')
    axes[1].set_title('Boxplot of Adjusted Ratings by Gender')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and display the plots
    plt.tight_layout()
    plt.show()
    
    return None

def Question_3():
    # --Question 3 Code and Answer-- 
    
    filtered_data=Preprocessing_Q1_3()

    # Separate data by gender
    male_ratings = filtered_data[filtered_data['Male'] == 1]['Adjusted_Average']
    female_ratings = filtered_data[filtered_data['Female'] == 1]['Adjusted_Average']

    # Assuming 'male_ratings' and 'female_ratings' are available
    male_ratings = np.array(male_ratings)
    female_ratings = np.array(female_ratings)

    # Function to compute bootstrap confidence intervals
    def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=10000, ci=95):
        bootstraps = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
        stat_values = np.array([stat_func(bs) for bs in bootstraps])
        lower_bound = np.percentile(stat_values, (100 - ci) / 2)
        upper_bound = np.percentile(stat_values, 100 - (100 - ci) / 2)
        return lower_bound, upper_bound

    # Function to compute Cliff's delta
    def cliffs_delta(x, y):
        """Calculate Cliff's delta for two groups."""
        n1, n2 = len(x), len(y)
        favorable = sum(1 for xi in x for yj in y if xi > yj)
        unfavorable = sum(1 for xi in x for yj in y if xi < yj)
        delta = (favorable - unfavorable) / (n1 * n2)
        return delta

    # Function to bootstrap confidence interval for variance difference
    def bootstrap_variance_diff_ci(male_data, female_data, n_bootstrap=10000, ci=95):
        male_bootstraps = np.random.choice(male_data, (n_bootstrap, len(male_data)), replace=True)
        female_bootstraps = np.random.choice(female_data, (n_bootstrap, len(female_data)), replace=True)
        
        variance_differences = [
            male_bs.var() - female_bs.var()
            for male_bs, female_bs in zip(male_bootstraps, female_bootstraps)
        ]
        
        lower_bound = np.percentile(variance_differences, (100 - ci) / 2)
        upper_bound = np.percentile(variance_differences, 100 - (100 - ci) / 2)
        
        return lower_bound, upper_bound

    # Function to bootstrap confidence interval for variance ratio
    def bootstrap_variance_ratio_ci(male_data, female_data, n_bootstrap=10000, ci=95):
        male_bootstraps = np.random.choice(male_data, (n_bootstrap, len(male_data)), replace=True)
        female_bootstraps = np.random.choice(female_data, (n_bootstrap, len(female_data)), replace=True)
        
        variance_ratios = [
            male_bs.var() / female_bs.var()
            for male_bs, female_bs in zip(male_bootstraps, female_bootstraps)
        ]
        
        lower_bound = np.percentile(variance_ratios, (100 - ci) / 2)
        upper_bound = np.percentile(variance_ratios, 100 - (100 - ci) / 2)
        
        return lower_bound, upper_bound

    # --- Effect 1: Gender Bias in Average Ratings ---
    # Compute mean and confidence intervals for male and female ratings
    male_mean_ci = bootstrap_ci(male_ratings, np.mean)
    female_mean_ci = bootstrap_ci(female_ratings, np.mean)

    # Compute Cliff's delta
    cliffs_d = cliffs_delta(male_ratings, female_ratings)

    # Print results for average ratings
    print("\n### Effect 1: Gender Bias in Average Ratings ###")
    print(f"Male Mean: {male_ratings.mean():.4f}, 95% CI: {male_mean_ci}")
    print(f"Female Mean: {female_ratings.mean():.4f}, 95% CI: {female_mean_ci}")
    print(f"Cliff's Delta: {cliffs_d:.4f}")

    # Interpret Cliff's Delta
    if abs(cliffs_d) < 0.147:
        interpretation = "Negligible"
    elif abs(cliffs_d) < 0.33:
        interpretation = "Small"
    elif abs(cliffs_d) < 0.474:
        interpretation = "Medium"
    else:
        interpretation = "Large"
    print(f"Effect Size Interpretation: {interpretation}")

    # --- Effect 2: Gender Bias in Spread of Ratings ---
    # Compute variance and confidence intervals
    male_variance_ci = bootstrap_ci(male_ratings, np.var)
    female_variance_ci = bootstrap_ci(female_ratings, np.var)

    # Variance difference and bootstrap confidence interval
    variance_difference = male_ratings.var() - female_ratings.var()
    variance_diff_ci = bootstrap_variance_diff_ci(male_ratings, female_ratings)

    # Variance ratio and bootstrap confidence interval
    variance_ratio = male_ratings.var() / female_ratings.var()
    variance_ratio_ci = bootstrap_variance_ratio_ci(male_ratings, female_ratings)

    # Print results for variance
    print("\n### Effect 2: Gender Bias in Spread of Ratings ###")
    print(f"Male Variance: {male_ratings.var():.4f}, 95% CI: {male_variance_ci}")
    print(f"Female Variance: {female_ratings.var():.4f}, 95% CI: {female_variance_ci}")
    print(f"Variance Difference (Male - Female): {variance_difference:.4f}, 95% CI: {variance_diff_ci}")
    print(f"Variance Ratio (Male/Female): {variance_ratio:.4f}, 95% CI: {variance_ratio_ci}")

    # --- Visualizations ---
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot confidence intervals for means by gender
    axes[0].errorbar(
        x=0, y=male_ratings.mean(),
        yerr=[[male_ratings.mean() - male_mean_ci[0]], [male_mean_ci[1] - male_ratings.mean()]],
        fmt='o', color='blue', label='Male Confidence Interval'
    )
    axes[0].errorbar(
        x=1, y=female_ratings.mean(),
        yerr=[[female_ratings.mean() - female_mean_ci[0]], [female_mean_ci[1] - female_ratings.mean()]],
        fmt='o', color='orange', label='Female Confidence Interval'
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['Male', 'Female'])
    axes[0].set_xlabel("Gender")
    axes[0].set_ylabel("Adjusted Average Rating")
    axes[0].set_title("Bootstrap Confidence Intervals for Means by Gender")
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[0].legend(loc='upper right')

    # Plot confidence intervals for variance by gender
    axes[1].errorbar(
        x=0, y=male_ratings.var(),
        yerr=[[male_ratings.var() - male_variance_ci[0]], [male_variance_ci[1] - male_ratings.var()]],
        fmt='o', color='blue', label='Male Confidence Interval'
    )
    axes[1].errorbar(
        x=1, y=female_ratings.var(),
        yerr=[[female_ratings.var() - female_variance_ci[0]], [female_variance_ci[1] - female_ratings.var()]],
        fmt='o', color='orange', label='Female Confidence Interval'
    )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Male', 'Female'])
    axes[1].set_xlabel("Gender")
    axes[1].set_ylabel("Variance of Ratings")
    axes[1].set_title("Bootstrap Confidence Intervals for Variance by Gender")
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1].legend(loc='upper right')

    # Apply tight layout and show the plots
    plt.tight_layout()
    plt.show()

    return None


def Question_4():
    # --Question 4 Code and Answer-- 
    # Load the data
    tags_df = pd.read_csv("rmpCapstoneTags.csv")  # Path to tag data
    num_df = pd.read_csv("rmpCapstoneNum.csv")  # Path to gender data

    # Ensure alignment of rows
    assert len(tags_df) == len(num_df), "Mismatch in dataset lengths."

    # Filter rows where either Male = 1 or Female = 1, and at least 5 ratings
    filtered_rows = num_df[
        ((num_df['Male'] == 1) & (num_df['Female'] == 0)) |
        ((num_df['Male'] == 0) & (num_df['Female'] == 1))
    ]
    filtered_rows = filtered_rows[filtered_rows['Number of Ratings'] >= 5]

    # Apply the filtering to tags_df
    filtered_tags_df = tags_df.loc[filtered_rows.index]

    # Normalize tag counts by the number of ratings
    normalized_tags_df = filtered_tags_df.div(filtered_rows['Number of Ratings'], axis=0)

    # Separate the filtered gender data
    male_tags = normalized_tags_df[filtered_rows['Male'] == 1]
    female_tags = normalized_tags_df[filtered_rows['Female'] == 1]

    # Extract tag column names
    tag_columns = tags_df.columns

    # Store results
    results = []

    # Analyze each tag for gender differences using Mann-Whitney U Test
    for tag in tag_columns:
        # Perform Mann-Whitney U Test
        stat, p_value = mannwhitneyu(male_tags[tag], female_tags[tag], alternative='two-sided')
        
        # Compute mean normalized tag counts for males and females
        male_mean = male_tags[tag].mean()
        female_mean = female_tags[tag].mean()
        
        # Store results
        results.append({
            "Tag": tag,
            "Male Mean (Normalized)": male_mean,
            "Female Mean (Normalized)": female_mean,
            "P-Value": p_value
        })

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Apply significance threshold
    alpha = 0.005
    significant_tags = results_df[results_df['P-Value'] < alpha]

    # Sort results by p-value
    results_df = results_df.sort_values(by="P-Value")
    most_gendered = results_df.head(3)  # Tags with smallest p-values
    least_gendered = results_df.tail(3)  # Tags with largest p-values

    # Print results
    print("\n### Significant Tags (p < 0.005) ###")
    print(significant_tags)

    print("\n### Most Gendered Tags (Lowest P-Values) ###")
    print(most_gendered)

    print("\n### Least Gendered Tags (Highest P-Values) ###")
    print(least_gendered)

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot most gendered tags
    axes[0].bar(
        most_gendered['Tag'], most_gendered['Male Mean (Normalized)'],
        alpha=0.7, label='Male (Normalized)', color='blue'
    )
    axes[0].bar(
        most_gendered['Tag'], most_gendered['Female Mean (Normalized)'],
        alpha=0.7, label='Female (Normalized)', color='orange'
    )
    axes[0].set_title("Most Gendered Tags (Top 3 - Normalized)")
    axes[0].set_ylabel("Normalized Tag Count")
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # Plot least gendered tags
    axes[1].bar(
        least_gendered['Tag'], least_gendered['Male Mean (Normalized)'],
        alpha=0.7, label='Male (Normalized)', color='blue'
    )
    axes[1].bar(
        least_gendered['Tag'], least_gendered['Female Mean (Normalized)'],
        alpha=0.7, label='Female (Normalized)', color='orange'
    )
    axes[1].set_title("Least Gendered Tags (Bottom 3 - Normalized)")
    axes[1].set_ylabel("Normalized Tag Count")
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()

    return None

def Question_5():
    
    filtered_num_df=Preprocessing_Q5_6()

    # Separate adjusted difficulty scores by gender
    male_difficulty = filtered_num_df[filtered_num_df['Male'] == 1]['Adjusted Difficulty']
    female_difficulty = filtered_num_df[filtered_num_df['Female'] == 1]['Adjusted Difficulty']

    # Perform normality tests (Kolmogorov-Smirnov test)
    male_normality = ks_1samp(male_difficulty, cdf=norm.cdf, alternative='two-sided')
    female_normality = ks_1samp(female_difficulty, cdf=norm.cdf, alternative='two-sided')

    # Print normality test results
    print("\n### Normality Test Results (Kolmogorov-Smirnov) ###")
    print(f"Male Adjusted Difficulty: Test Statistic = {male_normality.statistic:}, P-value = {male_normality.pvalue:}")
    print(f"Female Adjusted Difficulty: Test Statistic = {female_normality.statistic:}, P-value = {female_normality.pvalue:}")

    if male_normality.pvalue < 0.005:
        print("Result: Male Adjusted Difficulty distribution is NOT normal.")
    else:
        print("Result: Male Adjusted Difficulty distribution is normal.")

    if female_normality.pvalue < 0.005:
        print("Result: Female Adjusted Difficulty distribution is NOT normal.")
    else:
        print("Result: Female Adjusted Difficulty distribution is normal.")

    # Perform Mann-Whitney U Test on adjusted difficulty
    stat, p_value = mannwhitneyu(male_difficulty, female_difficulty, alternative='two-sided')

    # Print Mann-Whitney U test results
    print("\n### Question 5: Gender Difference in Adjusted Average Difficulty ###")
    print(f"Male Adjusted Mean Difficulty: {male_difficulty.mean():}")
    print(f"Female Adjusted Mean Difficulty: {female_difficulty.mean():}")
    print(f"Mann-Whitney U Test Statistic: {stat:}")
    print(f"P-value: {p_value:}")

    # Visualize the distribution of adjusted difficulty ratings by gender
    plt.figure(figsize=(12, 6))

    # Male histogram
    plt.hist(male_difficulty, bins=20, alpha=0.5, label='Male Professors', color='blue', density=False)

    # Female histogram
    plt.hist(female_difficulty, bins=20, alpha=0.5, label='Female Professors', color='orange', density=False)

    # Customize plot
    plt.xlabel('Adjusted Average Difficulty')
    plt.ylabel('Count')
    plt.title('Distribution of Adjusted Average Difficulty Ratings by Gender')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Display the plot
    plt.tight_layout()
    plt.show()

    return None

def Question_6():
    # --Question 6 Code and Answer-- 

    filtered_num_df=Preprocessing_Q5_6()

    # Separate adjusted difficulty scores by gender
    male_difficulty = filtered_num_df[filtered_num_df['Male'] == 1]['Adjusted Difficulty']
    female_difficulty = filtered_num_df[filtered_num_df['Female'] == 1]['Adjusted Difficulty']

    # Confidence interval function using bootstrapping
    def bootstrap_ci(data, stat_func=np.mean, n_bootstrap=10000, ci=95):
        """
        Computes bootstrap confidence intervals for a given dataset and statistic.

        Parameters:
            data (array-like): Input data.
            stat_func (function): Function to compute the statistic (default is mean).
            n_bootstrap (int): Number of bootstrap samples (default is 10,000).
            ci (float): Confidence interval percentage (default is 95%).

        Returns:
            tuple: Lower and upper bounds of the confidence interval.
        """
        bootstraps = np.random.choice(data, (n_bootstrap, len(data)), replace=True)
        stat_values = np.array([stat_func(bs) for bs in bootstraps])
        lower_bound = np.percentile(stat_values, (100 - ci) / 2)
        upper_bound = np.percentile(stat_values, 100 - (100 - ci) / 2)
        return lower_bound, upper_bound

    # Compute Cliff's delta for effect size
    def cliffs_delta(x, y):
        """
        Calculate Cliff's delta for two groups.

        Parameters:
            x (array-like): Data from group 1.
            y (array-like): Data from group 2.

        Returns:
            float: Cliff's delta value.
        """
        n1, n2 = len(x), len(y)
        favorable = sum(1 for xi in x for yj in y if xi > yj)
        unfavorable = sum(1 for xi in x for yj in y if xi < yj)
        delta = (favorable - unfavorable) / (n1 * n2)
        return delta

    # Compute mean and confidence intervals for male and female difficulty
    male_diff_ci = bootstrap_ci(male_difficulty, np.mean)
    female_diff_ci = bootstrap_ci(female_difficulty, np.mean)

    # Compute the mean difference
    mean_diff = male_difficulty.mean() - female_difficulty.mean()

    # Compute Cliff's delta
    cliffs_d = cliffs_delta(male_difficulty, female_difficulty)

    # Interpret Cliff's Delta
    if abs(cliffs_d) < 0.147:
        interpretation = "Negligible"
    elif abs(cliffs_d) < 0.33:
        interpretation = "Small"
    elif abs(cliffs_d) < 0.474:
        interpretation = "Medium"
    else:
        interpretation = "Large"

    # Print results
    print("\n### Question 6: Quantifying the Size of the Effect ###")
    print(f"Male Mean Difficulty: {male_difficulty.mean():.4f}, 95% CI: {male_diff_ci}")
    print(f"Female Mean Difficulty: {female_difficulty.mean():.4f}, 95% CI: {female_diff_ci}")
    print(f"Mean Difference in Difficulty: {mean_diff:.4f}")
    print(f"Cliff's Delta: {cliffs_d:.4f}")
    print(f"Effect Size Interpretation: {interpretation}")

    # --- Visualization ---
    # Plot confidence intervals for difficulty by gender
    plt.figure(figsize=(10, 6))

    # Plot male confidence interval
    plt.errorbar(
        x=0, y=male_difficulty.mean(),
        yerr=[[male_difficulty.mean() - male_diff_ci[0]], [male_diff_ci[1] - male_difficulty.mean()]],
        fmt='o', color='blue', label='Male Confidence Interval'
    )

    # Plot female confidence interval
    plt.errorbar(
        x=1, y=female_difficulty.mean(),
        yerr=[[female_difficulty.mean() - female_diff_ci[0]], [female_diff_ci[1] - female_difficulty.mean()]],
        fmt='o', color='orange', label='Female Confidence Interval'
    )

    # Customize the plot
    plt.xticks([0, 1], ['Male', 'Female'])
    plt.xlabel("Gender")
    plt.ylabel("Average Difficulty")
    plt.title("Bootstrap Confidence Intervals for Difficulty by Gender")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')

    # Display the plot
    plt.tight_layout()
    plt.show()

def Question_7_1():
    # --Question 7 Code and Answer Approach 1 -- 

    file_path = 'rmpCapstoneNum.csv'
    data = pd.read_csv(file_path)

    categorical_columns = ["Received a Pepper?", "Male", "Female"]
    numerical_columns = ["Average Difficulty", "Number of Ratings", "Online Ratings"]
    target = "Average Rating"

    X = data[categorical_columns + numerical_columns]
    y = data[target]
    data_combined = pd.concat([X, y], axis=1)

    data_cleaned = data_combined.dropna()
    print("Shape of cleaned data (after dropping missing values):", data_cleaned.shape)

    X_clean = data_cleaned[numerical_columns + categorical_columns]
    y_clean = data_cleaned[target]

    X_clean = pd.get_dummies(X_clean, columns=categorical_columns, drop_first=True)

    numerical_predictors_cleaned = X_clean.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_clean, test_size=0.2, random_state=15456214)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    y_pred_train = ridge.predict(X_train)
    y_pred_test = ridge.predict(X_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    coefficients = pd.DataFrame({
        "Feature": numerical_predictors_cleaned,
        "Coefficient": ridge.coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    print("Ridge Regression Model (Excluding Missing Values)")
    print(f"Train R²: {r2_train:.2f}")
    print(f"Test R²: {r2_test:.2f}")
    print(f"Train RMSE: {rmse_train:.2f}")
    print(f"Test RMSE: {rmse_test:.2f}")
    print("\nFeature Importance (Coefficients):")
    print(coefficients)

    return None

def Question_7_2():

    file_path = 'rmpCapstoneNum.csv'
    data = pd.read_csv(file_path)

    data_cleaned = data.dropna()
    print("Shape of cleaned data (after dropping missing values):", data_cleaned.shape)

    predictors = ['Average Difficulty', 'Number of Ratings', 'Proportion Retake', 'Online Ratings', 'Male', 'Female']
    target = 'Average Rating'

    X = data_cleaned[predictors]
    y = data_cleaned[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    vif_data = pd.DataFrame()
    vif_data["Feature"] = predictors
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=15456214)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\nModel Performance:\nR²: {r2:.2f}\nRMSE: {rmse:.2f}")

    feature_importance = pd.DataFrame({
        'Feature': predictors,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # Visualizations

    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color='skyblue')
    plt.xlabel("Coefficient Value", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("Feature Importance (Linear Regression with Standardized Features)", fontsize=14)
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, residuals, color='purple', alpha=0.5)
    plt.axhline(y=0, color='black', linestyle='--', lw=1)
    plt.xlabel("Actual Ratings", fontsize=12)
    plt.ylabel("Residuals", fontsize=12)
    plt.title("Residual Plot", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Predicted vs Actual Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='green', alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=1)
    plt.xlabel("Actual Ratings", fontsize=12)
    plt.ylabel("Predicted Ratings", fontsize=12)
    plt.title("Predicted vs. Actual Ratings", fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return None

def Question_8():
    # --Question 8 Code and Answer Approach -- 

    # Load the ratings and tags datasets
    ratings_file_path = 'rmpCapstoneNum.csv'  # Replace with your file path
    tags_file_path = 'rmpCapstoneTags.csv'    # Replace with your file path

    ratings_data = pd.read_csv(ratings_file_path)
    tags_data = pd.read_csv(tags_file_path)

    # Filter the ratings data for professors with more than 5 ratings
    filtered_ratings = ratings_data[ratings_data['Number of Ratings'] >= 5]

    # Merge tags data with the filtered ratings dataset to get the target variable (Average Rating)
    merged_data = pd.concat([filtered_ratings['Average Rating'], tags_data.loc[filtered_ratings.index]], axis=1)

    # Drop rows with missing target variable (Average Rating)
    merged_data = merged_data.dropna(subset=['Average Rating'])

    # Normalize the tag columns by dividing by their row sum
    tag_columns = tags_data.columns
    normalized_data = merged_data.copy()
    normalized_data[tag_columns] = normalized_data[tag_columns].div(merged_data[tag_columns].sum(axis=1), axis=0)

    # Replace any NaN values resulting from normalization with 0
    normalized_data = normalized_data.fillna(0)

    # Split the dataset into predictors and target
    X = normalized_data[tag_columns]
    y = normalized_data['Average Rating']

    # Check multicollinearity using VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15456214)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': tag_columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    # Display results
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print("\nVIF Analysis:")
    print(vif_data)

    print("\nFeature Importance:")
    print(feature_importance)

    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot Feature Importance (Coefficients)
    axes[0].bar(feature_importance["Feature"], feature_importance["Coefficient"], color='lightgreen')
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Feature Importance (Coefficients)")
    axes[0].tick_params(axis='x', rotation=90)

    # Plot Predicted vs Actual Ratings
    axes[1].scatter(y_test, y_pred, alpha=0.6, edgecolors='k', color='purple')
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Actual Ratings")
    axes[1].set_ylabel("Predicted Ratings")
    axes[1].set_title("Predicted vs Actual Ratings")

    # Adjust layout for better spacing
    plt.tight_layout()
    # Show the combined plots
    plt.show()

    return None

def Question_9():
    # --Question 9 Code and Answer Approach -- 

    # Load the ratings and tags datasets
    ratings_file_path = 'rmpCapstoneNum.csv'  # Replace with your file path
    tags_file_path = 'rmpCapstoneTags.csv'    # Replace with your file path

    ratings_data = pd.read_csv(ratings_file_path)
    tags_data = pd.read_csv(tags_file_path)

    # Filter the ratings data for professors with more than 5 ratings
    filtered_ratings = ratings_data[ratings_data['Number of Ratings'] >= 5]

    # Merge tags data with the filtered ratings dataset to get the target variable (Average Rating)
    merged_data = pd.concat([filtered_ratings['Average Difficulty'], tags_data.loc[filtered_ratings.index]], axis=1)

    # Drop rows with missing target variable (Average Rating)
    merged_data = merged_data.dropna(subset=['Average Difficulty'])

    # Normalize the tag columns by dividing by their row sum
    tag_columns = tags_data.columns
    normalized_data = merged_data.copy()
    normalized_data[tag_columns] = normalized_data[tag_columns].div(merged_data[tag_columns].sum(axis=1), axis=0)

    # Replace any NaN values resulting from normalization with 0
    normalized_data = normalized_data.fillna(0)

    # Split the dataset into predictors and target
    X = normalized_data[tag_columns]
    y = normalized_data['Average Difficulty']

    # Check multicollinearity using VIF
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15456214)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': tag_columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    # Display results
    print(f"R²: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print("\nVIF Analysis:")
    print(vif_data)

    print("\nFeature Importance:")
    print(feature_importance)

    # Visualize VIF Analysis
    plt.figure(figsize=(12, 6))
    plt.bar(vif_data["Feature"], vif_data["VIF"], color='skyblue')
    plt.xlabel("Features")
    plt.ylabel("Variance Inflation Factor (VIF)")
    plt.title("VIF Analysis for Tag Features")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    # Create a figure with two subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Plot Feature Importance (Coefficients)
    axes[0].bar(feature_importance["Feature"], feature_importance["Coefficient"], color='lightgreen')
    axes[0].set_xlabel("Features")
    axes[0].set_ylabel("Coefficient")
    axes[0].set_title("Feature Importance (Coefficients)")
    axes[0].tick_params(axis='x', rotation=90)

    # Plot Predicted vs Actual Values
    axes[1].scatter(y_test, y_pred, alpha=0.6, edgecolors='k', color='purple')
    axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Actual Average Difficulty")
    axes[1].set_ylabel("Predicted Average Difficulty")
    axes[1].set_title("Predicted vs Actual Average Difficulty")

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the combined plots
    plt.show()

    return None

def Question_10():

    # Load data
    data_file = "preprocessed_combined_data.csv"
    comb_df = pd.read_csv(data_file)

    # Dynamically match column names to resolve naming issues
    numerical_features = [
        "Average Difficulty", "Number of Ratings", "Proportion Retake",
        "Online Ratings", "Male", "Female"
    ]
    tags_columns = [
        "Tough Grader", "Good Feedback", "Respected", "Lots to Read",
        "Participation Matters", "Don't Skip Class", "Lots of Homework", "Inspirational",
        "Pop Quizzes!", "Accessible", "So Many Papers", "Clear Grading",
        "Hilarious", "Test Heavy", "Graded by Few Things", "Amazing Lectures",
        "Caring", "Extra Credit", "Group Projects", "Lecture Heavy"
    ]

    # Ensure columns exist in the DataFrame
    numerical_features = [col for col in numerical_features if col in comb_df.columns]
    tags_columns = [col for col in tags_columns if col in comb_df.columns]

    # Combine features
    X = pd.concat([comb_df[numerical_features], comb_df[tags_columns]], axis=1)
    y = comb_df["Received a Pepper?"]

    # Handle missing values
    numerical_cols = X.select_dtypes(include=["float64", "int64"]).columns
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns

    X[numerical_cols] = X[numerical_cols].fillna(X[numerical_cols].mean())

    if len(categorical_cols) > 0:
        X[categorical_cols] = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

    # Clean target variable
    y = y.replace("NA", 0)
    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    # Address class imbalance using SMOTE
    smote = SMOTE(random_state=15456214)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=15456214)

    # Train Random Forest model
    rf = RandomForestClassifier(random_state=15456214)
    param_grid = {
        "n_estimators": [100],
        "max_depth": [10, None],
        "min_samples_split": [2]
    }
    grid_rf = GridSearchCV(rf, param_grid, scoring="roc_auc", cv=5)
    grid_rf.fit(X_train, y_train)

    # Best parameters
    print("\nBest Parameters from GridSearchCV:")
    print(grid_rf.best_params_)

    # Best model
    best_rf = grid_rf.best_estimator_

    # Predictions
    y_pred = best_rf.predict(X_test)
    y_proba = best_rf.predict_proba(X_test)[:, 1]

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # AU(ROC) score
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"AU(ROC): {roc_auc:.2f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc_value = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC Curve (AUC = {roc_auc_value:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve for Pepper Prediction", fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Model evaluation
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("Model Performance Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"AU(ROC): {roc_auc:.2f}")

    return None

def Question_11():
    # --Extra Credit Question Code and Answer Approach -- 
    data_file = "preprocessed_combined_data.csv"
    comb_df = pd.read_csv(data_file)

    # Correct column name for 'Major/Field'
    major_column = "Major/Field"

    # Unique values in the 'Major/Field' column
    unique_subjects = comb_df[major_column].unique()

    # STEM keywords, uncategorized subjects
    stem_keywords = [
        "math", "physics", "chemistry", "biology", "engineering", "computer",
        "informatics", "technology", "statistics", "biotechnology", "bioinformatics",
        "medical", "neuroscience", "health", "nursing", "bio", "geology", "data",
        "astronomy", "science", "kinesiology", "physiology", "mechanical"
    ]
    non_stem_keywords = [
        "art", "history", "english", "writing", "music", "drama", "film", "philosophy",
        "political", "communication", "religion", "theology", "media", "cultural",
        "education", "psychology", "sociology", "literature", "language", "theatre",
        "design", "management", "business", "law", "marketing"
    ]

    # Classify subjects based on keywords
    def classify_major(subject):
        if pd.isna(subject):
            return "Unknown"
        subject_lower = str(subject).lower()
        if any(keyword in subject_lower for keyword in stem_keywords):
            return "STEM"
        elif any(keyword in subject_lower for keyword in non_stem_keywords):
            return "Non-STEM"
        else:
            return "Uncategorized"

    # Apply classification
    comb_df["category"] = comb_df[major_column].apply(classify_major)

    # Count results
    category_counts = comb_df["category"].value_counts()
    print("\nCategory Counts:")
    print(category_counts)

    # Check uncategorized subjects
    uncategorized_subjects = comb_df[comb_df["category"] == "Uncategorized"][major_column].unique()
    print("\nUncategorized Subjects:")
    for subject in sorted(uncategorized_subjects):
        print(subject)

    # Split the dataset into STEM and non-STEM based on the classification
    stem_ratings = comb_df[comb_df["category"] == "STEM"]["Average Rating"].dropna()
    non_stem_ratings = comb_df[comb_df["category"] == "Non-STEM"]["Average Rating"].dropna()

    # Independent t-test
    t_stat, p_value = mannwhitneyu(stem_ratings, non_stem_ratings)

    # Output the results
    print("U-test Results:")
    print(f"U-statistic: {t_stat:}")
    print(f"P-value: {p_value:}")

    # Interpretation
    alpha = 0.005
    if p_value < alpha:
        print("The difference in average ratings between STEM and Non-STEM professors is statistically significant.")
    else:
        print("There is no statistically significant difference in average ratings between STEM and Non-STEM professors.")

    return None
    

def main():
    Basic_Processing()
    choice=-1
    while(choice!=0):
        choice=int(input("Choose which question you want answered ? - ")) 
        match choice:
            case 1:
                Question_1()
            case 2:
                Question_2()
            case 3:
                Question_3()
            case 4:
                Question_4()
            case 5:
                Question_5()
            case 6:
                Question_6()
            case 7:
                i=int(input("Choose which approach you want answered ? - ")) 
                if i==1:
                    Question_7_1()
                elif i==2:
                    Question_7_2()
                else:
                    print("Wrong Choice")
            case 8:
                Question_8()
            case 9:
                Question_9()
            case 10:
                Question_10()
            case 11:
                Question_11()
            case 0:
                print("Exit")
            case _:
                print("Default: No valid option selected.")

# Execute the main function
if __name__ == "__main__":
    main()