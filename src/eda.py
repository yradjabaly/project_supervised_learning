"""Exploratory data analysis: summary tables and figures.

Counts use the full dataset. Relationships with the target use the
training set only, so the test set stays unseen.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.config import FIG_DIR
from src.data import basic_summary, load_data, make_train_test_split

plt.rcParams["font.size"] = 11
UNIVERSITY_DEGREES = ["Bachelors", "Masters", "Prof-school", "Doctorate"]


def rate_by_group(df, feature):
    return (df.groupby(feature)["income"]
            .agg(n_individuals="size", rate_above_50k="mean")
            .sort_values("rate_above_50k"))


def iqr_bounds(series):
    """Tukey's bounds: [Q1 - 1.5 IQR, Q3 + 1.5 IQR]."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def iqr_outlier_share(series):
    """Share of values outside Tukey's bounds."""
    low, high = iqr_bounds(series)
    return ((series < low) | (series > high)).mean()


def missing_profile(df):
    """Compare people with and without a reported work class."""
    missing = df["workclass"].isna().map({False: "reported", True: "missing"})
    working_age = df["age"].between(21, 64)
    return pd.DataFrame({
        "aged <= 20 or >= 65": (~df["age"].between(21, 64)).groupby(missing).mean(),
        "works < 35 hours": (df["hours_per_week"] < 35).groupby(missing).mean(),
        "relationship = Own-child": (df["relationship"] == "Own-child").groupby(missing).mean(),
        "rate >50K": df["income"].groupby(missing).mean(),
        "rate >50K, aged 21-64": df.loc[working_age, "income"].groupby(missing[working_age]).mean(),
    }).T.round(3)


def country_table(df):
    """>50K rate and share of university graduates by country (at least 50 people)."""
    df = df.assign(university=df["education"].isin(UNIVERSITY_DEGREES))
    table = df.groupby("native_country").agg(
        n=("income", "size"), rate=("income", "mean"), share_university=("university", "mean"))
    return table[table["n"] >= 50].sort_values("rate")


def plot_histogram_by_class(df, feature, bins, bounds=None):
    """Each class is normalised separately: bars show the % of that class in each bin."""
    plt.figure(figsize=(6, 4))
    for label, name in [(0, "<=50K"), (1, ">50K")]:
        values = df.loc[df["income"] == label, feature]
        weights = np.ones(len(values)) / len(values)
        plt.hist(values, bins=bins, weights=weights, alpha=0.6, label=f"{name} (n={len(values)})")
    if bounds is not None:
        for bound in bounds:
            plt.axvline(bound, color="black", linestyle="--")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    plt.xlabel(feature)
    plt.ylabel("Share of the class")
    plt.title(f"Distribution of {feature} by income class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/hist_{feature}.pdf")
    plt.close()


def plot_capital_net(df):
    """Appendix: non-zero net capital income before and after the signed log transform."""
    net = df["capital_gain"] - df["capital_loss"]
    net = net[net != 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].hist(net, bins=50)
    axes[0].set_xlabel("capital_gain - capital_loss")
    axes[0].set_title("Before transform")
    axes[1].hist(np.sign(net) * np.log1p(np.abs(net)), bins=50)
    axes[1].set_xlabel("sign(x) log(1 + |x|)")
    axes[1].set_title("After transform (capital_net)")
    axes[0].set_ylabel("Number of individuals")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/capital_net_transform.pdf")
    plt.close(fig)


def plot_rate_by_group(df, feature):
    table = rate_by_group(df, feature)
    overall = df["income"].mean()

    fig, ax = plt.subplots(figsize=(6, max(4, 0.3 * len(table) + 1.5)))
    bars = ax.barh(table.index.astype(str), table["rate_above_50k"])
    labels = [f"{rate:.0%} (n={n})" for rate, n in
              zip(table["rate_above_50k"], table["n_individuals"])]
    ax.bar_label(bars, labels=labels, padding=3, fontsize=9)
    ax.axvline(overall, color="black", linestyle="--")

    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(0, table["rate_above_50k"].max() * 1.45)
    ax.set_xlabel(f"Share earning >50K (dashed line: overall rate, {overall:.0%})")
    ax.set_title(f"Share earning >50K by {feature}")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rate_{feature}.pdf")
    plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)
    df = load_data()

    # 1. Overview (full dataset)
    print(basic_summary(df))
    print("Duplicates if fnlwgt is ignored:", df.drop(columns="fnlwgt").duplicated().sum())
    print("education_num values per education level:",
          df.groupby("education")["education_num"].nunique().max())
    df = df.drop_duplicates()
    missing = df.isna().sum()
    print("\nMissing values (count and %):\n",
          pd.DataFrame({"n": missing, "pct": 100 * missing / len(df)})[missing > 0].round(1))
    print("\nworkclass missing (rows) vs occupation missing (columns):\n",
          pd.crosstab(df["workclass"].isna(), df["occupation"].isna()))
    print("workclass when only occupation is missing:",
          df.loc[df["occupation"].isna() & df["workclass"].notna(), "workclass"].value_counts().to_dict())
    print("\nNumerical features:\n", df.describe().T.round(1))

    # 2. Extreme and unusual values (full dataset)
    for col in ["age", "hours_per_week", "capital_gain"]:
        print(f"\nThree highest values of {col} (value: count):",
              df[col].value_counts().sort_index().tail(3).to_dict())
    print("\nShare flagged by the IQR rule:",
          {col: round(iqr_outlier_share(df[col]), 3) for col in ["age", "hours_per_week"]})
    print("Tukey bounds on age:", iqr_bounds(df["age"]))
    print("Share declaring exactly 40 hours:", round((df["hours_per_week"] == 40).mean(), 3))
    print("Share of zeros:", (df[["capital_gain", "capital_loss"]] == 0).mean().round(3).to_dict())
    print("capital_gain Q1, Q3:", df["capital_gain"].quantile([0.25, 0.75]).tolist())
    print("Median capital_gain among non-zero values:", df.loc[df["capital_gain"] > 0, "capital_gain"].median())
    print("Rows with both capital gain and capital loss:",
          ((df["capital_gain"] > 0) & (df["capital_loss"] > 0)).sum())
    print("Share born in the United States (country known):",
          round((df["native_country"].dropna() == "United-States").mean(), 3))
    print("Countries with fewer than 50 people:", (df["native_country"].value_counts() < 50).sum())
    print("\nRelationship vs sex:\n", pd.crosstab(df["relationship"], df["sex"]))

    # 3. Relationships with the target (training set only)
    train_df, test_df = make_train_test_split(df, df["income"])[:2]
    print(f"\nTraining set: {len(train_df)} rows, rate >50K = {train_df['income'].mean():.3f}")
    print(f"Test set: {len(test_df)} rows, rate >50K = {test_df['income'].mean():.3f}")
    print("\nProfile by work class status (training set):\n", missing_profile(train_df))
    # Display only: show missing values as their own bar (not the model's imputation)
    train_df = train_df.fillna({"workclass": "NA (not reported)", "occupation": "NA (not reported)"})

    for feature in ["workclass", "education", "marital_status", "occupation", "sex", "race"]:
        print(f"\n{rate_by_group(train_df, feature).round(3)}")
        plot_rate_by_group(train_df, feature)

    print("\nRate >50K when native_country is missing:",
          round(train_df.loc[train_df["native_country"].isna(), "income"].mean(), 3))
    print("Rate >50K when capital_gain = 99999:",
          round(train_df.loc[train_df["capital_gain"] == 99999, "income"].mean(), 3))

    known = train_df.dropna(subset=["native_country"])
    print("Rate >50K, born in the US vs elsewhere:",
          known.groupby(known["native_country"] == "United-States")["income"].mean().round(3).to_dict())
    countries = country_table(train_df)
    print(f"\n{countries.round(3)}")
    print("Correlation between >50K rate and share of university graduates across countries:",
          round(countries["rate"].corr(countries["share_university"]), 2))

    print("\nMedian age by class:", train_df.groupby("income")["age"].median().to_dict())
    print("Rate >50K under 25:", round(train_df.loc[train_df["age"] < 25, "income"].mean(), 3))
    hours_group = np.sign(train_df["hours_per_week"] - 40).map({-1: "< 40", 0: "= 40", 1: "> 40"})
    print("Rate >50K by weekly hours:", train_df.groupby(hours_group)["income"].mean().round(3).to_dict())
    print("Share with a capital gain, by class:",
          train_df.groupby("income")["capital_gain"].apply(lambda s: (s > 0).mean()).round(3).to_dict())

    low, high = iqr_bounds(train_df["hours_per_week"])
    print(f"IQR bounds on hours_per_week (training set): {low} and {high}")
    plot_histogram_by_class(train_df, "age", bins=np.arange(14.5, 95, 5))
    plot_histogram_by_class(train_df, "hours_per_week", bins=np.arange(0.5, 100.5, 5),
                            bounds=(low, high))
    plot_capital_net(train_df)
    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()