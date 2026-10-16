"""
Unemployment Analysis with Python (India, May 2019 - Jun 2020)
Cleaning, exploration, visualization, Covid-19 impact, seasonality, insights.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- 1. LOAD
import sys, glob
# Usage: python unemployment_analysis.py [path/to/file.csv]
if len(sys.argv) > 1:
    CSV = sys.argv[1]
else:
    found = glob.glob("Unemployment*India*.csv")
    if not found:
        sys.exit("CSV not found. Put it next to this script or run: python unemployment_analysis.py <file.csv>")
    CSV = found[0]
df = pd.read_csv(CSV, encoding="utf-8-sig")
print("Raw shape:", df.shape)

# ---------------------------------------------------------------- 2. CLEAN
df.columns = df.columns.str.strip()
df = df.rename(columns={
    "Estimated Unemployment Rate (%)": "Unemployment_Rate",
    "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "Labour_Participation",
})
df = df.dropna(how="all")                       # blank separator rows
for c in ["Region", "Date", "Frequency", "Area"]:
    df[c] = df[c].astype(str).str.strip()
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
df = df.drop_duplicates()
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.strftime("%b")
df["YearMonth"] = df["Date"].dt.to_period("M")

print("Clean shape:", df.shape)
print("Missing values:\n", df.isna().sum())
print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
print("States:", df["Region"].nunique())

# Covid period flags (India national lockdown began 25 Mar 2020)
def period(d):
    if d < pd.Timestamp("2020-03-01"):
        return "Pre-Covid"
    if d <= pd.Timestamp("2020-03-31"):
        return "Lockdown start (Mar 2020)"
    return "Covid (Apr-Jun 2020)"
df["Period"] = df["Date"].apply(period)

# ---------------------------------------------------------------- 3. EXPLORE
print("\nDescribe:\n", df[["Unemployment_Rate", "Employed", "Labour_Participation"]].describe().round(2))

# Simple average across states/areas per month (data has no population weights
# for rate, so use employed-weighted average as a better national proxy)
def wavg(g):
    return np.average(g["Unemployment_Rate"], weights=g["Employed"])
national = df.groupby("YearMonth").apply(wavg, include_groups=False).rename("Weighted_UR")
national_simple = df.groupby("YearMonth")["Unemployment_Rate"].mean().rename("Simple_UR")
nat = pd.concat([national, national_simple], axis=1)
nat.index = nat.index.to_timestamp()
print("\nMonthly national unemployment rate:\n", nat.round(2))

state_avg = df.groupby("Region")["Unemployment_Rate"].mean().sort_values(ascending=False)
print("\nTop 5 states by avg unemployment:\n", state_avg.head(5).round(2))
print("\nLowest 5 states:\n", state_avg.tail(5).round(2))

corr = df[["Unemployment_Rate", "Employed", "Labour_Participation"]].corr()
print("\nCorrelation:\n", corr.round(2))

# ---------------------------------------------------------------- 4. VISUALS
# 4.1 National trend with lockdown marker
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(nat.index, nat["Simple_UR"], marker="o", label="Simple average", color="#c0392b")
ax.plot(nat.index, nat["Weighted_UR"], marker="s", label="Employment-weighted", color="#2c3e50")
ax.axvline(pd.Timestamp("2020-03-25"), ls="--", color="grey")
ax.text(pd.Timestamp("2020-03-27"), nat["Simple_UR"].max() * 0.55, "National lockdown\n(25 Mar 2020)", fontsize=9)
ax.set_title("India: Monthly Unemployment Rate (May 2019 - Jun 2020)")
ax.set_ylabel("Unemployment rate (%)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(f"{OUT}/01_national_trend.png", dpi=150); plt.close(fig)

# 4.2 Rural vs Urban
ru = df.groupby(["YearMonth", "Area"])["Unemployment_Rate"].mean().unstack()
ru.index = ru.index.to_timestamp()
fig, ax = plt.subplots(figsize=(11, 5))
ru.plot(ax=ax, marker="o")
ax.axvline(pd.Timestamp("2020-03-25"), ls="--", color="grey")
ax.set_title("Rural vs Urban Unemployment Rate")
ax.set_ylabel("Unemployment rate (%)")
fig.tight_layout(); fig.savefig(f"{OUT}/02_rural_vs_urban.png", dpi=150); plt.close(fig)

# 4.3 Heatmap state x month
pivot = df.pivot_table(index="Region", columns="YearMonth", values="Unemployment_Rate", aggfunc="mean")
pivot.columns = [str(c) for c in pivot.columns]
fig, ax = plt.subplots(figsize=(13, 9))
sns.heatmap(pivot, cmap="YlOrRd", annot=False, ax=ax, cbar_kws={"label": "Unemployment rate (%)"})
ax.set_title("Unemployment Rate Heatmap: State x Month")
fig.tight_layout(); fig.savefig(f"{OUT}/03_state_month_heatmap.png", dpi=150); plt.close(fig)

# 4.4 Pre-Covid vs Covid by state
pre = df[df["Period"] == "Pre-Covid"].groupby("Region")["Unemployment_Rate"].mean()
cov = df[df["Period"] == "Covid (Apr-Jun 2020)"].groupby("Region")["Unemployment_Rate"].mean()
peak = df[df["YearMonth"].isin([pd.Period("2020-04"), pd.Period("2020-05")])] \
    .groupby("Region")["Unemployment_Rate"].mean()
impact = pd.DataFrame({"Pre_Covid": pre, "Covid_AprJun": cov, "Peak_AprMay": peak}).dropna()
impact["Change_pp"] = impact["Covid_AprJun"] - impact["Pre_Covid"]
impact = impact.sort_values("Change_pp", ascending=False)
impact.round(2).to_csv(f"{OUT}/covid_impact_by_state.csv")
print("\nCovid impact (percentage-point rise), top 8:\n", impact.head(8).round(2))

fig, ax = plt.subplots(figsize=(10, 9))
impact["Change_pp"].sort_values().plot.barh(ax=ax, color="#e67e22")
ax.set_title("Rise in Unemployment: Pre-Covid vs Apr-Jun 2020 (percentage points)")
ax.set_xlabel("Change (pp)")
fig.tight_layout(); fig.savefig(f"{OUT}/04_covid_impact_by_state.png", dpi=150); plt.close(fig)

# 4.5 Top states trend
top = impact.head(5).index
fig, ax = plt.subplots(figsize=(11, 5))
for s in top:
    t = df[df["Region"] == s].groupby("Date")["Unemployment_Rate"].mean()
    ax.plot(t.index, t.values, marker="o", label=s)
ax.axvline(pd.Timestamp("2020-03-25"), ls="--", color="grey")
ax.set_title("Five Most Affected States")
ax.set_ylabel("Unemployment rate (%)")
ax.legend()
fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(f"{OUT}/05_top_affected_states.png", dpi=150); plt.close(fig)

# 4.6 Labour participation & employed (national)
lp = df.groupby("YearMonth").agg(Employed=("Employed", "sum"), LPR=("Labour_Participation", "mean"))
lp.index = lp.index.to_timestamp()
fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.bar(lp.index, lp["Employed"] / 1e6, width=20, color="#95a5a6", label="Employed (millions)")
ax1.set_ylabel("Employed (millions, sample total)")
ax2 = ax1.twinx()
ax2.plot(lp.index, lp["LPR"], color="#2980b9", marker="o", label="Avg labour participation (%)")
ax2.set_ylabel("Labour participation rate (%)")
ax1.set_title("Employment and Labour Participation")
fig.autofmt_xdate()
fig.tight_layout(); fig.savefig(f"{OUT}/06_employment_participation.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 5. SEASONALITY
# Only pre-Covid months are fair for seasonality (Mar-Jun 2020 is shock-driven).
pc = df[df["Date"] < pd.Timestamp("2020-03-01")]
seasonal = pc.groupby(["Month", "Month_Name"])["Unemployment_Rate"].mean().reset_index().sort_values("Month")
# order months May 2019 -> Feb 2020 chronologically
order = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2]
seasonal["o"] = seasonal["Month"].apply(order.index)
seasonal = seasonal.sort_values("o")
print("\nPre-Covid month-wise average unemployment:\n", seasonal[["Month_Name", "Unemployment_Rate"]].round(2))

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=seasonal, x="Month_Name", y="Unemployment_Rate", color="#3498db", ax=ax)
ax.set_title("Pre-Covid Month-wise Average Unemployment (May 2019 - Feb 2020)")
ax.set_ylabel("Unemployment rate (%)")
ax.set_xlabel("")
fig.tight_layout(); fig.savefig(f"{OUT}/07_seasonality_precovid.png", dpi=150); plt.close(fig)

# Boxplot by period
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="Period", y="Unemployment_Rate", order=["Pre-Covid", "Lockdown start (Mar 2020)", "Covid (Apr-Jun 2020)"], ax=ax)
ax.set_title("Unemployment Rate Distribution by Period")
ax.set_ylabel("Unemployment rate (%)")
fig.tight_layout(); fig.savefig(f"{OUT}/08_period_boxplot.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- 6. KEY NUMBERS
pre_mean = df[df["Period"] == "Pre-Covid"]["Unemployment_Rate"].mean()
apr = df[df["YearMonth"] == pd.Period("2020-04")]["Unemployment_Rate"].mean()
may = df[df["YearMonth"] == pd.Period("2020-05")]["Unemployment_Rate"].mean()
jun = df[df["YearMonth"] == pd.Period("2020-06")]["Unemployment_Rate"].mean()
print(f"\nPre-Covid avg: {pre_mean:.2f}% | Apr 2020: {apr:.2f}% | May 2020: {may:.2f}% | Jun 2020: {jun:.2f}%")
emp_pre = df[df["Date"] == pd.Timestamp("2020-03-31")]["Employed"].sum()
emp_apr = df[df["Date"] == pd.Timestamp("2020-04-30")]["Employed"].sum()
print(f"Employed (sum across surveyed rows) Mar -> Apr 2020: {emp_pre/1e6:.1f}M -> {emp_apr/1e6:.1f}M "
      f"({(emp_apr/emp_pre-1)*100:.1f}%)")
print("\nRural vs Urban pre-Covid avg:\n",
      df[df["Period"] == "Pre-Covid"].groupby("Area")["Unemployment_Rate"].mean().round(2))
print("Rural vs Urban Apr-Jun 2020 avg:\n",
      df[df["Period"] == "Covid (Apr-Jun 2020)"].groupby("Area")["Unemployment_Rate"].mean().round(2))

df.to_csv(f"{OUT}/unemployment_cleaned.csv", index=False)
nat.round(2).to_csv(f"{OUT}/monthly_national_rate.csv")
print("\nSaved charts and CSVs to", OUT)