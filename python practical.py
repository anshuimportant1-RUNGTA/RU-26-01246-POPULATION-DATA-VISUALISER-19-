"""
PS 19 - Population Data Visualizer
Rungta International Skills University | B.Tech CSE

Analyses city population data from a CSV file (columns: City, Year, Population)
to find the total population, the largest city and each city's growth between
years, and visualises the results with Matplotlib charts.

Growth % = (latest - earliest) / earliest * 100
Usage:  python population_visualizer.py [path_to_csv]
Note:   Dummy data only. If the CSV does not exist, a dummy one is created.
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_CSV = "population.csv"
COLUMNS = ["City", "Year", "Population"]

DUMMY_DATA = (
    "Raipur,2020,1000000\n"
    "Raipur,2024,1100000\n"
    "Durg,2020,500000\n"
    "Durg,2024,600000\n"
)


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
def create_dummy_csv(path):
    """Create a dummy CSV file (used only when the file is missing)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(DUMMY_DATA)
    print(f"'{path}' not found. Dummy data file created.")


def load_and_clean(path):
    """Load the CSV with Pandas and drop invalid / duplicate rows.

    Row numbers in messages are 1-based data-row numbers (header excluded).
    Returns a clean DataFrame with columns City (str), Year (int),
    Population (int).
    """
    try:
        raw = pd.read_csv(path, header=None, names=COLUMNS, dtype=str,
                          skipinitialspace=True, skip_blank_lines=True)
    except pd.errors.EmptyDataError:
        print("Error: the CSV file is empty.")
        return pd.DataFrame(columns=COLUMNS)

    # Skip an optional header line such as "City,Year,Population"
    if len(raw) and str(raw.iloc[0]["City"]).strip().lower() == "city":
        raw = raw.iloc[1:].reset_index(drop=True)

    valid_rows = []
    for i, row in raw.iterrows():
        row_no = i + 1
        city = row["City"]
        year = row["Year"]
        pop = row["Population"]

        if pd.isna(city) or str(city).strip() == "":
            print(f"Row {row_no} skipped: City is missing.")
            continue
        city = str(city).strip()

        try:
            year = int(str(year).strip())
        except (ValueError, TypeError):
            print(f"Row {row_no} skipped: invalid Year '{year}'.")
            continue

        try:
            pop = int(float(str(pop).strip()))
        except (ValueError, TypeError):
            print(f"Row {row_no} skipped: invalid Population '{pop}'.")
            continue

        if pop <= 0:
            print(f"Row {row_no} skipped: Population must be positive.")
            continue

        valid_rows.append({"City": city, "Year": year, "Population": pop})

    df = pd.DataFrame(valid_rows, columns=COLUMNS)

    # Remove duplicate (City, Year) rows, keeping the first one
    before = len(df)
    df = df.drop_duplicates(subset=["City", "Year"], keep="first")
    removed = before - len(df)
    if removed:
        print(f"{removed} duplicate row(s) removed.")

    print(f"Valid rows: {len(df)}")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Analysis (uses groupby)
# --------------------------------------------------------------------------
def analyse(df):
    """Return a per-city table: Earliest, Latest, Years, Growth %.

    sort=False keeps cities in the order they first appear in the CSV.
    """
    rows = []
    for city, g in df.groupby("City", sort=False):
        g = g.sort_values("Year")
        first, last = g.iloc[0], g.iloc[-1]
        growth = None
        if first["Year"] != last["Year"]:
            growth = (last["Population"] - first["Population"]) \
                / first["Population"] * 100
        rows.append({
            "City": city,
            "EarliestYear": int(first["Year"]),
            "LatestYear": int(last["Year"]),
            "EarliestPop": int(first["Population"]),
            "LatestPop": int(last["Population"]),
            "Growth": growth,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Menu option 1: Summary
# --------------------------------------------------------------------------
def show_summary(df):
    if df.empty:
        print("No valid data to analyse.")
        return

    stats = analyse(df)
    total = int(stats["LatestPop"].sum())
    latest_year = int(df["Year"].max())
    largest = stats.loc[stats["LatestPop"].idxmax()]
    growth_stats = stats.dropna(subset=["Growth"])

    if df["Year"].nunique() < 2 or growth_stats.empty:
        # Only one year of data -> growth cannot be calculated
        print(f"Total population: {total} | "
              f"Largest city: {largest['City']}")
        print("Growth: not available (only one year of data)")
        return

    fastest = growth_stats.loc[growth_stats["Growth"].idxmax()]
    print(f"Total population ({latest_year}): {total}")
    print(f"Largest city: {largest['City']} ({largest['LatestPop']})")
    print(f"Fastest growing: {fastest['City']} (+{fastest['Growth']:.2f}%)")


# --------------------------------------------------------------------------
# Menu option 2: Growth and charts
# --------------------------------------------------------------------------
def show_growth_and_charts(df):
    if df.empty:
        print("No valid data to analyse.")
        return

    stats = analyse(df)
    growth_stats = stats.dropna(subset=["Growth"])
    if growth_stats.empty:
        print("Growth: not available (only one year of data)")
        return

    print(" | ".join(f"{r['City']}: {r['Growth']:+.2f}%"
                     for _, r in growth_stats.iterrows()))

    # Chart 1: Bar chart - growth % by city
    plt.figure(figsize=(7, 5))
    bars = plt.bar(growth_stats["City"], growth_stats["Growth"],
                   color="steelblue")
    for bar, val in zip(bars, growth_stats["Growth"]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}%", ha="center", va="bottom")
    plt.title("Population Growth by City")
    plt.xlabel("City")
    plt.ylabel("Growth (%)")
    plt.tight_layout()
    plt.savefig("population_growth_bar.png", dpi=120)

    # Chart 2: Line chart - population trend over years
    plt.figure(figsize=(7, 5))
    for city, g in df.groupby("City", sort=False):
        g = g.sort_values("Year")
        plt.plot(g["Year"], g["Population"], marker="o", label=city)
    plt.title("Population Trend")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.xticks(sorted(df["Year"].unique()))
    plt.legend(title="City")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("population_trend_line.png", dpi=120)

    print("Bar chart 'Population Growth by City' and line chart "
          "'Population Trend' displayed")
    print("(Charts also saved as population_growth_bar.png and "
          "population_trend_line.png)")
    plt.show()


# --------------------------------------------------------------------------
# Main menu
# --------------------------------------------------------------------------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(path):
        create_dummy_csv(path)

    df = load_and_clean(path)

    while True:
        print("\n=== Population Data Visualizer ===")
        print("1. Summary")
        print("2. Growth and charts")
        print("3. Exit")
        try:
            choice = input("Enter your choice: ").strip()
        except EOFError:
            break

        if choice == "1":
            show_summary(df)
        elif choice == "2":
            show_growth_and_charts(df)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2 or 3.")


if __name__ == "__main__":
    main()