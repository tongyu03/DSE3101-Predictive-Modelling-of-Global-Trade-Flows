import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


#import trade product data
trade_pdt_df = pd.read_csv("data/cleaned data/10 years Trade Product Data.csv")

# Rename industries
industry_rename_map = {
    'Chemical products n.e.c.': 'Other Chemicals',
    'Electrical machinery and equipment and parts thereof; sound recorders and reproducers; television image and sound recorders and reproducers, parts and accessories of such articles': 'Electrical Equipment & AV Gear',
    'Essential oils and resinoids; perfumery, cosmetic or toilet preparations': 'Cosmetics & Fragrances',
    'Metal; miscellaneous products of base metal': 'Base Metal Products',
    'Mineral fuels, mineral oils and products of their distillation; bituminous substances; mineral waxes': 'Mineral Fuels & Oils',
    'Natural, cultured pearls; precious, semi-precious stones; precious metals, metals clad with precious metal, and articles thereof; imitation jewellery; coin': 'Jewellery & Precious Metals',
    'Nuclear reactors, boilers, machinery and mechanical appliances; parts thereof': 'Machinery & Boilers',
    'Optical, photographic, cinematographic, measuring, checking, medical or surgical instruments and apparatus; parts and accessories': 'Optical & Medical Instruments',
    'Organic chemicals': 'Organic Chemicals',
    'Plastics and articles thereof': 'Plastics',
    'Machinery and mechanical appliances, boilers, nuclear reactors; parts thereof': 'Machinery & Boilers'  # same as above
}

# plot trade line graph
def plot_trade_line_graph(country, industry, trade_data_df):
    # Filter the data for the given country and industry
    filtered_data = trade_data_df[
        (trade_data_df['Country'] == country) & 
        (trade_data_df['Product'] == industry)
    ]
    # Create the line plot
    fig = px.line(
        filtered_data, 
        x='Year', 
        y=['Imports', 'Exports'], 
        title=f"Trade of {industry} between Singapore and {country}",
        labels={'Year': 'Year', 'value': 'Trade Value (USD)', 'variable': 'Trade Type'},
        markers=True
    )
    # Customize the layout
    fig.update_layout(
        template='plotly_white',
        xaxis_title="Year",
        yaxis_title="Trade Value (USD)",
        legend_title="Trade Type",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[filtered_data['Year'].min(), 2024]) 
    )
    # Show the figure
    return fig














'''
# functions for old script

### Historical Trade Data
trade_df = pd.read_csv("data\cleaned data\cleaned_monthly_trade_data.csv")
exchange_df = pd.read_csv("data\cleaned data\ER_sg.csv")
gdp_df = pd.read_csv("data\cleaned data\Processed_GDP.csv")

# Function to generate trade graph across the years
def generate_trade_graph(df, partner_country, year):
    df_filtered = df[(df["Country"] == partner_country) & (df["Year"] >= 2009) & (df["Year"] <= 2024)]
    df_grouped = df_filtered.groupby(['Year'])[['Imports', 'Exports']].sum().reset_index()
    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_grouped['Year'], df_grouped['Imports'], label='Imports', color='red')
    ax.plot(df_grouped['Year'], df_grouped['Exports'], label='Exports', color='blue')
    ax.set_title(f"Trade Volume Between Singapore and {partner_country} Across the Years")
    ax.set_xlabel('Year')
    ax.set_ylabel('Trade Value (US$ Mil)')
    # Annotate based on specific conditions for each partner country
    if partner_country == "Hong Kong":
        ax.axvline(x=2016, color='green', linestyle='--')
        fig.text(0.1, -0.0001, f"Singapore and {partner_country} have an FTA since 2016", 
                 ha="left", fontsize=11, color='green')
    elif partner_country == "Saudi Arabia":
        ax.axvline(x=2018, color='green', linestyle='--')
        fig.text(0.1, -0.0001, f"Singapore and {partner_country} have an FTA only in 2018", 
                 ha="left", fontsize=11, color='green')
    else:
        # For other countries, add a note under the graph
        fig.text(0.1, -0.0001, f"Singapore and {partner_country} have an FTA across all years", 
                 ha="left", fontsize=11, color='green')
    year_data = df_grouped[df_grouped['Year'] == year]
    imports_value = year_data['Imports'].values[0]
    exports_value = year_data['Exports'].values[0]
    ax.scatter(year, imports_value, color='red', s=50, zorder=5)
    ax.scatter(year, exports_value, color='blue', s=50, zorder=5)
    ax.legend()
    return fig

# Function to generate the yearly trade graph
def generate_yearly_trade_graph(trade_df, country, year):
    country_df = trade_df[trade_df["Country"] == country]
    df = country_df[country_df["Year"] == year]
    df = df.sort_values(by="Month")
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Month"], df["Exports"], label="Exports", color="blue")
    ax.plot(df["Month"], df["Imports"], label="Imports", color="red")
    ax.set_title(f"Trade Volume Between Singapore and {country} in {year}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Trade Value (US$ Mil)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_names)  # Replace numbers with month names
    ax.legend()
    return fig


# Function to generate ER text
def get_ex_rate(exchange_df, country, year):
    ex_rate_row = exchange_df[(exchange_df["Data Source"] == country) & (exchange_df["Year"] == year)]
    if not ex_rate_row.empty:
        er = ex_rate_row["ER"].values[0]
        currency = ex_rate_row["Currency"].values[0]
        return f"{er} {currency} per SGD"
    else:
        return f"Exchange rate data not available for {country} in {year}"

def get_title_text(year):
    return f"Exchange Rate in {year}"

# Function to generate GDP text
gdp_df["Country"] = gdp_df["Country Code"].replace({
    "HKG": "Hong Kong",
    "KOR": "South Korea",
    "JPN": "Japan",
    "CHN": "China",
    "MYS": "Malaysia",
    "SAU": "Saudi Arabia",
    "SGP": "Singapore",
    "THA": "Thailand"
})
gdp_df = gdp_df[(gdp_df["Year"] >= 2003)]

def get_gdp_comparison(gdp_df, country, year):
    gdp_row_sg = gdp_df[(gdp_df["Country"] == "Singapore") & (gdp_df["Year"] == year)]
    gdp_row_ctry = gdp_df[(gdp_df["Country"] == country) & (gdp_df["Year"] == year)]

    if not gdp_row_sg.empty and not gdp_row_ctry.empty:
        sg_gdp = gdp_row_sg["GDP"].values[0] / 1e9  # Convert to billions
        ctry_gdp = gdp_row_ctry["GDP"].values[0] / 1e9  # Convert to billions

        value = f"Singapore GDP in {year}: {sg_gdp:,.2f}B USD<br>{country} GDP in {year}: {ctry_gdp:,.2f}B USD"
        return value
    else:
        return f"GDP data not available for {country} in {year}"
    
    '''