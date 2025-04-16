import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


#import trade product data
trade_pdt_df = pd.read_csv("data/cleaned data/10 years Trade Product Data.csv")
#import Geopolitical distance data
geo_pol_df = pd.read_csv("data/cleaned data/geopolitical_data.csv")
#import geopolitical distance data
from Geopolitical_dist import get_geopolitical_data
#import trade predictions data
trade_pred_df = pd.read_csv("data/cleaned data/trade_with_predicted.csv")
#import geopolitical dist year function
from Geopolitical_dist import get_geopolitical_data_for_year


# Pg 2 Figure 1: Bubble Plot


def plot_bubble(industry, trade_type_col, year, trade_pdt_df):
    # Filter the DataFrame for the specified industry and year
    filtered = trade_pdt_df[
        (trade_pdt_df['Product'] == industry) & 
        (trade_pdt_df['Year'] == year)
    ]
    # Exclude South Korea from the data
    filtered = filtered[filtered['Country'] != 'South Korea']
    
    if filtered.empty:
        return px.scatter(
            title="No data available for the selected filters.", x=[], y=[]
        )
    geo_data = get_geopolitical_data_for_year(year)
    if 'Country' not in geo_data.columns or 'Geopolitical_Score' not in geo_data.columns:
        print("Required columns missing in geopolitical data.")
        return None

    geo_data = geo_data[['Country', 'Geopolitical_Score']]  
    geo_data_dict = geo_data.set_index('Country')['Geopolitical_Score'].to_dict()
    
    # Add a new column to the filtered dataframe with the geopolitical scores
    filtered['Geopolitical_Score'] = filtered['Country'].map(geo_data_dict)

    # Sort the countries by Geopolitical_Score for consistent coloring
    geo_data = geo_data.sort_values(by="Geopolitical_Score", ascending=True)
    num_colors = len(geo_data)
    color_scale = px.colors.sequential.Viridis[:num_colors]
    # Reverse the color scale
    reversed_color_scale = color_scale[::-1]
    # Create the bubble plot 
    fig = px.scatter(
        filtered, 
        x="Country", 
        y=trade_type_col,
        size=trade_type_col,
        color="Geopolitical_Score",  # Color by Geopolitical_Score
        hover_name="Country", 
        hover_data={ 
            "Product": True, 
            trade_type_col: ':.2f',  
            "Country": False,  
            "Year": False,  
            "Geopolitical_Score": ':,.2f'  # Display the geopolitical score in hover
        },
        title=f"Singapore {industry} {trade_type_col} in {year}",
        size_max=60,
        color_continuous_scale=reversed_color_scale  # Use reversed color scale
    )
    fig.update_layout(
        xaxis=dict(
            title="Country", 
            categoryorder='array', 
            categoryarray=filtered.sort_values(by="Geopolitical_Score")["Country"].unique() 
        ),
        yaxis_title=trade_type_col,
        margin=dict(t=60, l=100, r=20, b=100),
        template='plotly_white',
        showlegend= False,  # Remove the legend
        title=dict(
            x=0.5, 
            xanchor='center',
            font=dict(size=20)),
        coloraxis_showscale=False
    )
    fig.add_annotation(
        text="Fig 1: Bubble plot displaying level of imports/exports for Singapore's main trade partners per industry, colored by geopolitical score",
        xref="paper", yref="paper",  
        x=0.5, y=-0.3,
        showarrow=False,
        font=dict(size=14), 
        align="center"
    )
    return fig

# Pg 2 Figure 2: Geopolitical Distance Bar Graph

# List of countries
Countries = ['China', 'Hong Kong', 'South Korea', 'Thailand', 'Malaysia', 'United States', 'Saudi Arabia', 'Japan', 'Indonesia']

def plot_geopol_distance(input_year):
    # Get geopolitical data for all countries in the input year
    year_data = get_geopolitical_data_for_year(input_year)
    year_data['Geopolitical_Score'] = year_data['Geopolitical_Score'].astype(float)
    year_data['Country'] = year_data['Country'].astype(str)
    
    if year_data.empty:
        print(f"No data available for the year {input_year}.")
        return None
    # Sort the countries by Geopolitical_Score in descending order for the bar plot
    year_data = year_data.sort_values(by="Geopolitical_Score", ascending=False)
    num_colors = len(year_data)
    color_scale = px.colors.sequential.Viridis[:num_colors] 
    fig = px.bar(
        year_data,
        x="Geopolitical_Score",
        y="Country",
        orientation="h",  
        title=f"Geopolitical Distance with Singapore in {input_year}",
        labels={"Geopolitical_Score": "Geopolitical Distance", "Country": "Country"},
        color="Country", 
        color_discrete_sequence=color_scale  
    )

    fig.update_layout(
        xaxis=dict(range=[year_data['Geopolitical_Score'].min() - 10, year_data['Geopolitical_Score'].max()]),  
        yaxis=dict(categoryorder='array', categoryarray=year_data['Country'].values),  
        margin=dict(t=60, l=100, r=20, b=100),  
        template="plotly_white", 
        showlegend=False,
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        )
    )
    fig.add_annotation(
        text="Fig 2: Bar plot displaying geopolitical distance of Singapore with key trade partner over the years",
        xref="paper", yref="paper", 
        x=0.5, y=-0.3,  
        showarrow=False,
        font=dict(size=14), 
        align="center"
    )
    return fig


# Pg 3 Fig 1: Trade line graph
def plot_trade_line_graph(country, industry, trade_pred_df):
    # Filter the data for the given country and industry
    filtered_data = trade_pred_df[
        (trade_pred_df['Country'] == country) & 
        (trade_pred_df['Product'] == industry)
    ]

    # Split historical (up to 2023) and predicted (2024 onwards)
    historical_data = filtered_data[filtered_data['Year'] <= 2023]
    predicted_data = filtered_data[filtered_data['Year'] >= 2023]
    fig = go.Figure()
    # Plot historical data (Imports)
    fig.add_trace(go.Scatter(
        x=historical_data['Year'], y=historical_data['Imports'],
        mode='lines+markers',
        name='Imports',
        line=dict(color='#1f9e89', dash='solid')
    ))
    # Plot historical data (Exports)
    fig.add_trace(go.Scatter(
        x=historical_data['Year'], y=historical_data['Exports'],
        mode='lines+markers',
        name='Exports',
        line=dict(color='#f8961e', dash='solid')
    ))
    # Plot predicted data (without legend entry)
    fig.add_trace(go.Scatter(
        x=predicted_data['Year'], y=predicted_data['Imports'],
        mode='lines+markers',
        line=dict(color='#1f9e89', dash='dot'),
        showlegend=False  
    ))
    fig.add_trace(go.Scatter(
        x=predicted_data['Year'], y=predicted_data['Exports'],
        mode='lines+markers',
        line=dict(color='#f8961e', dash='dot'),
        showlegend=False  
    ))

    fig.update_layout(
        title=f"Trade of {industry} between Singapore and {country}",
        template='plotly_white',
        xaxis_title="Year",
        yaxis_title="Trade Value (USD)",
        legend_title="Trade Type",
        margin=dict(t=60, l=100, r=20, b=100),
        xaxis=dict(range=[filtered_data['Year'].min(), 2024.05]),
        title_font=dict(size=20),
        title_x=0.5,
        shapes=[
            dict(
                type="line",
                x0=2023, x1=2023,
                y0=0, y1=1,
                xref='x',
                yref='paper',
                line=dict(color="red", width=1, dash="dash")
            )
        ]
    )
    fig.add_annotation(
        text="Fig 3: Line plot displaying level of imports/exports for specified trade partner per industry over the years",
        xref="paper", yref="paper",
        x=0.5, y=-0.3,
        showarrow=False,
        font=dict(size=14),
        align="center"
    )
    return fig



# Pg 3 Fig 2: Geopolitical Distance line graph
def plot_geo_pol_line_graph(country):
    years = geo_pol_df["year"].unique()
    years.sort()
    scores = []
    for y in years:
        data = get_geopolitical_data(country, y)
        if not data.empty:
            scores.append(data[["year", "Geopolitical_Score"]].iloc[0])
    score_df = pd.DataFrame(scores)
    fig = px.line(score_df, 
                  x = 'year', 
                  y = "Geopolitical_Score", 
                  title=f'Geopolitical Distance of {country} with Singapore Over Time',
                  labels={'year': 'Year', 'Geopolitical_Score': 'Geopolitical Score'},
                  markers=True)
    fig.update_layout(
        template='plotly_white',
        xaxis_title="Year",
        yaxis_title="Geopolitical Distance",
        margin=dict(t=60, l=100, r=20, b=100),
        xaxis=dict(range=[score_df['year'].min(), 2024]),
        title=dict(
            x=0.5,
            xanchor='center',
            font=dict(size=20))
    )
    fig.add_annotation(
        text="Fig 4: Line plot displaying geopolitical distance of Singapore with specified trade partner over the years",
        xref="paper", yref="paper",  
        x=0.5, y=-0.3,  
        showarrow=False,
        font=dict(size=14), 
        align="center"
    )
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
