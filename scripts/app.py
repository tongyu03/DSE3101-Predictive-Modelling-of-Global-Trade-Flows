import seaborn as sns
import pandas as pd
import plotly.express as px
from shiny import App, ui, reactive, render
from shinywidgets import render_plotly
import matplotlib.pyplot as plt
import seaborn as sns
import json
# Import data for map
import pathlib
from ipyleaflet import Map, GeoData, basemaps, LayersControl, Marker, Popup
from ipywidgets import HTML
import shinywidgets
from shinywidgets import render_widget

### Historical Trade Data
trade_df = pd.read_csv("data/cleaned data/cleaned_monthly_trade_data.csv")
exchange_df = pd.read_csv("data/cleaned data/ER_sg.csv")
gdp_df = pd.read_csv("data/cleaned data/Processed_GDP.csv")

## Port location trade data
with open(r'data\ports.json', 'r', encoding='utf-8') as f:
    ports = json.load(f)

# Country coordinates
countries_coords = {
    "China": (35.8617, 104.1954),
    "Hong Kong": (22.3193, 114.1694),
    "Japan": (36.2048, 138.2529),
    "South Korea": (35.9078, 127.7669),
    "Malaysia": (4.2105, 101.9758),
    "Saudi Arabia": (23.8859, 45.0792),
    "Thailand": (15.8700, 100.9925),
    "U.S.A": (37.09020, -95.7129)
}

# Get city coordinates for each country
cities_coords = {}

for port in ports:
    country = port["COUNTRY"]
    city = port["CITY"]
    lat, lon = port["LATITUDE"], port["LONGITUDE"]
    # Initialize country if not exists
    if country not in cities_coords:
        cities_coords[country] = {}
    # Add city and its coordinates
    cities_coords[country][city] = (lat, lon)


# Function to generate trade graph across the years
def generate_trade_graph(df, partner_country):
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


### ui
app_ui = ui.page_fluid(
    ui.navset_pill_list(  
        ui.nav_panel("Introduction", "Explain project + how to use"
                     
                     ),
        ui.nav_panel("Historical Trade", 
                      ui.input_selectize(
                          "select_country", "Select a Trade Partner:", 
                          choices=["China", "Hong Kong", "Japan", "South Korea", "Malaysia", "Saudi Arabia", "Thailand", "United States"],  # Options for the user to select
                          selected="China"  # Default selected value
                      ),
                      ui.output_plot("trade_plot_years"),  # line graph
                      ui.hr(), 
                      ui.input_slider("slide_year", "Choose a Year:", 2009, 2024, value = 2020),
                      ui.output_plot("trade_plot"),  # line graph
                      ui.value_box(
                          ui.output_text("er_value_title"),  # Dynamic title
                          ui.output_text("er_value_text"),  # Dynamic text for the value
                          theme = "bg-gradient-indigo-purple"
                      ),
                      ui.value_box(
                          "",  # Empty title
                          ui.output_ui("gdp_value_text"),  # GDP value dynamically updated
                          theme = "bg-gradient-indigo-purple"
                          ),
                    ),
        ui.nav_panel("Predicted Trade Volume", "model"),
        ui.nav_panel("Trading Ports", 
                     ui.input_select("country", label = "Select country", choices = list(countries_coords.keys())),
                     ui.input_select("city", label="Select city", choices=[]),
                     shinywidgets.output_widget("map") # Output widget for the map
                    )
    )
)

### server
def server(input, output, session):
    
    @output
    @render.plot
    def trade_plot_years():
        country = input.select_country()
        return generate_trade_graph(trade_df, country)

    @output
    @render.plot
    def trade_plot():
        country = input.select_country()
        year = input.slide_year()
        return generate_yearly_trade_graph(trade_df, country, year)
    
    # Reactive rendering of the title (Exchange Rate in {year})
    @output
    @render.text
    def er_value_title():
        year = input.slide_year()
        return get_title_text(year)

    # Reactive rendering of exchange rate text for the value box
    @output
    @render.text
    def er_value_text():
        country = input.select_country()
        year = input.slide_year()
        return get_ex_rate(exchange_df, country, year)
    
    # Reactive value for GDP comparison 
    @output
    @render.ui
    def gdp_value_text():
        country = input.select_country()
        year = input.slide_year()
        value = get_gdp_comparison(gdp_df, country, year)

        # Return formatted text with line breaks
        return ui.HTML(f"<p>{value}</p>")

    
    @reactive.effect
    def update_cities():
        country = input.country()
        city_choices = list(cities_coords.get(country, {}).keys())

        if city_choices:
            ui.update_select("city", choices=city_choices)  
        else:
            ui.update_select("city", choices=[])
    
    @render_widget
    def map():
        # Initialize the map centered on Singapore
        m = Map(center=(1.290270, 103.851959), zoom=12)

        # Add port markers
        for port in ports:
            if port["COUNTRY"] in countries_coords:
                marker = Marker(location=(port["LATITUDE"], port["LONGITUDE"]))
                m.add(marker)

        m.add(LayersControl())  # FIXED: Add control after markers
        return m

    @reactive.effect
    def update_map():
        city = input.city()
        country = input.country()
        if city and country:
            coords = cities_coords.get(country, {}).get(city)
        else:
            coords = countries_coords.get(country)
        
        if coords:
            map.widget.center = coords

app = App(app_ui, server)