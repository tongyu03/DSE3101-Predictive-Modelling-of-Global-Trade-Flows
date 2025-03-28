import seaborn as sns
import pandas as pd
import plotly.express as px
from shiny import App, ui, reactive, render
from shinywidgets import render_plotly
import matplotlib.pyplot as plt
import tempfile

### Historical Trade Data
trade_df = pd.read_csv("data/cleaned_monthly_trade_data.csv")

# Function to generate the trade graph
def generate_trade_graph(trade_df, country, year):
    country_df = trade_df[trade_df["Country"] == country]
    df = country_df[country_df["Year"] == year]
    df = df.sort_values(by="Month")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Month"], df["Exports"], label="Exports", color="blue")
    ax.plot(df["Month"], df["Imports"], label="Imports", color="red")

    ax.set_title(f"Trade Volume Between Singapore and {country}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Trade Value (US$ Mil)")
    ax.set_xticks(range(1, 13))
    ax.legend()

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
                      ui.input_slider("slide_year", "Choose a Year:", 2003, 2025, value = 2024),
                      ui.output_plot("trade_plot"),  # Output plot will be rendered here
                      ui.output_image("trade_image")  # New image output
                    ),
        ui.nav_panel("Predicted Trade Volume", "model"),
        ui.nav_panel("Trading Ports", "interactive map"),
    )
)  


def server(input, output, session):

    @output
    @render.plot
    def trade_plot():
        country = input.select_country()
        year = input.slide_year()
        return generate_trade_graph(trade_df, country, year)
    
    @output
    @render.image
    def trade_image():
        country = input.select_country()
        year = input.slide_year()
        fig = generate_trade_graph(trade_df, country, year)  
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(temp_file.name) 
        temp_file.close()

        return {"src": temp_file.name, "width": "70%"}


app = App(app_ui, server)