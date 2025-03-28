import seaborn as sns
import pandas as pd
import plotly.express as px
from shiny import App, ui, reactive, render
from shinywidgets import render_plotly
import matplotlib.pyplot as plt

### Historical Trade Data
trade_df = pd.read_csv("../data/cleaned_monthly_trade_data.csv")

# Function to generate the trade graph
def generate_trade_graph(trade_df, country, year):
    country_df = trade_df[trade_df["Country"] == country]
    df = country_df[country_df["Year"] == year]
    df = df.sort_values(by="Month")

    # Create the plot using Plotly
    fig = px.line(
        df, 
        x="Month", 
        y=["Exports", "Imports"], 
        labels={"value": "Trade Value (US$ Mil)", "year": "Year"},
        title=f"Trade Volume Between Singapore and {country}",
        color_discrete_map={"Exports": "blue", "Imports": "red"}
        )
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(1, 13))),
        template="plotly_white",
        legend_title=None
    )
    return fig

app_ui = ui.page_fluid(
    ui.navset_pill_list(  
        ui.nav_panel("Introduction", "Explain project + how to use"),
        ui.nav_panel("Historical Trade", 
                     
                      ui.input_selectize(
                          "select_country", "Select a Trade Partner:", 
                          choices=["China", "Hong Kong", "Japan", "Korea, Rep of", "Malaysia", "Saudi Arabia", "Thailand", "United States"],  # Options for the user to select
                          selected="China"  # Default selected value
                      ),
                      ui.input_slider("slide_year", "Choose a Year", 2003, 2025, value = 2024),
                      ui.output_plot("trade_plot"),  # Output plot will be rendered here
                    ),
        ui.nav_panel("Predicted Trade Volume", "model"),
        ui.nav_panel("Trading Ports", "interactive map"),
    )
)  


def server(input, output, session):
    # Define trade_plot as a reactive calculation
    @reactive.calc
    def trade_plot():
        country = input.select_country()
        year = input.slide_year()
        return generate_trade_graph(trade_df, country, year)

    # Render the plot
    @output
    @render.plot
    def plot():
        return trade_plot()


app = App(app_ui, server)