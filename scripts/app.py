from shiny import App, ui, reactive, render
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shinywidgets
from shinywidgets import render_widget
from shinyswatch import theme
import numpy as np
import matplotlib.pyplot as plt


#import trade product data
trade_pdt_df = pd.read_csv("data/cleaned data/10_years_trade_frontend.csv")

#product list for industry
product_list = sorted(trade_pdt_df["Product"].dropna().unique().tolist())

# import intro text
def read_intro():
    with open("data\intro.txt", "r", encoding="utf-8") as f:
        return f.read()

# import functions
from shiny_functions import plot_trade_line_graph
from shiny_functions import plot_geopol_distance

# Create Synthetic data for Geopolitical Distance
np.random.seed(42) 
years = list(range(2013, 2023 + 1))
countries = ["China", "Hong Kong", "South Korea", "Thailand", "Malaysia",
             "Japan", "USA", "Indonesia", "Saudi Arabia"]
# Create a DataFrame with all combinations
data = pd.DataFrame([(year, country) for year in years for country in countries],
                    columns=["year", "country"])
data["geo_distance"] = np.round(np.random.uniform(0.2, 0.9, size=len(data)), 2)


## ui
app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style("""
            .navbar { background-color: #004080 !important; }
            .navbar-brand { color: white !important; font-weight: bold; }
            .nav-link { color: white !important; }
            .nav-link.active { background-color: #0066cc !important; color: white !important; }
            .nav-link:hover { background-color: #0059b3 !important; color: #ffffff !important; }
        """)
    ),
    ui.page_navbar(
        ui.nav_panel(
            "Introduction",
            ui.output_ui("intro_text")
        ),

        ui.nav_panel(
            "Historical Trade Data Across Countries",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_selectize(
                        "select_industry1", "Select an Industry:",
                        choices=product_list,
                        selected=product_list[0] if product_list else None
                        ),
                    ui.input_selectize(
                        "select_trade", "Select Imports or Exports:",
                        choices=["Exports", "Imports"]
                    ),
                    ui.input_slider("slide_year", "Choose a Year:", 2013, 2023, value=2020),
                ),
                    shinywidgets.output_widget("bar_plot"),
                    shinywidgets.output_widget("bubble_plot")

            )
        ),

        ui.nav_panel(
            "Trade Data Across Time",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_selectize(
                        "select_country", "Select a Trade Partner:",
                        choices=["China", "Hong Kong", "Japan", "South Korea", "Malaysia", "Saudi Arabia", "Thailand", "United States"],
                        selected="China"
                    ),
                    ui.input_selectize(
                        "select_industry2", "Select an Industry:",
                        choices=product_list,
                        selected=product_list[0] if product_list else None
                    )
                ),
                shinywidgets.output_widget("trade_lineplot")

            )
        ),
        title="TideTrackers",
        id="page"
    ),
    theme=theme.darkly()
)


def server(input, output, session):

    @output
    @render.ui
    def intro_text():
        text_content = read_intro().replace("\n", "<br>") 
        return ui.HTML(f"<p>{text_content}</p>")  # Display formatted text
    
    @output
    @render_widget
    def bar_plot():
        return plot_geopol_distance(data, input.slide_year())

    
    @output
    @render_widget
    def bubble_plot():
        trade_type_col = input.select_trade().strip()
        filtered = trade_pdt_df[
            (trade_pdt_df['Product'] == input.select_industry1()) &
            (trade_pdt_df['Year'] == input.slide_year())
        ]

        if filtered.empty:
            return px.scatter(
            title="No data available for the selected filters.",
            x=[],
            y=[]
            )
        #plotting
        fig = px.scatter(
            filtered, 
            x= "Country", 
            y= trade_type_col,
            size= trade_type_col,
            color= "Country", 
            hover_name= "Product",
            title= f"Singapore {input.select_industry1()} {trade_type_col} in {input.slide_year()}",
            size_max= 60
        )

        fig.update_layout(
        xaxis_title="Country",
        yaxis_title=trade_type_col,
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_white'
    )
        return fig

    @output
    @render_widget
    def trade_lineplot():
        return plot_trade_line_graph(input.select_country(), input.select_industry2(), trade_pdt_df)



app = App(app_ui, server)