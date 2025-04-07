from shiny import App, ui, reactive, render
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shinywidgets
from shinywidgets import render_widget


# import intro text
def read_intro():
    with open("data\intro.txt", "r", encoding="utf-8") as f:
        return f.read()

## ui
app_ui = ui.page_navbar(  
    ui.nav_panel("Introduction",
                 ui.output_ui("intro_text")
                ),  
    ui.nav_panel("Historical Trade Data Across Countries",
                 ui.layout_sidebar(
                     ui.sidebar(
                        ui.input_selectize(
                            "select_industry1", "Select an Industry:", 
                            choices=["Manufacturing", "B", "C", "D", "E", "F", "G", "H"]
                        ),
                        ui.input_selectize(
                            "select_trade", "Select Imports or Exports:",
                            choices=["Exports", "Imports"], 
                        ),
                        ui.input_slider("slide_year", "Choose a Year:", 2009, 2024, value = 2020),
                     )
                )
    ),  
    ui.nav_panel("Trade Data Across Time",
                 ui.layout_sidebar(
                     ui.sidebar(
                        ui.input_selectize(
                            "select_country", "Select a Trade Partner:", 
                            choices=["China", "Hong Kong", "Japan", "South Korea", "Malaysia", "Saudi Arabia", "Thailand", "United States"], 
                            selected="China"  # Default selected value
                        ),
                        ui.input_selectize(
                            "select_industry2", "Select an Industry:", 
                            choices=["Manufacturing", "B", "C", "D", "E", "F", "G", "H"]
                        )
                     )
                )
    ),  
    title="TideTrackers",  
    id="page",  
)  


def server(input, output, session):

    @output
    @render.ui
    def intro_text():
        text_content = read_intro().replace("\n", "<br>") 
        return ui.HTML(f"<p>{text_content}</p>")  # Display formatted text


app = App(app_ui, server)