from shiny import App, ui, reactive, render
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shinywidgets
from shinywidgets import render_widget
from shinyswatch import theme
import numpy as np



#import trade product data
trade_pdt_df = pd.read_csv("data/cleaned data/10_years_trade_frontend.csv")

#product list for industry
product_list = sorted(trade_pdt_df["Product"].dropna().unique().tolist())

#Geopolitical distance data
geo_pol_df = pd.read_csv("data/cleaned data/geopolitical_data.csv")

# import intro text
def read_intro():
    with open("data\intro.txt", "r", encoding="utf-8") as f:
        return f.read()


# import functions
from shiny_functions import plot_trade_line_graph
from shiny_functions import plot_geopol_distance
from shiny_functions import plot_bubble
from shiny_functions import plot_geo_pol_line_graph
from Geopolitical_dist import get_geopolitical_data


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
                ui.output_ui("pg2_intro_text"),
                shinywidgets.output_widget("bubble_plot"),
                shinywidgets.output_widget("bar_plot")
            )
        ),

        ui.nav_panel(
            "Trade Data Across Time",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_selectize(
                        "select_country", "Select a Trade Partner:",
                        choices=["China", "Hong Kong","Indonesia", "Japan", "South Korea", "Malaysia", "Saudi Arabia", "Thailand", "United States"],
                        selected="China"
                    ),
                    ui.input_selectize(
                        "select_industry2", "Select an Industry:",
                        choices=product_list,
                        selected=product_list[0] if product_list else None
                    )
                ),
                ui.output_ui("pg3_intro_text"),
                shinywidgets.output_widget("trade_lineplot"),
                ui.output_text("trade_lineplot_text"),
                shinywidgets.output_widget("geo_pol_line_plot"),
                ui.output_text("geo_pol_line_plot_text")
            )
        ),
        title="TideTrackers",
        id="page"
    ),
    theme=theme.flatly() 
)


def server(input, output, session):

    @output
    @render.text
    def trade_lineplot_text():
        return "Fig 3: Line plot displaying level of imports/exports for specified trade partner per industry over the years"

    @output
    @render.text
    def geo_pol_line_plot_text():
        return "Fig 4: Line plot displaying geopolitical distance of Singapore with specified trade partner over the years"

    @output
    @render.ui
    def intro_text():
        text_content = read_intro().replace("\n", "<br>") 
        return ui.HTML(f"<p>{text_content}</p>")  # Display formatted text
    
    @output
    @render.ui
    def pg2_intro_text():
        return ui.HTML("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                <p><strong>Explore Singapore's imports and exports across major industries and trade partners between 2013 and 2023.</strong><br><br>
                Use the filters on the left to select an industry, trade type, and year to visualize historical trade volumes in the bubble plot.<br><br>
                The bar graph shows the Geopolitical Score with Singapore for the selected year. 
                <strong>A higher score indicates a greater geopolitical distance — i.e., less political and economic alignment with Singapore.</strong></p>
                
                <p style="margin-top: 20px; cursor: pointer; color: #007bff;" onclick="var x = document.getElementById('geo-explainer'); x.style.display = x.style.display === 'none' ? 'block' : 'none';">
                <strong>ⓘ What is Geopolitical Distance?</strong>
                </p>
                <div id="geo-explainer" style="display: none; background-color: #ffffff; padding: 10px; border-left: 4px solid #007bff; margin-top: 5px; border-radius: 6px;">
                    <p>Geopolitical Distance is a composite score reflecting how closely a country aligns with Singapore’s foreign policy and economic priorities.
                    It is computed using historical indicators and scaled as part of our modeling process.</p>
                    <p>The score incorporates:</p>
                    <ul style="padding-left: 20px; margin-top: 5px;">
                        <li>United Nations General Assembly (UNGA) voting similarity</li>
                        <li>Presence of Free Trade Agreements (FTAs)</li>
                        <li>Trade import and export volumes</li>
                        <li>Gross Domestic Product (GDP)</li>
                    </ul>
                    <p>Lower scores (shorter bars) indicate stronger alignment with Singapore.</p>
                </div>
            </div>
        """)

    @output
    @render_widget
    def bar_plot():
        return plot_geopol_distance(input.slide_year())

    @output
    @render_widget
    def bubble_plot():
        industry = input.select_industry1()
        trade_type = input.select_trade()
        year = input.slide_year()
        return plot_bubble(industry, trade_type, year, trade_pdt_df)
    
    #Pg 3

    @output
    @render.ui
    def pg3_intro_text():
        return ui.HTML("""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px;">
                <p><strong>Track how a country's geopolitical relationship with Singapore has changed over time.</strong><br><br>
                This line chart shows the evolution of the Geopolitical Distance score between Singapore and the selected country from 2013 to 2023.</p>

                <p style="margin-top: 20px; cursor: pointer; color: #007bff;" onclick="var x = document.getElementById('geo-trend-explainer'); x.style.display = x.style.display === 'none' ? 'block' : 'none';">
                <strong>ⓘ What does this score mean?</strong>
                </p>
                <div id="geo-trend-explainer" style="display: none; background-color: #ffffff; padding: 10px; border-left: 4px solid #007bff; margin-top: 5px; border-radius: 6px;">
                    <p>This score is computed based on multiple historical indicators, including:</p>
                    <ul style="padding-left: 20px;">
                        <li>UNGA voting patterns</li>
                        <li>Free Trade Agreements</li>
                        <li>Trade data with Singapore</li>
                        <li>GDP</li>
                    </ul>
                    <p>Each year's score is scaled for comparability and used in our modeling of trade relationships.</p>
                    <p><strong>A rising score</strong> indicates increasing geopolitical distance (weaker alignment), while <strong>a falling score</strong> suggests closer ties with Singapore.</p>
                </div>
            </div>
        """)


    @output
    @render_widget
    def trade_lineplot():
        return plot_trade_line_graph(input.select_country(), input.select_industry2(), trade_pdt_df)

    @output
    @render_widget
    def geo_pol_line_plot():
        country = input.select_country()
        return plot_geo_pol_line_graph(country)


app = App(app_ui, server)