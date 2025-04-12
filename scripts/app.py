from shiny import App, ui, reactive, render
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import shinywidgets
from shinywidgets import render_widget
from shinyswatch import theme
import numpy as np
from shiny_functions import plot_trade_line_graph
from shiny_functions import plot_geopol_distance
from shiny_functions import plot_bubble
from shiny_functions import plot_geo_pol_line_graph

#Import datasets
trade_pdt_df = pd.read_csv("data/cleaned data/10_years_trade_frontend.csv")
geo_pol_df = pd.read_csv("data/cleaned data/geopolitical_data.csv")
trade_pred_df = pd.read_csv("data/cleaned data/trade_with_predicted.csv")


#product list for industry
product_list = sorted(trade_pdt_df["Product"].dropna().unique().tolist())

# import intro text
def read_intro():
    with open("data\intro.txt", "r", encoding="utf-8") as f:
        return f.read()



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
                        choices=["China","Indonesia", "Japan", "Malaysia", "Saudi Arabia","South Korea", "Thailand", "United States"],
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
                shinywidgets.output_widget("geo_pol_line_plot")
            )
        ),
        title="TideTrackers",
        id="page"
    ),
    theme=theme.flatly() 
)


def server(input, output, session):

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
                <p><strong>Track how a partner country's trade and geopolitical relationship with Singapore has changed over time.</strong><br><br>
                The first line chart shows historical trade volume with Singapore from 2013–2023.<br>
                <strong>It also includes a prediction of import and export volumes for the selected industry from 2024 to 2026.</strong></p>

                <p style="margin-top: 20px; cursor: pointer; color: #007bff;" onclick="var x = document.getElementById('geo-trend-explainer'); x.style.display = x.style.display === 'none' ? 'block' : 'none';">
                <strong>ⓘ How is this trade prediction made?</strong>
                </p>
                <div id="geo-trend-explainer" style="display: none; background-color: #ffffff; padding: 10px; border-left: 4px solid #007bff; margin-top: 5px; border-radius: 6px;">
                    <p>The 2024 prediction is generated using a multiple linear regression model trained on standardized historical data. The model considers:</p>
                    <ul style="padding-left: 20px;">
                        <li>Import and export volumes from the past 3 years</li>
                        <li>UN General Assembly (UNGA) voting alignment with Singapore</li>
                        <li>GDP data for both countries</li>
                        <li>Exchange rate with the partner country and its rate of change</li>
                    </ul>
                    <p>These features allow the model to capture both economic and geopolitical patterns that influence trade over time.</p>
                    <p>Predictions for 2025 and 2026 are extrapolated from the 2024 forecast, assuming that geopolitical and macroeconomic factors (e.g. UNGA alignment, FTAs) remain constant at 2023 levels due to data availability constraints.</p>
                    <p>If up-to-date data were available (e.g., through April 2025), the model could confidently predict trade flows up to 2026.</p>
                </div>
                <p><strong>Geopolitical Distance</strong> is also charted to show how bilateral alignment has evolved annually.<br>
                    A <strong>rising score</strong> reflects increasing geopolitical distance (weaker alignment), while a <strong>declining score</strong> indicates closer ties with Singapore over time.</p>
            </div>
        """)



    @output
    @render_widget
    def trade_lineplot():
        return plot_trade_line_graph(input.select_country(), input.select_industry2(), trade_pred_df)

    @output
    @render_widget
    def geo_pol_line_plot():
        country = input.select_country()
        return plot_geo_pol_line_graph(country)


app = App(app_ui, server)