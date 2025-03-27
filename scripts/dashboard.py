import seaborn as sns
import pandas as pd
import plotly.express as px
from shiny import App, ui, render_plot

### Historical Trade Data
df = pd.read_csv("en_SGP_AllYears_WITS_Trade_Summary.CSV")
df.rename(columns={"Indicator Type": "Type"}, inplace=True)


# Function to generate the trade graph
def generate_trade_graph(df, country):
    df_filtered = df[
        (df["Partner"] == country) & 
        (df["Indicator"].isin([
            "Trade (US$ Mil)-Top 5 Export Partner", "Trade (US$ Mil)-Top 5 Import Partner"
        ]))
    ]
    df_filtered = df_filtered.drop(columns=["Product categories", "Reporter", "Partner", "Indicator"])
    df_long = df_filtered.melt(id_vars=["Type"], var_name="year", value_name="Trade")
    df_long["year"] = pd.to_numeric(df_long["year"])
    df_long = df_long[df_long["year"] >= 2002]
    df_wide = df_long.pivot(index="year", columns="Type", values="Trade").reset_index()
    df_wide.rename(columns={"Export": "Exports", "Import": "Imports"}, inplace=True)
    # Plot using Plotly Express
    fig = px.line(
        df_wide, 
        x="year", 
        y=["Exports", "Imports"], 
        labels={"value": "Trade Value (US$ Mil)", "year": "Year"},
        title=f"Trade Volume Between Singapore and {country}",
        color_discrete_map={"Exports": "blue", "Imports": "red"}
    )
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(2002, 2022, 2))),
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
                          choices=["China", "Malaysia", "United States"],  # Options for the user to select
                          selected="China"  # Default selected value
                      ),
                        ui.output_plot("trade_plot"),  # Output plot will be rendered here
                    ),
        ui.nav_panel("Predicted Trade Volume", "model"),
        ui.nav_panel("Trading Ports", "interactive map"),
    )
)  


def server(input, output, session):
    @output
    @render_plotly
    def trade_plot():
        country = input.select_country  # Get the selected country dynamically 
        fig = generate_trade_graph(df, country)
        return fig  # Return the plot for rendering


app = App(app_ui, server)