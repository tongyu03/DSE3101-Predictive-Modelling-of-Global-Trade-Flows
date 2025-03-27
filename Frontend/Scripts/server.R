

library(shiny)

df <- read_csv("../data/en_SGP_AllYears_WITS_Trade_Summary.CSV") %>%
  rename(Type = `Indicator Type`)

# Define Function to generate Line graph
generate_trade_graph <- function(df, country) {
  df %>%
    filter(Partner == country) %>%
    filter(Indicator %in% c("Trade (US$ Mil)-Top 5 Export Partner", "Trade (US$ Mil)-Top 5 Import Partner")) %>%
    select(-`Product categories`, -Reporter, -Partner, -Indicator) %>%
    pivot_longer(`2021`:`1988`, names_to = "year", values_to = "Trade") %>%
    mutate(year = as.numeric(year)) %>% 
    filter(year >= 2002) %>%
    pivot_wider(names_from = Type, values_from = Trade) %>%
    rename(Exports = Export) %>%
    rename(Imports = Import) %>%
    ggplot(aes(x = year)) +  
    geom_line(aes(y = Exports, color = "Exports"), linewidth = 1) + 
    geom_line(aes(y = Imports, color = "Imports"), linewidth = 1) +
    scale_x_continuous(breaks = seq(2002, 2021, by = 2)) + 
    labs(title = paste("Trade Volume Between Singapore and", country),
         x = "Year",
         y = "Trade Value (US$ Mil)") +
    theme_minimal() +
    theme(legend.title = element_blank())
}

# Trading Ports
ports_data <- fromJSON("../data/ports.json") %>%
  filter(COUNTRY %in% c("Singapore", "Japan", "China", "U.S.A.", "Malaysia", "Thailand", "Saudi Arabia", "South Korea"))



# Define server logic required to draw a line graph
function(input, output, session) {
    output$tradePlot <- renderPlot({
      generate_trade_graph(df, input$country)  
      # Use input$country to generate the plot dynamically
    })
    
    # Render the map (non-reactive)
    output$map <- renderLeaflet({
      leaflet(data = ports_data) %>%
        addTiles() %>%
        addMarkers(~LONGITUDE, ~LATITUDE,
                   popup = ~paste(CITY, ", ", STATE, ", ", COUNTRY))
      
    })

}
