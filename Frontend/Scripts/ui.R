
library(shinydashboard)
library(shiny)

header <- dashboardHeader(title = "Trade Relations of Singapore", titleWidth = 300)

sidebar <- dashboardSidebar(
  # Set up the sidebar menu
  sidebarMenu(
    # Main page
    menuItem("Introduction", tabName = "intro", icon = icon("user")),
    # Page for Line Graphs
    menuItem("line graphs", tabName = "linegraphs", icon = icon("chart-line")),
    # Page for Maps
    menuItem("maps", tabName = "maps", icon = icon("futbol"))
  )
)

body <- dashboardBody(tabItems(
  tabItem("intro", "Introduction of project, teach people how to use/ readme?"),
  tabItem("linegraphs",
          fluidRow(selectInput("country", "Choose a Trade Partner of SG",
                               choices = c("China", "Malaysia", "United States"))),
          fluidRow(plotOutput("tradePlot"))
          ),
  tabItem("maps", 
          fluidRow(box(title = "Ports of Top Trading Partners of Singapore",
                       leafletOutput("map", height = 700, width = 800)))
          )))


dashboardPage(header, sidebar, body)
