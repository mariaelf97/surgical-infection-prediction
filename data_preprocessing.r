library(tidyverse)
library(data.table)

read_data<-function(URL)
{
  data <- fread(URL)%>%select(Year,County,Operative_Procedure,Facility_ID,Facility_Name,Hospital_Category_RiskAdjustment,Facility_Type,Procedure_Count,Infections_Predicted)
  return(data)
  
}

URL_list <- c("https://data.chhs.ca.gov/datastore/odata3.0/294eff96-096e-4b91-bff5-8de78fda318b",
              "https://data.chhs.ca.gov/datastore/odata3.0/2f0efc0b-9d44-4d8a-bd4d-ed4d77cb2a04",
              "https://data.chhs.ca.gov/datastore/odata3.0/e6828e3c-bf54-4991-865b-b4cc11719b8d",
              "https://data.chhs.ca.gov/datastore/odata3.0/075191c4-1754-47e8-96cb-575c764c84e3",
              "https://data.chhs.ca.gov/datastore/odata3.0/238ff746-71d9-4507-95d0-2a7b5146f4f5"
              )
SSI_adult_patients<- mapply(read_table, URL_list, SIMPLIFY = FALSE)
