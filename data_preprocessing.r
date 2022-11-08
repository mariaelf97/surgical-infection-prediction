library(tidyverse)
library(data.table)

read_data<-function(URL)
{
  data <- fread(URL)%>%select(Year,County,Operative_Procedure,Facility_ID,Facility_Name,Hospital_Category_RiskAdjustment,Facility_Type,Procedure_Count,Infections_Predicted)
  return(data)
  
}

URL_list <- c("https://data.chhs.ca.gov/dataset/f243090b-4c05-4c61-b614-7cb49b86b21d/resource/294eff96-096e-4b91-bff5-8de78fda318b/download/cdph_ssi_adult_odp_2021.csv",
              "https://data.chhs.ca.gov/dataset/f243090b-4c05-4c61-b614-7cb49b86b21d/resource/e6828e3c-bf54-4991-865b-b4cc11719b8d/download/cdph_ssi_adult_odp_2020h1.csv",
              "https://data.chhs.ca.gov/dataset/f243090b-4c05-4c61-b614-7cb49b86b21d/resource/eb324b4d-d2d2-41e8-aa53-f1b8380b4692/download/cdph_ssi_adult_odp_2020h2.csv",
              "https://data.chhs.ca.gov/dataset/f243090b-4c05-4c61-b614-7cb49b86b21d/resource/238ff746-71d9-4507-95d0-2a7b5146f4f5/download/cdph_ssi_adult_odp_2019.csv"
              
              )
SSI_adult_patients<- lapply(URL_list,read_data)
