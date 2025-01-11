# Time-Series-Forcasting-on-Web-Traffic 

Performed time-series analysis and forecasting on Google's web traffic data to forecast number of views of Wikipedia web page

  - Tech-Stack: - 

     AWS S3 , Python , Boto3 , pandas , os , Streamlit , Statsmodel , CI-CD Pipeline
    

## Project Description

 - - the lifecycle of the project is as shown - -
                   
    ![Project Structure](./images/diagram-export-1-11-2025-12_14_22-PM.png)

    1. Loading a RAW Data of Time Series into a S3 Bucket
       (Configure AWS CLI , IAM User/Permission etc etc)

    2. Modularly Coded a ETL Pipeline . The pipeline Structure is as Shown 
     
      Extracts Raw Data From S3 ---> Store in Local data/raw dir ---> Load and Transform the Data for Further Approach ---> Store transformed Data into data/processed dir

    3. ETS Decomposition the Data :
       Observed trend of the Data fluctuation are mostly consistent over time , So performed a Additive ETS

    4. Train the Data with 3 Different Model 
          ARIMA - Auto Regression Integrated Moving Average
          SARIMA - Seasonal ARIMA
          SARIMAX - Seasonal ARIMA with Exogenous Variable

    6. Hyperparameter Tuned the Entire Model
          selecting the best parametes for the individual models
        ![Hyperparameters](./images/Screenshot 2025-01-11 at 1.06.17 PM.png)


    7. Saved the Model in the form of Pickle file(if needed it also can be deployed in any cloud services)

    8. Streamlit Integration to Deploy the Model 

    9. Github Intergration:
        
        --> Deployed the Entire Model In Github 
        Though the Dataset was Not so Big So I didnt initialize DVC pipeline
        (For Big amount of Data We will track the Dataflow With DVC)

        --> An Automated CI/CD Pipeline is also Integrated for the automated tests and automated deployement

        > The CI-CD integration was a challenging Tasks For me bcz of the dependencies issue (Github considered Python 3.10 as 3.1 😅)



