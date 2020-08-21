# Time Series Modeling - Mod 4 Project
**Predicting Trends in Real Estate Value**

## Overview
Our group has been hired by a real estate investing company to find the best areas in the US to invest in real estate.  The investment company is only interested in capital gains on the homes it invests in after selling the homes after holding them for a 5 year investment period.

This project recommends 5 zipcodes in which the company should invest, and then predicts how much the homes in those zipcodes will appreciate in value over the next 5 years.

**The Questions**
* Where will real estate values increase the most?
* How much can we expect to make if we invest in those zipcodes?

### Readme Navigation:
[Data](https://github.com/jrf6xh/Time-Series-Modeling#data) -
[Selecting Zipcodes](https://github.com/jrf6xh/Time-Series-Modeling#selecting-zipcodes) -
[Model & Results](https://github.com/jrf6xh/Time-Series-Modeling#model) - 
[Recommendations](https://github.com/jrf6xh/Time-Series-Modeling#results) - 
[Limitations & Next Steps](https://github.com/jrf6xh/Time-Series-Modeling#limitations-and-next-steps) - 
[Project Information](https://github.com/jrf6xh/Time-Series-Modeling#project-information)

## Data
**Source:**
Our data is provided on a zipcode by zipcode basis by [Zillow](https://www.zillow.com/research/data/).

**Details:**
* 30,000 Zipcodes
* Median Home Value per Zipcode
* Dates from 1996 to 2019
* Data Limited to the USA

## Selecting Zipcodes
Before modeling and forecasting investment returns, we chose which zipcodes to model through Exploratory Data Analysis.

The factors we considered in our zipcode decisions are:
* Growth Since 1996
* Comparing Growth by Region
* Growth During the Recession

We chose to model the zipcodes with the highest growth since 1996 because of their consistent growth over time.  The choice of these zipcodes was reinforced by regional analysis.

All of these models are in California.  Four are located in the Los Angeles metro area, and one is located in Oakland.

**The Recommended Zipcodes:**
* 91108
* 90211
* 90027
* 94610
* 90048

## Model
In order to predict the 5 year return on our zipcodes, we used 5 ARIMA models (one for each zipcode) with an Auto Regressive term of 12, 2 orders of differencing, and a Moving Average term of 0.

## Results
Predictions accurately capture general trends, lending credibility to our forecasts.

![]()

![]()

![]()

![]()

![]()

## Limitations & Next Steps
**Limitations:**
* Accuracy of the model.
* The data for one zipcode is not stationary. This data will be made stationary in future iterations of this project.
* External factors were not considered.

**Next Steps:**

There are still further improvements that could be made to this project.

* Use additional models and compare accuracy.
* Adjust methods for choosing which zipcodes to model.
* Add external factors to the model.

## Project Information
**Contributors:** 
* [Jim Fay](https://github.com/jrf6xh)
* [Clair Marie Wholean](https://github.com/clairmarie8)

**Languages:** Python

**Libraries:** sklearn, statsmodels, pandas, matplotlib, numpy

**Duration:** August 17 - August 21, 2020

**Last Update:** August 21, 2020