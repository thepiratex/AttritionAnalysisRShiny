# Attrition Analysis using RShiny App

## Background
Employee Attrition impacts all businesses, irrespective of geography, industry and size of the company. Average attrition rate was 4.7% in August 2019. Attrition costs to company to post new jobs, hire employees and train them. It also reduces the quality of work if attrition is high.

## How does this app help?
*Our prediction system can develop certain “at risk” categories of employees.
*The tool can help HR managers understand important factors that drive the attrition rate
*The tool can Predict probabilities of attrition for each employee which can enable you address specific needs for that employee

## Analytics Perspective

*Problem formulated as a Classification Problem
*Classify whether the employee will leave the firm by analyzing the historical data.
*Assumption : Data unbiased and represents true nature of employee statistics.
*Metrics of success: AUC curve
*Data has 1400 observations only

## Features
* Mobile Friendly 
* Adapts to screen size on it's own
* Material Design
* Built using R-Shiny, therefore blazing fast

## Modeling
* We used multiple models to classifiy whether the employee will leave or stay in the firm. The baseline for XGB was higher than other models, therefore we chose to go forward with it and tuned it further for higher accuracy. 

## Screenshots
#### Frontpage Dashboard | iPad View
<img src="https://i.imgur.com/loGvOSu.png" width="500">

#### Sidebar | iPad View
<img src="https://i.imgur.com/tB5Cefk.png" width="500">

#### Data Analysis | iPad View 
<img src="https://i.imgur.com/yk73PXL.png" width="500">

#### Live Predicition | Desktop View
<img src="https://i.imgur.com/gSGcbyG.png" width="500">

#### Data Analysis | Desktop View
<img src="https://i.imgur.com/lpymR3B.png" width="500">

#### Model Info
<img src="https://i.imgur.com/DNpsXWt.png">

