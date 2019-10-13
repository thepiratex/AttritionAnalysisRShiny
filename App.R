
#############load the required packages################
library(shiny)
require(shinydashboard)
library(ggplot2)
library(dplyr)
library(magrittr)
library(caret)
library(gbm)
library(pROC)

######################### Loading Dataset ##########################
df <- read.csv("data.csv")
colnames(df)[1] <- 'Age'

######################### New Dataframe for further preprocessing #############################
model <- df
model[, 'Education'] <-
  factor(
    model[, 'Education'],
    levels = c('1', '2', '3', '4', '5'),
    labels = c('Below College', 'College', 'Bachelor', 'Master', 'Doctor'),
    ordered = TRUE
  )
model[, 'EnvironmentSatisfaction'] <-
  factor(
    model[, 'EnvironmentSatisfaction'],
    levels = c('1', '2', '3', '4'),
    ordered = TRUE,
    labels = c('Low', 'Medium', 'High', 'Very High')
  )
model[, 'JobInvolvement'] <-
  factor(
    model[, 'JobInvolvement'],
    levels = c('1', '2', '3', '4'),
    labels = c('Low', 'Medium', 'High', 'Very High'),
    ordered = TRUE
  )
head(model[, 'JobInvolvement'], n = 2)
model[, 'JobSatisfaction'] <-
  factor(
    model[, 'JobSatisfaction'],
    levels = c('1', '2', '3', '4'),
    labels = c('Low', 'Medium', 'High', 'Very High'),
    ordered = TRUE
  )
model[, 'PerformanceRating'] <-
  factor(
    model[, 'PerformanceRating'],
    levels = c('1', '2', '3', '4'),
    labels = c('Low', 'Good', 'Excellent', 'Outstanding'),
    ordered = TRUE
  )
model[, 'RelationshipSatisfaction'] <-
  factor(
    model[, 'RelationshipSatisfaction'],
    levels = c('1', '2', '3', '4'),
    labels = c('Low', 'Medium', 'High', 'Very High'),
    ordered = TRUE
  )
model[, 'WorkLifeBalance'] <-
  factor(
    model[, 'WorkLifeBalance'],
    levels = c('1', '2', '3', '4'),
    labels = c('Bad', 'Good', 'Better', 'Best'),
    ordered = TRUE
  )

model[, 'JobLevel'] <-
  factor(model[, 'JobLevel'],
         levels = c('1', '2', '3', '4', '5'),
         ordered = TRUE)

model %>% mutate_if(is.integer, as.numeric)
colnames(model)[2] <- 'y'

model <- model[c(
  "y",
  "Age",
  "BusinessTravel",
  "DailyRate",
  "Department",
  "DistanceFromHome",
  "Education",
  "EducationField",
  "EmployeeNumber",
  "EnvironmentSatisfaction",
  "Gender",
  "HourlyRate",
  "JobInvolvement",
  "JobLevel",
  "JobRole",
  "JobSatisfaction",
  "MaritalStatus",
  "MonthlyIncome",
  "MonthlyRate",
  "NumCompaniesWorked",
  "OverTime",
  "PercentSalaryHike",
  "PerformanceRating",
  "RelationshipSatisfaction",
  "StockOptionLevel",
  "TotalWorkingYears",
  "TrainingTimesLastYear",
  "WorkLifeBalance",
  "YearsAtCompany",
  "YearsInCurrentRole",
  "YearsSinceLastPromotion",
  "YearsWithCurrManager"
)]

model[, 'y'] <- (ifelse(model[, 'y'] == 'Yes', 1, 0))
model[, 'OverTime'] <- ifelse(model[, 'OverTime'] == 'Yes', 1, 0)
model[, 'Gender'] <- ifelse(model[, 'Gender'] == 'Male', 1, 0)

######### Variable to be used for training #########
model_train <- model[c(
  "Age",
  "BusinessTravel",
  "DailyRate",
  "Department",
  "DistanceFromHome",
  "Education",
  "EducationField",
  "EmployeeNumber",
  "EnvironmentSatisfaction",
  "Gender",
  "HourlyRate",
  "JobInvolvement",
  "JobLevel",
  "JobRole",
  "JobSatisfaction",
  "MaritalStatus",
  "MonthlyIncome",
  "MonthlyRate",
  "NumCompaniesWorked",
  "OverTime",
  "PercentSalaryHike",
  "PerformanceRating",
  "RelationshipSatisfaction",
  "StockOptionLevel",
  "TotalWorkingYears",
  "TrainingTimesLastYear",
  "WorkLifeBalance",
  "YearsAtCompany",
  "YearsInCurrentRole",
  "YearsSinceLastPromotion",
  "YearsWithCurrManager"
)]

############# One-Hot Encoding on training variables ##############
dummies <- dummyVars(~ ., data = model_train)
ex <- data.frame(predict(dummies, newdata = model_train))
names(ex) <- gsub("\\.", "", names(ex))
d <- cbind(model$y, ex)
names(d)[1] <- "y"
sum(model$y)

########### Find linear combinations and remove #################
comboInfo <- findLinearCombos(d)
comboInfo
# remove columns identified that led to linear combos
d <- d[,-comboInfo$remove]
# remove the "ones" column in the first column
d <- d[, c(2:ncol(d))]
y <- model$y
# Add the target variable back to our data.frame
d <- cbind(y, d)
rm(y, comboInfo)

##########Removing all values which have zero variability##########
nzv <- nearZeroVar(d, saveMetrics = TRUE)
head(nzv)
? nearZeroVar
#d <- d[, c(TRUE,!nzv$zeroVar[2:ncol(d)])]
d <- d[, c(TRUE, !nzv$nzv[2:ncol(d)])]
rm(nzv) #Cleaning R environment

preProcValues <- preProcess(d[, 2:ncol(d)], method = c("range"))
d <- predict(preProcValues, d)
# te set
rm(preProcValues)
sum(d$y)

##### Reading default values to predict for test model ########
df1 <- read.csv("test.csv")
names(df1)[1] <- "Age"

###################### Partition data into test-train ######################
set.seed(1234) # set a seed so you can replicate your results
inTrain <- createDataPartition(y = d$y,   # outcome variable
                               p = .90,   # % of training data you want
                               list = F)
# create your partitions
train <- d[inTrain, ]  # training data set
test <- d[-inTrain, ]
train[, 'y'] <- as.factor(ifelse(train[, 'y'] == 1, 'Yes', 'No'))
#train[,'y'] <- as.factor(train[,'y'])

####### Specify training control parameters ########
ctrl <- caret::trainControl(
  method = "cv",
  number = 5,
  summaryFunction = twoClassSummary,
  classProbs = TRUE
)


####### Train GBM model on dataset ########## 
gbmfit <- train(
  y ~ .,
  data = train,
  method = "gbm",
  verbose = FALSE,
  metric = "ROC",
  trControl = ctrl
)

gbmtrain <- predict(gbmfit, train, type = 'prob')
gbmpreds <- predict(gbmfit, test, type = 'prob')
gbmvalid <- predict(gbmfit, df1, type = "prob")

##### Evaluate ROC on train dataset #######
gbm.ROC.train <-
  pROC::roc(
    train$y,
    gbmtrain$Yes,
    ci = TRUE,
    ci.alpha = 0.9,
    # arguments for plot
    plot = TRUE,
    auc.polygon = TRUE,
    max.auc.polygon = TRUE,
    show.thres = TRUE,
    print.auc = TRUE
  )
##### Evaluate ROC on test dataset #######
gbm.ROC.test <-
  pROC::roc(
    test$y,
    gbmpreds$Yes,
    ci = TRUE,
    ci.alpha = 0.9,
    # arguments for plot
    plot = TRUE,
    auc.polygon = TRUE,
    max.auc.polygon = TRUE,
    show.thres = TRUE,
    print.auc = TRUE
  )

##################### R-Shiny App UI ###########################

#Dashboard header carrying the title of the dashboard
header <- dashboardHeader(title = "Employee Attrition")

#Sidebar content of the dashboard
sidebar <- dashboardSidebar(sidebarMenu(
  menuItem(
    "Dashboard",
    tabName = "dashboard",
    icon = icon("dashboard")
  ),
  menuItem(
    "Data Analysis",    tabName = "EDA",    icon = icon("calendar")
  ),
  menuItem("Predict!", tabName = "predict", icon = icon("book")),
  menuItem("Model Stats", tabName = "modelstats", icon = icon("book"))
))

tbs <- tabItems(
  # First tab content
  tabItem(
    tabName = "dashboard",
    fluidRow(
      valueBoxOutput("value1", width = 3)      ,
      valueBoxOutput("value2", width = 3)      ,
      valueBoxOutput("value3", width = 3)      ,
      valueBoxOutput("value4", width = 3)
      ),
    fluidRow(
      column(width=2),
      imageOutput("image_display")
    )
  ),
  ######################################################
  #### Second tab content #####
  tabItem(
    tabName = "EDA",
    fluidRow(
      box(
        title = "Attrition by Dept"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("attritionbydept", height = "300px")
      )
      
      ,
      box(
        title = "Overtime vs Attrition"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("attritionovertime", height = "300px")
      )
    ) #End of Fluid Row 1
    ,
    fluidRow(
      box(
        title = "Income vs Attrition"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("attritionincome", height = "300px")
      )
      ,
      box(
        title = "Environment Satisfaction vs Attrition"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("attritionenv", height = "300px")
      )
      
    ) # End of fluid row 2
    ,
    fluidRow(
      box(
        title = "Business Travel"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("attritionb", height = "300px")
      ),
      box(
        title = "Gender vs income"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("genderincome", height = "300px")
      )
    ) # End of fluid Row 3
    ,
    fluidRow(
      box(
        width = 12,
        title = "Distance from Home"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("distancehome", height = "300px")
      )
    ) #End of fluid row 4
    ,
    fluidRow(
      box(
        width = 6,
        title = "Gender vs Department"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("genderdept", height = "300px")
      )
      ,
      box(
        width = 6,
        title = "Gender vs Job Satisfaction"        ,
        status = "primary"        ,
        solidHeader = TRUE        ,
        collapsible = TRUE        ,
        plotOutput("gendersat", height = "300px")
      )
    ) #End of fluid row 4
    
  ),
  #######################################################
  
  tabItem(tabName = "modelstats",
          fluidRow(box(
            title = "Model Info (Stochastic Gradient Boosting)",
            width = 12,
            verbatimTextOutput("modelinfo")
          )), #END OF FLUID ROW 2
          fluidRow(
            box(
              title = "AUC Curve on Test Dataset",
              width = 6,
              plotOutput("modelteststats", height = "300px")
            ),
            box(
              title = "AUC Curve on Train Dataset",
              width = 6,
              plotOutput("modeltrainstats", height = "300px")
            )
          ) #END OF FLUID ROW 1
        ), 
  
  # Fourth tab content
  tabItem(tabName = "predict",
          fluidRow(
            box(
              selectInput("Overtime", "Pick Overtime",
                          c("Yes" = 1,
                            "No" = 0)),
              radioButtons(
                "btravel",
                "Business Travel",
                choices = c(
                  "Non Travel" = "A",
                  "Travel Frequently" = "B",
                  "Travel Rarely" = "C"
                )
              ),
              sliderInput("age", "Age ", 19, 60, 25),
              sliderInput("salary", "Monthly Income ", 1000, 20000, 10000),
              numericInput("yearsatcompany", "Years at Company", value=25, min=0, max=40),
              numericInput("totalworkingyears", "Total Work Ex", value=25, min=0, max=40)
              # numer
              # numericInput("", "Years at company", 0, 40, 25),
              # selectInput("yearsatcompany", "Years at company",
              #             c()),
            ),
            # valueBoxOutput("approvalBox",width = 6)
            infoBoxOutput("approvalBox", width = 6)
          ))
)


# combine the two fluid rows to make the body
body <- dashboardBody(tbs,
                      tags$head(
                        tags$link(rel = "stylesheet", type = "text/css", href = "custom.css")
                      ))

#completing the ui part with dashboardPage
ui <-
  dashboardPage(title = 'This is my Page title', header, sidebar, body, skin =
                  'green')

# create the server functions for the dashboard
server <- function(input, output) {
  #some data manipulation to derive the values of KPI boxes
  Average_Age <- mean(df$Age)
  # sales.account <- recommendation %>% group_by(Account) %>% summarise(value = sum(Revenue)) %>% filter(value==max(value))
  # prof.prod <- recommendation %>% group_by(Product) %>% summarise(value = sum(Revenue)) %>% filter(value==max(value))
  Median_Income <- median(df$MonthlyIncome)
  Dist_From_Home <- round(mean(df$DistanceFromHome),2)
  Total_Work_Exp <- round(mean(df$TotalWorkingYears),2)
  
  #creating the valueBoxOutput content
  output$value1 <- renderValueBox({
    valueBox(
      formatC(Median_Income, format = "d", big.mark = ',')
      ,
      paste('Median Income $', Median_Income)
      ,
      icon = icon("stats", lib = 'glyphicon')
      ,
      color = "purple",
      width = 3
    )
  })
  
  output$value2 <- renderValueBox({
    valueBox(
      formatC(Average_Age, format = "d", big.mark = ',')
      ,
      'Average Age'
      ,
      icon = icon("gbp", lib = 'glyphicon')
      ,
      color = "green",
      width = 3
    )
    
  })
  
  output$value3 <- renderValueBox({
    valueBox(
      formatC(Dist_From_Home, format = "d", big.mark = ',')
      ,
      paste('Distance From Home', Dist_From_Home,"mi")
      ,
      icon = icon("menu-hamburger", lib = 'glyphicon')
      ,
      color = "yellow",
      width = 3
    )
    
  })
  
  output$value4 <- renderValueBox({
    valueBox(
      formatC(Total_Work_Exp, format = "d", big.mark = ',')
      ,
      paste('Average Work Exp', Total_Work_Exp," years")
      ,
      icon = icon("menu-hamburger", lib = 'glyphicon')
      ,
      color = "blue"
    )
    
  })
  
  output$modelinfo <- renderPrint({
    gbmfit$bestTune
  })
  
  
  ###### EDA #######
  output$attritionbydept <- renderPlot({
    ggplot(df, aes(Department)) +
      geom_bar(aes(fill = Attrition)) + coord_flip()  +
      scale_fill_manual(values = c("#00bcd4", "#8bc34a"))
  })
  
  output$attritionovertime <- renderPlot({
    ggplot(df,
           aes(x = OverTime, group = Attrition)) +
      geom_bar(aes(y = ..prop.., fill = factor(..x..)),
               stat = "count",
               alpha = 0.7) +
      geom_text(aes(label = scales::percent(..prop..), y = ..prop..),
                stat = "count",
                vjust = -.5) +
      labs(y = "Percentage", fill = "OverTime") +
      facet_grid( ~ Attrition) +
      scale_fill_manual(values = c("#386cb0", "#fdb462")) +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5)) +
      ggtitle("Attrition")  +
      scale_fill_manual(values = c("#00bcd4", "#8bc34a"))
  })
  
  output$attritionincome <- renderPlot({
    ggplot(df,
           aes(x = MonthlyIncome, fill = Attrition)) +
      geom_density(alpha = 0.7) +
      scale_fill_manual(values = c("#00bcd4", "#8bc34a"))
    
  })
  
  output$attritionenv <- renderPlot({
    toplot <-
      df %>% select(EnvironmentSatisfaction, JobRole, Attrition) %>% group_by(JobRole, Attrition) %>%
      summarize(avg_envsatisfaction = mean(EnvironmentSatisfaction))
    
    ggplot(toplot, aes(x = JobRole, y = avg_envsatisfaction)) + geom_line(aes(group =
                                                                                Attrition), linetype = "dashed") +
      geom_point(aes(color = Attrition), size = 3)  +  scale_color_manual(values = c("Yes" = "#00bcd4", "No" = "#8bc34a")) +
      labs(title = "Working Environment", y = "Average Environment Satisfaction",  x =
             "Job Position") + theme(axis.text.x = element_text(angle = 90))
    
  })
  
  output$attritionb <- renderPlot({
    ggplot(df,
           aes(x = BusinessTravel,  group = Attrition)) +
      geom_bar(aes(y = ..prop.., fill = factor(..x..)),
               stat = "count",
               alpha = 0.7) +
      geom_text(aes(label = scales::percent(..prop..), y = ..prop..),
                stat = "count",
                vjust = -.5) +
      labs(y = "Percentage", fill = "Business Travel") +
      facet_grid( ~ Attrition) +
      #scale_y_continuous(labels=percent) +
      scale_fill_manual(values = c("#00bcd4", "#8bc34a", "#3f51b5")) +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5)) +
      ggtitle("Attrition")
    
    
  })
  
  output$hist <- renderText({
    df1['OverTime'] <- as.numeric(input$Overtime)
    df1['Age'] <- as.numeric(input$age) / 100
    df1['MonthlyIncome'] <- as.numeric(input$salary) / 20000
    if (input$btravel == 'A') {
      df1['BusinessTravelNonTravel'] <- 1
    } else if (input$btravel == 'B') {
      df1['BusinessTravelTravel_Frequently'] <- 1
    } else if (input$btravel == 'B') {
      df1['BusinessTravelTravel_Rarely'] <- 1
    }
    preds <- predict(gbmfit, df1, type = 'prob')
    result <- as.numeric(preds[[2]])
    result
  })
  
  output$image_display <- renderImage({
    list(
      src="featured.png"
    )
  })
  
  #####THE RED BOX OUTPUT####
  output$approvalBox <- renderInfoBox({
    df1['OverTime'] <- as.numeric(input$Overtime)
    df1['Age'] <- as.numeric(input$age) / 100
    df1['MonthlyIncome'] <- as.numeric(input$salary) / 20000
    if (input$btravel == 'A') {
      df1['BusinessTravelNonTravel'] <- 1
    } else if (input$btravel == 'B') {
      df1['BusinessTravelTravel_Frequently'] <- 1
    } else if (input$btravel == 'B') {
      df1['BusinessTravelTravel_Rarely'] <- 1
    }
    df1['YearsAtCompany'] <- as.numeric(input$yearsatcompany)/40
    df1['TotalWorkingYears'] <- as.numeric(input$totalworkingyears)/40
    preds <- predict(gbmfit, df1, type = 'prob')
    result <- round(as.numeric(preds[[2]]), 2)
    
    if (result >= 0.5) {
      result <- paste(result * 100, '%')
      infoBox(
        result,
        paste(
          'There is a',
          result,
          "probability the employee of age",
          input$age,
          "and salary $",
          input$salary,
          'will leave the firm'
        )
        ,
        icon = icon("thumbs-down", lib = "glyphicon"),
        color = "red",
        fill = TRUE
      )
      # valueBox(
      #   formatC(result, format="d", big.mark=',')
      #   ,paste('There is a',result,"probability the employee of age",input$age,"and salary $",input$salary, 'will leave the firm')
      #   ,icon = icon("thumbs-down",lib='glyphicon')
      #   ,color = "red")
    } else {
      result <- paste(result * 100, '%')
      valueBox(
        formatC(result, format = "d", big.mark = ',')
        ,
        paste(
          'There is a',
          result,
          "probability the employee of age",
          input$age,
          "and salary $",
          input$salary,
          'will leave the firm'
        )
        ,
        icon = icon("thumbs-up", lib = 'glyphicon')
        ,
        color = "green"
      )
    }
    
    
    
    
  })
  
  output$genderincome <- renderPlot({
    ggplot(df, aes(x = Gender, y = MonthlyIncome, fill = Attrition)) +
      geom_boxplot() + scale_fill_manual(values = c("#00bcd4", "#8bc34a"))
  })
  
  output$distancehome <- renderPlot({
    ggplot(
      df,
      aes(
        x = WorkLifeBalance,
        y = DistanceFromHome,
        group = WorkLifeBalance,
        fill = WorkLifeBalance
      )
    ) +
      geom_boxplot(alpha = 0.7) +
      facet_wrap( ~ Attrition) +
      ggtitle("Attrition") + scale_colour_manual(values = c('#00bcd4', '#3f51b5', '#8bc34a'))
    
  })
  
  output$modeltrainstats <- renderPlot({
    pROC::roc(
      train$y,
      gbmtrain$Yes,
      ci = TRUE,
      ci.alpha = 0.9,
      # arguments for plot
      plot = TRUE,
      auc.polygon = TRUE,
      max.auc.polygon = TRUE,
      show.thres = TRUE,
      print.auc = TRUE
    )
  })
  
  output$modelteststats <- renderPlot({
    pROC::roc(
      test$y,
      gbmpreds$Yes,
      ci = TRUE,
      ci.alpha = 0.9,
      # arguments for plot
      plot = TRUE,
      auc.polygon = TRUE,
      max.auc.polygon = TRUE,
      show.thres = TRUE,
      print.auc = TRUE
    )
    
  })
  
  
  output$genderdept <- renderPlot({
    ggplot(df, aes(Department)) + geom_bar(aes(fill = Attrition)) +
      facet_grid( ~ Gender) + coord_flip() + labs(y = "Number of Employees") +
      scale_fill_manual(values = c("#00bcd4", "#8bc34a", "#3f51b5"))
    
    
  })
  
  output$gendersat <- renderPlot({
    ggplot(df, aes(x = JobSatisfaction,  group = Gender)) + geom_bar(aes(y = ..prop.., fill = factor(..x..)),
                                                                     stat = "count",
                                                                     alpha = 0.7) +
      geom_text(aes(label = scales::percent(..prop..), y = ..prop..),
                stat = "count",
                vjust = -.5) +
      labs(y = "Percentage", fill = "JobSatisfaction") +
      facet_grid( ~ Gender) +
      scale_fill_manual(values = c("#00bcd4", "#8bc34a", "#fdb462", "#3f51b5")) +
      theme(legend.position = "none",
            plot.title = element_text(hjust = 0.5)) +
      ggtitle("Job Satisfaction on Gender basis")
    
    
  })
  
}


shinyApp(ui, server)
